# IPPO: independent per-agent PPO with local critic (arXiv:2011.09533).
# Algorithm spine from safepo.single_agent.ppo (ActorVCritic + VectorizedOnPolicyBuffer).
# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
# ==============================================================================

from __future__ import annotations

import copy
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import isaacgym
except ImportError:
    pass

import torch
import torch.nn as nn
from gymnasium.spaces import Box
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from safepo.common.buffer import VectorizedOnPolicyBuffer
from safepo.common.env import make_ma_isaac_env, make_ma_mujoco_env, make_ma_multi_goal_env
from safepo.common.lagrange import Lagrange
from safepo.common.logger import EpochLogger
from safepo.common.model import ActorVCritic
from safepo.utils.config import (
    isaac_gym_map,
    multi_agent_args,
    multi_agent_goal_tasks,
    multi_agent_velocity_map,
    parse_sim_params,
    set_np_formatting,
    set_seed,
)


def _cfg_train_to_ppo_config(cfg_train: dict[str, Any]) -> dict[str, Any]:
    n_env = int(cfg_train.get("n_rollout_threads", 1))
    ep_len = int(cfg_train.get("episode_length", 1000))
    steps_per_epoch = ep_len * n_env
    hidden = int(cfg_train.get("hidden_size", 64))
    # MAPPO-style: split buffer into num_mini_batch chunks (ignore SA batch_size).
    num_mini_batch = max(1, int(cfg_train.get("num_mini_batch", 1)))
    ent = cfg_train.get("ent_coef", cfg_train.get("entropy_coef", 0.0))
    return {
        "steps_per_epoch": steps_per_epoch,
        "local_steps_per_epoch": ep_len,
        "num_envs": n_env,
        "num_env_steps": int(cfg_train.get("num_env_steps", steps_per_epoch)),
        "gamma": float(cfg_train.get("gamma", 0.99)),
        "lam": float(cfg_train.get("gae_lambda", 0.95)),
        "lam_c": float(cfg_train.get("gae_lambda", 0.95)),
        "clip_ratio": float(cfg_train.get("clip_param", 0.2)),
        "ent_coef": float(ent),
        "hidden_sizes": list(cfg_train.get("hidden_sizes", [hidden, hidden])),
        "learning_iters": int(cfg_train.get("learning_iters", 10)),
        "target_kl": float(cfg_train.get("target_kl", 0.05)),
        "max_grad_norm": float(cfg_train.get("max_grad_norm", 40.0)),
        "num_mini_batch": num_mini_batch,
        "actor_lr": float(cfg_train.get("actor_lr", 3e-4)),
        "critic_lr": float(cfg_train.get("critic_lr", 3e-4)),
        "lr_end_factor": float(cfg_train.get("lr_end_factor", 1.0)),
        "use_critic_norm": bool(cfg_train.get("use_critic_norm", True)),
        "use_value_coefficient": bool(cfg_train.get("use_value_coefficient", False)),
        "cost_limit": float(cfg_train.get("cost_limit", 0.0)),
        "lagrangian_multiplier_init": float(
            cfg_train.get("lagrangian_multiplier_init", 0.001)
        ),
        "lagrangian_multiplier_lr": float(
            cfg_train.get("lagrangian_multiplier_lr", 0.035)
        ),
        "share_policy": bool(cfg_train.get("share_policy", True)),
        "save_interval": int(cfg_train.get("save_interval", 10)),
        "use_eval": bool(cfg_train.get("use_eval", False)),
        "eval_episodes": int(cfg_train.get("eval_episodes", 25)),
        "eval_interval": int(cfg_train.get("eval_interval", 0)),
        "seed": int(cfg_train.get("seed", 0)),
        "device": cfg_train.get("device", "cpu"),
    }


def _box_spaces(obs_dim: int, act_dim: int) -> tuple[Box, Box]:
    return (
        Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32),
    )


@dataclass
class AgentPPOBundle:
    policy: ActorVCritic
    buffer: VectorizedOnPolicyBuffer
    actor_optimizer: torch.optim.Optimizer
    reward_critic_optimizer: torch.optim.Optimizer
    cost_critic_optimizer: torch.optim.Optimizer
    actor_scheduler: LinearLR | None = None
    lagrange: Lagrange | None = None
    cost_deque: deque | None = None


def _build_bundles(
    num_agents: int,
    obs_dim: int,
    act_dim: int,
    *,
    share_policy: bool,
    device: torch.device,
    ppo_cfg: dict[str, Any],
    use_lagrange: bool,
    epochs: int,
) -> list[AgentPPOBundle]:
    obs_space, act_space = _box_spaces(obs_dim, act_dim)
    bundles: list[AgentPPOBundle] = []
    shared_policy: ActorVCritic | None = None
    shared_opts: tuple | None = None
    shared_sched: LinearLR | None = None
    shared_lagrange: Lagrange | None = None

    if use_lagrange and share_policy:
        shared_lagrange = Lagrange(
            cost_limit=ppo_cfg["cost_limit"],
            lagrangian_multiplier_init=ppo_cfg["lagrangian_multiplier_init"],
            lagrangian_multiplier_lr=ppo_cfg["lagrangian_multiplier_lr"],
        )

    for _agent_id in range(num_agents):
        if share_policy and shared_policy is not None:
            policy = shared_policy
            actor_opt, rew_opt, cost_opt = shared_opts  # type: ignore[misc]
            sched = shared_sched
            lagrange = shared_lagrange
        else:
            policy = ActorVCritic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_sizes=ppo_cfg["hidden_sizes"],
            ).to(device)
            actor_opt = torch.optim.Adam(
                policy.actor.parameters(), lr=ppo_cfg["actor_lr"]
            )
            rew_opt = torch.optim.Adam(
                policy.reward_critic.parameters(), lr=ppo_cfg["critic_lr"]
            )
            cost_opt = torch.optim.Adam(
                policy.cost_critic.parameters(), lr=ppo_cfg["critic_lr"]
            )
            # torch>=2.7 dropped LinearLR(verbose=...); do not pass it.
            sched = LinearLR(
                actor_opt,
                start_factor=1.0,
                end_factor=ppo_cfg["lr_end_factor"],
                total_iters=max(epochs, 1),
            )
            lagrange = None
            if use_lagrange and not share_policy:
                lagrange = Lagrange(
                    cost_limit=ppo_cfg["cost_limit"],
                    lagrangian_multiplier_init=ppo_cfg["lagrangian_multiplier_init"],
                    lagrangian_multiplier_lr=ppo_cfg["lagrangian_multiplier_lr"],
                )
            if share_policy:
                shared_policy = policy
                shared_opts = (actor_opt, rew_opt, cost_opt)
                shared_sched = sched

        buffer = VectorizedOnPolicyBuffer(
            obs_space=obs_space,
            act_space=act_space,
            size=ppo_cfg["local_steps_per_epoch"],
            device=device,
            num_envs=ppo_cfg["num_envs"],
            gamma=ppo_cfg["gamma"],
            lam=ppo_cfg["lam"],
            lam_c=ppo_cfg["lam_c"],
        )
        bundles.append(
            AgentPPOBundle(
                policy=policy,
                buffer=buffer,
                actor_optimizer=actor_opt,
                reward_critic_optimizer=rew_opt,
                cost_critic_optimizer=cost_opt,
                actor_scheduler=sched,
                lagrange=lagrange,
                cost_deque=deque(maxlen=50)
                if (use_lagrange and not share_policy)
                else None,
            )
        )
    return bundles


def _merge_buffer_data(datas: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate per-agent buffer.get() dicts along the batch dim."""
    keys = datas[0].keys()
    return {k: torch.cat([d[k] for d in datas], dim=0) for k in keys}


def _should_record_episode(
    *,
    done: bool,
    epoch_end: bool,
    episode_steps: float,
    rollout_horizon: int,
) -> bool:
    """Whether to append this env's episode stats to the rolling train deque.

    Record on natural termination (success, truncation, or env done). Also record
    when a single episode fills the full rollout horizon without a done signal
    (timeout-at-horizon). Do **not** record mid-episode rollout cuts — those
    episodes continue in the next epoch and are logged once when they end.
    """
    if done:
        return True
    return epoch_end and episode_steps >= float(rollout_horizon)


def _actor_param_delta_norm(policy: ActorVCritic, before: list[torch.Tensor]) -> float:
    after = [p.detach() for p in policy.actor.parameters()]
    return float(
        torch.sqrt(sum((a - b).pow(2).sum() for a, b in zip(after, before))).item()
    )


def _ppo_update_agent(
    bundle: AgentPPOBundle,
    ppo_cfg: dict[str, Any],
    logger: EpochLogger,
    *,
    use_lagrange: bool,
    data: dict[str, torch.Tensor] | None = None,
) -> tuple[int, float]:
    if data is None:
        data = bundle.buffer.get()
    actor_params_before = [
        p.detach().clone() for p in bundle.policy.actor.parameters()
    ]
    batch_steps = int(data["obs"].shape[0])
    if batch_steps <= 0:
        return 0, 0.0

    num_mini_batch = max(1, int(ppo_cfg.get("num_mini_batch", 1)))
    if batch_steps < num_mini_batch:
        raise ValueError(
            f"IPPO needs buffer length ({batch_steps}) >= num_mini_batch ({num_mini_batch}) "
            "(MAPPO-style split)."
        )
    # Same as SeparatedReplayBuffer.feed_forward_generator: floor-divide, drop remainder.
    mini_batch_size = batch_steps // num_mini_batch

    old_distribution = bundle.policy.actor(data["obs"])
    advantage = data["adv_r"]
    if use_lagrange and bundle.lagrange is not None:
        lam = bundle.lagrange.lagrangian_multiplier
        advantage = (data["adv_r"] - lam * data["adv_c"]) / (lam + 1)

    dataloader = DataLoader(
        TensorDataset(
            data["obs"],
            data["act"],
            data["log_prob"],
            data["target_value_r"],
            data["target_value_c"],
            advantage,
        ),
        batch_size=mini_batch_size,
        shuffle=True,
        drop_last=True,
    )
    update_counts = 0
    grad_steps = 0
    final_kl = 0.0
    ent_coef = float(ppo_cfg["ent_coef"])
    clip = float(ppo_cfg["clip_ratio"])

    for _ in range(ppo_cfg["learning_iters"]):
        for obs_b, act_b, log_prob_b, target_r_b, target_c_b, adv_b in dataloader:
            bundle.reward_critic_optimizer.zero_grad()
            loss_r = nn.functional.mse_loss(
                bundle.policy.reward_critic(obs_b), target_r_b
            )
            bundle.cost_critic_optimizer.zero_grad()
            loss_c = nn.functional.mse_loss(
                bundle.policy.cost_critic(obs_b), target_c_b
            )
            if ppo_cfg.get("use_critic_norm", True):
                for param in bundle.policy.reward_critic.parameters():
                    loss_r = loss_r + param.pow(2).sum() * 0.001
                for param in bundle.policy.cost_critic.parameters():
                    loss_c = loss_c + param.pow(2).sum() * 0.001

            distribution = bundle.policy.actor(obs_b)
            log_prob = distribution.log_prob(act_b).sum(dim=-1)
            entropy = distribution.entropy().sum(dim=-1).mean()
            ratio = torch.exp(log_prob - log_prob_b)
            ratio_clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
            loss_pi = -torch.min(ratio * adv_b, ratio_clipped * adv_b).mean()
            loss_pi = loss_pi - ent_coef * entropy

            bundle.actor_optimizer.zero_grad()
            total_loss = (
                loss_pi + 2 * loss_r + loss_c
                if ppo_cfg.get("use_value_coefficient", False)
                else loss_pi + loss_r + loss_c
            )
            total_loss.backward()
            clip_grad_norm_(bundle.policy.parameters(), ppo_cfg["max_grad_norm"])
            bundle.reward_critic_optimizer.step()
            bundle.cost_critic_optimizer.step()
            bundle.actor_optimizer.step()
            grad_steps += 1

            logger.store(
                **{
                    "Loss/Loss_reward_critic": loss_r.mean().item(),
                    "Loss/Loss_cost_critic": loss_c.mean().item(),
                    "Loss/Loss_actor": loss_pi.mean().item(),
                    "Misc/Entropy": entropy.item(),
                }
            )

        new_distribution = bundle.policy.actor(data["obs"])
        final_kl = (
            torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
            .sum(-1, keepdim=True)
            .mean()
            .item()
        )
        update_counts += 1
        if final_kl > ppo_cfg["target_kl"]:
            break

    if bundle.actor_scheduler is not None:
        bundle.actor_scheduler.step()

    param_delta = _actor_param_delta_norm(bundle.policy, actor_params_before)
    logger.store(
        **{
            "Train/ActorParamDelta": param_delta,
            "Train/PPOBatchSteps": float(batch_steps),
            "Train/PPOGradSteps": float(grad_steps),
            "Train/NumMiniBatch": float(num_mini_batch),
            "Train/MiniBatchSize": float(mini_batch_size),
        }
    )
    return update_counts, final_kl


def _save_checkpoints(
    bundles: list[AgentPPOBundle], save_dir: str, *, share_policy: bool
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    # share_policy → one actor file (actor_agent0.pt); not a bug — both agents share nets.
    agents = range(1) if share_policy else range(len(bundles))
    for agent_id in agents:
        torch.save(
            bundles[agent_id].policy.actor.state_dict(),
            os.path.join(save_dir, f"actor_agent{agent_id}.pt"),
        )


def _restore_checkpoints(
    bundles: list[AgentPPOBundle],
    model_dir: str,
    device: torch.device,
    *,
    share_policy: bool,
) -> None:
    for agent_id, bundle in enumerate(bundles):
        aid = 0 if share_policy else agent_id
        path = os.path.join(model_dir, f"actor_agent{aid}.pt")
        bundle.policy.actor.load_state_dict(
            torch.load(path, map_location=device, weights_only=False)
        )


def build_ma_envs(args, cfg_train, cfg_env=None):
    agent_index = [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]]
    if args.task in multi_agent_velocity_map:
        env = make_ma_mujoco_env(
            scenario=args.scenario,
            agent_conf=args.agent_conf,
            seed=args.seed,
            cfg_train=cfg_train,
        )
        cfg_eval = copy.deepcopy(cfg_train)
        cfg_eval["seed"] = args.seed + 10000
        cfg_eval["n_rollout_threads"] = cfg_eval["n_eval_rollout_threads"]
        eval_env = make_ma_mujoco_env(
            scenario=args.scenario,
            agent_conf=args.agent_conf,
            seed=cfg_eval["seed"],
            cfg_train=cfg_eval,
        )
    elif args.task in isaac_gym_map:
        sim_params = parse_sim_params(args, cfg_env, cfg_train)
        env = make_ma_isaac_env(args, cfg_env, cfg_train, sim_params, agent_index)
        cfg_train["n_rollout_threads"] = env.num_envs
        cfg_train["n_eval_rollout_threads"] = env.num_envs
        eval_env = env
    elif args.task in multi_agent_goal_tasks:
        env = make_ma_multi_goal_env(task=args.task, seed=args.seed, cfg_train=cfg_train)
        need_eval = (
            cfg_train.get("use_eval")
            or getattr(args, "model_dir", "")
            or int(cfg_train.get("eval_interval", 0)) > 0
        )
        if need_eval:
            cfg_eval = copy.deepcopy(cfg_train)
            cfg_eval["seed"] = args.seed + 10000
            cfg_eval["n_rollout_threads"] = cfg_eval["n_eval_rollout_threads"]
            eval_env = make_ma_multi_goal_env(
                task=args.task, seed=args.seed + 10000, cfg_train=cfg_eval
            )
        else:
            eval_env = None
    else:
        raise NotImplementedError(f"Unsupported MA task: {args.task}")
    return env, eval_env


class Runner:
    """Independent per-agent IPPO on a Share* vec env (PPO spine)."""

    def __init__(
        self,
        vec_env,
        vec_eval_env,
        cfg_train: dict[str, Any],
        model_dir: str = "",
        *,
        use_lagrange: bool = False,
    ) -> None:
        self.envs = vec_env
        self.eval_envs = vec_eval_env
        self.use_lagrange = use_lagrange
        self.num_agents = vec_env.num_agents
        self.ppo_cfg = _cfg_train_to_ppo_config(cfg_train)
        self.share_policy = self.ppo_cfg["share_policy"]
        self.device = torch.device(cfg_train["device"])

        total_steps = self.ppo_cfg["num_env_steps"]
        steps_per_epoch = self.ppo_cfg["steps_per_epoch"]
        self.epochs = max(1, total_steps // steps_per_epoch)

        obs_dim = int(vec_env.observation_space[0].shape[0])
        act_dim = int(vec_env.action_space[0].shape[0])
        self.bundles = _build_bundles(
            self.num_agents,
            obs_dim,
            act_dim,
            share_policy=self.share_policy,
            device=self.device,
            ppo_cfg=self.ppo_cfg,
            use_lagrange=use_lagrange,
            epochs=self.epochs,
        )

        self.logger = EpochLogger(
            log_dir=cfg_train["log_dir"],
            seed=str(cfg_train["seed"]),
        )
        self.save_dir = os.path.join(
            cfg_train["log_dir"], f"models_seed{cfg_train['seed']}"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        cfg_for_log = dict(cfg_train)
        nmb = int(self.ppo_cfg["num_mini_batch"])
        steps = int(self.ppo_cfg["steps_per_epoch"])
        cfg_for_log["num_agents"] = self.num_agents
        cfg_for_log["num_mini_batch"] = nmb
        cfg_for_log["num_mini_batches"] = nmb
        cfg_for_log["mini_batch_size_per_agent"] = steps // nmb
        if self.share_policy:
            cfg_for_log["mini_batch_size_shared_merge"] = (
                steps * self.num_agents
            ) // nmb
        self.logger.save_config(cfg_for_log)

        self.shared_lagrange = (
            self.bundles[0].lagrange
            if use_lagrange and self.share_policy
            else None
        )

        if model_dir:
            _restore_checkpoints(
                self.bundles, model_dir, self.device, share_policy=self.share_policy
            )

        self.rew_deque: deque = deque(maxlen=50)
        self.cost_deque: deque = deque(maxlen=50)
        self.len_deque: deque = deque(maxlen=50)

    def _as_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _record_episode_metrics(self, env_idx: int) -> None:
        ret = float(self._ep_ret[env_idx])
        cost = float(self._ep_cost[env_idx])
        length = float(self._ep_len[env_idx])
        self.rew_deque.append(ret)
        self.cost_deque.append(cost)
        self.len_deque.append(length)
        self.logger.store(
            **{
                "Metrics/EpRet": ret,
                "Metrics/EpCost": cost,
                "Metrics/EpLen": length,
            }
        )
        if self.use_lagrange and not self.share_policy:
            for a, bundle in enumerate(self.bundles):
                if bundle.cost_deque is not None:
                    bundle.cost_deque.append(self._agent_ep_cost[env_idx, a])
        self._ep_ret[env_idx] = 0.0
        self._ep_cost[env_idx] = 0.0
        self._ep_len[env_idx] = 0.0
        self._agent_ep_cost[env_idx, :] = 0.0

    def _finish_paths(
        self,
        env_idx: int,
        obs: torch.Tensor,
        *,
        epoch_end: bool,
        done: bool,
    ) -> None:
        # obs: [n_envs, n_agents, obs_dim] — index per agent (not all agents).
        # Mid-epoch mission success (done, not epoch_end) → zero bootstrap.
        # Epoch cut / time-limit (epoch_end) → value bootstrap (MA dones merge trunc).
        for agent_id, bundle in enumerate(self.bundles):
            last_r = torch.zeros(1, device=self.device)
            last_c = torch.zeros(1, device=self.device)
            need_bootstrap = epoch_end or (not done)
            if done and not epoch_end:
                need_bootstrap = False
            if need_bootstrap:
                with torch.no_grad():
                    _, _, last_r, last_c = bundle.policy.step(
                        obs[env_idx, agent_id], deterministic=False
                    )
                last_r = last_r.unsqueeze(0)
                last_c = last_c.unsqueeze(0)
            bundle.buffer.finish_path(
                last_value_r=last_r, last_value_c=last_c, idx=env_idx
            )

    def run(self) -> None:
        local_steps = self.ppo_cfg["local_steps_per_epoch"]
        n_envs = self.ppo_cfg["num_envs"]
        start = time.time()

        obs, _, _ = self.envs.reset()
        obs = self._as_tensor(obs)
        self._ep_ret = np.zeros(n_envs, dtype=np.float64)
        self._ep_cost = np.zeros(n_envs, dtype=np.float64)
        self._ep_len = np.zeros(n_envs, dtype=np.float64)
        self._agent_ep_cost = np.zeros((n_envs, self.num_agents), dtype=np.float64)

        for epoch in tqdm(
            range(self.epochs),
            desc="IPPO",
            unit="update",
            total=self.epochs,
            file=sys.stderr,
            dynamic_ncols=True,
        ):
            for step in range(local_steps):
                actions_collector = []
                store_batch = []

                for agent_id, bundle in enumerate(self.bundles):
                    obs_agent = obs[:, agent_id]
                    with torch.no_grad():
                        act, log_prob, value_r, value_c = bundle.policy.step(
                            obs_agent, deterministic=False
                        )
                    actions_collector.append(act)
                    store_batch.append(
                        (obs_agent, act, log_prob, value_r, value_c)
                    )

                obs, _, rewards, costs, dones, _infos, _ = self.envs.step(
                    actions_collector
                )
                rewards_t = self._as_tensor(rewards)
                costs_t = self._as_tensor(costs)
                dones_t = self._as_tensor(dones)

                reward_env = torch.mean(rewards_t, dim=1).flatten()
                cost_env = torch.mean(costs_t, dim=1).flatten()
                self._ep_ret += reward_env.detach().cpu().numpy()
                self._ep_cost += cost_env.detach().cpu().numpy()
                self._ep_len += 1.0
                for a in range(self.num_agents):
                    self._agent_ep_cost[:, a] += (
                        costs_t[:, a].flatten().detach().cpu().numpy()
                    )

                for agent_id, bundle in enumerate(self.bundles):
                    o, act, lp, vr, vc = store_batch[agent_id]
                    bundle.buffer.store(
                        obs=o,
                        act=act,
                        reward=rewards_t[:, agent_id].flatten(),
                        cost=costs_t[:, agent_id].flatten(),
                        value_r=vr,
                        value_c=vc,
                        log_prob=lp,
                    )

                obs = self._as_tensor(obs)
                epoch_end = step >= local_steps - 1
                dones_env = torch.all(dones_t, dim=1).flatten()

                for env_idx in range(n_envs):
                    done = bool(dones_env[env_idx].item())
                    if epoch_end or done:
                        self._finish_paths(
                            env_idx, obs, epoch_end=epoch_end, done=done
                        )
                        if _should_record_episode(
                            done=done,
                            epoch_end=epoch_end,
                            episode_steps=self._ep_len[env_idx],
                            rollout_horizon=local_steps,
                        ):
                            self._record_episode_metrics(env_idx)

            eval_rew = eval_cost = eval_len = 0.0
            eval_iv = int(self.ppo_cfg.get("eval_interval", 0))
            run_eval = self.eval_envs is not None and (
                self.ppo_cfg["use_eval"]
                or (
                    eval_iv > 0
                    and (epoch % eval_iv == 0 or epoch == self.epochs - 1)
                )
            )
            if run_eval:
                eval_rew, eval_cost, eval_len = self._eval(deterministic=True)

            if self.use_lagrange:
                if self.shared_lagrange is not None and self.cost_deque:
                    self.shared_lagrange.update_lagrange_multiplier(
                        float(np.mean(self.cost_deque))
                    )
                else:
                    for bundle in self.bundles:
                        if bundle.lagrange and bundle.cost_deque:
                            bundle.lagrange.update_lagrange_multiplier(
                                float(np.mean(bundle.cost_deque))
                            )

            stop_iter = 0
            total_kl = 0.0
            # share_policy=True: one shared ActorVCritic + optimizers. Must NOT run
            # independent PPO updates per agent (2nd update uses stale log_probs vs
            # already-updated weights → broken ratios / fake learning signal).
            if self.share_policy:
                datas = [b.buffer.get() for b in self.bundles]
                merged = _merge_buffer_data(datas)
                iters, kl = _ppo_update_agent(
                    self.bundles[0],
                    self.ppo_cfg,
                    self.logger,
                    use_lagrange=self.use_lagrange,
                    data=merged,
                )
                stop_iter, total_kl = iters, kl
            else:
                for bundle in self.bundles:
                    iters, kl = _ppo_update_agent(
                        bundle,
                        self.ppo_cfg,
                        self.logger,
                        use_lagrange=self.use_lagrange,
                    )
                    stop_iter = max(stop_iter, iters)
                    total_kl = max(total_kl, kl)

            for key in ("Metrics/EpRet", "Metrics/EpCost", "Metrics/EpLen"):
                if key not in self.logger.epoch_dict or len(self.logger.epoch_dict[key]) == 0:
                    self.logger.store(**{key: 0.0})

            self.logger.store(
                **{
                    "Eval/EpRet": eval_rew,
                    "Eval/EpCost": eval_cost,
                    "Eval/EpLen": eval_len,
                }
            )

            if epoch % self.ppo_cfg["save_interval"] == 0 or epoch == self.epochs - 1:
                _save_checkpoints(
                    self.bundles, self.save_dir, share_policy=self.share_policy
                )

            end = time.time()
            total_steps = (epoch + 1) * self.ppo_cfg["steps_per_epoch"]
            self.logger.log_tabular("Metrics/EpRet", min_and_max=True, std=True)
            self.logger.log_tabular("Metrics/EpCost", min_and_max=True, std=True)
            self.logger.log_tabular("Metrics/EpLen", min_and_max=True, std=True)
            self.logger.log_tabular("Eval/EpRet")
            self.logger.log_tabular("Eval/EpCost")
            self.logger.log_tabular("Eval/EpLen")
            self.logger.log_tabular("Train/Epoch", epoch)
            self.logger.log_tabular("Train/TotalSteps", total_steps)
            self.logger.log_tabular("Train/StopIter", stop_iter)
            self.logger.log_tabular("Train/KL", total_kl)
            self.logger.log_tabular("Train/ActorParamDelta")
            self.logger.log_tabular("Train/PPOBatchSteps")
            self.logger.log_tabular("Train/PPOGradSteps")
            self.logger.log_tabular("Train/NumMiniBatch")
            self.logger.log_tabular("Train/MiniBatchSize")
            self.logger.log_tabular("Loss/Loss_reward_critic")
            self.logger.log_tabular("Loss/Loss_cost_critic")
            self.logger.log_tabular("Loss/Loss_actor")
            self.logger.log_tabular("Misc/Entropy")
            self.logger.log_tabular("Time/Total", end - start)
            self.logger.log_tabular("Time/FPS", int(total_steps / max(end - start, 1e-6)))
            self.logger.dump_tabular()

    def _eval(self, *, deterministic: bool = True) -> tuple[float, float, float]:
        target = max(1, int(self.ppo_cfg["eval_episodes"]))
        eval_env = self.eval_envs
        obs, _, _ = eval_env.reset()
        obs = self._as_tensor(obs)
        n_eval = int(obs.shape[0])
        eval_rews: list[float] = []
        eval_costs: list[float] = []
        eval_lens: list[float] = []
        ep_rew = np.zeros(n_eval)
        ep_cost = np.zeros(n_eval)
        ep_len = np.zeros(n_eval)
        completed = 0

        while completed < target:
            actions = []
            for agent_id, bundle in enumerate(self.bundles):
                with torch.no_grad():
                    act, _, _, _ = bundle.policy.step(
                        obs[:, agent_id], deterministic=deterministic
                    )
                actions.append(act)
            obs, _, rewards, costs, dones, _, _ = eval_env.step(actions)
            rewards_t = self._as_tensor(rewards)
            costs_t = self._as_tensor(costs)
            dones_t = self._as_tensor(dones)
            ep_rew += torch.mean(rewards_t, dim=1).flatten().detach().cpu().numpy()
            ep_cost += torch.mean(costs_t, dim=1).flatten().detach().cpu().numpy()
            ep_len += 1.0
            obs = self._as_tensor(obs)
            dones_env = torch.all(dones_t, dim=1).flatten()
            for i in range(n_eval):
                if dones_env[i]:
                    eval_rews.append(ep_rew[i])
                    eval_costs.append(ep_cost[i])
                    eval_lens.append(ep_len[i])
                    ep_rew[i] = ep_cost[i] = ep_len[i] = 0.0
                    completed += 1
                    if completed >= target:
                        break
        if not eval_rews:
            return 0.0, 0.0, 0.0
        return (
            float(np.mean(eval_rews)),
            float(np.mean(eval_costs)),
            float(np.mean(eval_lens)),
        )

    def eval(self, eval_episodes: int = 100000, *, deterministic: bool = True) -> None:
        self.ppo_cfg["eval_episodes"] = eval_episodes
        r, c, l = self._eval(deterministic=deterministic)
        print(f"Eval EpRet={r:.4f} EpCost={c:.4f} EpLen={l:.2f}")


def train(args, cfg_train, cfg_env=None, *, use_lagrange: bool = False):
    env, eval_env = build_ma_envs(args, cfg_train, cfg_env)
    torch.set_num_threads(4)
    runner = Runner(
        env, eval_env, cfg_train, args.model_dir, use_lagrange=use_lagrange
    )
    if args.model_dir:
        runner.eval(100000)
    else:
        runner.run()


def _run_main(algo: str) -> None:
    set_np_formatting()
    args, cfg_env, cfg_train = multi_agent_args(algo=algo)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    use_lag = algo in ("ippo_lag", "ippo-lag")
    if args.write_terminal:
        train(args=args, cfg_train=cfg_train, cfg_env=cfg_env, use_lagrange=use_lag)
    else:
        os.makedirs(cfg_train["log_dir"], exist_ok=True)
        term = os.path.join(cfg_train["log_dir"], f"seed{args.seed}_terminal.log")
        err = os.path.join(cfg_train["log_dir"], f"seed{args.seed}_error.log")
        with open(term, "w", encoding="utf-8") as f_out:
            sys.stdout = f_out
            with open(err, "w", encoding="utf-8") as f_err:
                sys.stderr = f_err
                train(
                    args=args,
                    cfg_train=cfg_train,
                    cfg_env=cfg_env,
                    use_lagrange=use_lag,
                )


if __name__ == "__main__":
    _run_main("ippo")

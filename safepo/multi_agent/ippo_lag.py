# IPPO-Lag: independent per-agent PPO with Lagrangian constraint (PPO spine).
# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
# ==============================================================================

from safepo.multi_agent.ippo import _run_main, build_ma_envs, train as _train_ippo


def train(args, cfg_train, cfg_env=None):
    return _train_ippo(args, cfg_train, cfg_env, use_lagrange=True)


__all__ = ["build_ma_envs", "train"]


if __name__ == "__main__":
    _run_main("ippo_lag")

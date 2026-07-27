"""SA-style rolling episode metrics for SafePO multi-agent runners."""

from __future__ import annotations

from collections import deque
from statistics import fmean
from typing import Deque, Tuple

DEQUE_MAXLEN = 50


def make_metric_deques(
    maxlen: int = DEQUE_MAXLEN,
) -> Tuple[Deque[float], Deque[float], Deque[float]]:
    """Return (rew_deque, cost_deque, len_deque) matching SA PPO maxlen=50."""
    return (
        deque(maxlen=maxlen),
        deque(maxlen=maxlen),
        deque(maxlen=maxlen),
    )


def record_sa_style_episode_metrics(
    logger,
    rew_deque: Deque[float],
    cost_deque: Deque[float],
    len_deque: Deque[float],
    ep_ret: float,
    ep_cost: float,
    ep_len: float,
) -> None:
    """Append one completed episode and store rolling means (SA ppo.py pattern).

    Timeout (truncated) and mission-complete (terminated) completions both belong
    in the deques — same as SA ``if done or time_out``.
    """
    rew_deque.append(float(ep_ret))
    cost_deque.append(float(ep_cost))
    len_deque.append(float(ep_len))
    logger.store(
        **{
            "Metrics/EpRet": float(fmean(rew_deque)),
            "Metrics/EpCost": float(fmean(cost_deque)),
            "Metrics/EpLen": float(fmean(len_deque)),
        }
    )

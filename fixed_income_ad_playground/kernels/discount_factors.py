"""Discount factors kernels.

Currently only stepwise-constant instantaneous forwards.
"""

from fixed_income_ad_playground.types import JaxArray
import jax.numpy as jnp


def dfs_from_stepwise_const_forwards(
    knot_times: JaxArray,  # (interval_count+1,)
    ifr_values: JaxArray,  # (interval_count,)
    query_times: JaxArray,  # (Q,)
) -> JaxArray:
    """
    Discount factors P(0,t) for stepwise-constant instantaneous forwards.
    """
    interval_lengths = jnp.diff(knot_times)  # (interval_count,)
    cumulative_integral = jnp.cumsum(ifr_values * interval_lengths)  # (interval_count,)

    interval_index = jnp.searchsorted(knot_times, query_times, side="right") - 1
    interval_index = jnp.clip(interval_index, 0, ifr_values.size - 1)

    integral_to_prev = jnp.where(interval_index > 0, cumulative_integral[interval_index - 1], 0.0)
    partial = ifr_values[interval_index] * (query_times - knot_times[interval_index])
    return jnp.exp(-(integral_to_prev + partial))


def precompute_curve_integrals(
    knot_times: JaxArray, forwards_all: JaxArray
) -> tuple[JaxArray, JaxArray]:
    dt = jnp.diff(knot_times)
    cum_int = jnp.cumsum(forwards_all * dt[None, :], axis=1)
    return dt, cum_int


# only stepwise constant for now
def dfs_from_precomputed(
    knot_times: JaxArray,
    forwards: JaxArray,
    cum_int: JaxArray,
    query_times: JaxArray,
) -> JaxArray:
    """
    stepwise-constant instantaneous forwards on each interval.
    """
    idx = jnp.searchsorted(knot_times, query_times, side="right") - 1
    idx = jnp.clip(idx, 0, forwards.shape[-1] - 1)
    integral_to_prev = jnp.where(idx > 0, cum_int[..., idx - 1], 0.0)
    t_left = knot_times[idx]
    partial = forwards[..., idx] * (query_times - t_left)
    return jnp.exp(-(integral_to_prev + partial))

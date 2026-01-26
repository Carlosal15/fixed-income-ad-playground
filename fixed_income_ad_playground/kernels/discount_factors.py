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

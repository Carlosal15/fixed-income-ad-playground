"""Discount factors kernels.

Currently only stepwise-constant instantaneous forwards.
"""

from fixed_income_ad_playground.types import JaxArray
import jax.numpy as jnp


def dfs_from_stepwise_const_forwards(
    knot_times: JaxArray, ifr_vals: JaxArray, query_times: JaxArray
) -> JaxArray:
    """
    Convenience (single curve) wrapper. Kept for the python-layer Curve object.
    """
    dt = jnp.diff(knot_times)
    cum_int = jnp.cumsum(ifr_vals * dt, axis=0)
    idx = jnp.searchsorted(knot_times, query_times, side="right") - 1
    idx = jnp.clip(idx, 0, ifr_vals.size - 1)
    integral_to_prev = jnp.where(idx > 0, cum_int[idx - 1], 0.0)
    t_left = knot_times[idx]
    partial = ifr_vals[idx] * (query_times - t_left)
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

from fixed_income_ad_playground.kernels.utils import safe_div, safe_sqrt

from fixed_income_ad_playground.types import JaxArray

import jax.numpy as jnp


def penalty_slope(forwards: JaxArray, knot_times: JaxArray, lam: JaxArray | float) -> JaxArray:
    lengths = knot_times[1:] - knot_times[:-1]
    diffs = forwards[1:] - forwards[:-1]
    scale = safe_sqrt(jnp.maximum(lam, 0.0))
    return scale * safe_div(diffs, lengths[:-1])


def penalty_conv(forwards: JaxArray, knot_times: JaxArray, lam: JaxArray | float) -> JaxArray:
    lengths = knot_times[1:] - knot_times[:-1]
    second = forwards[2:] - 2.0 * forwards[1:-1] + forwards[:-2]
    scale = safe_sqrt(jnp.maximum(lam, 0.0))
    denom = lengths[1:-1] ** 2
    return scale * safe_div(second, denom)


def penalty_level(forwards: JaxArray, lam: JaxArray | float) -> JaxArray:
    scale = safe_sqrt(jnp.maximum(lam, 0.0))
    return scale * forwards

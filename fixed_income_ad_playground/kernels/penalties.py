from fixed_income_ad_playground.kernels.utils import safe_div, safe_sqrt

from fixed_income_ad_playground.types import JaxArray

import jax.numpy as jnp


def penalty_slope_fixedshape(
    forward_params_per_interval: JaxArray, knot_times: JaxArray, lam: float
) -> JaxArray:
    """
    First-difference penalty, shape (interval_count-1,).
    If lam == 0, scale is 0 => residuals all zeros but shape unchanged.
    """
    interval_lengths = knot_times[1:] - knot_times[:-1]  # (interval_count,)
    forward_diffs = (
        forward_params_per_interval[1:] - forward_params_per_interval[:-1]
    )  # (interval_count-1,)
    scale = safe_sqrt(jnp.maximum(lam, 0.0))
    return scale * safe_div(forward_diffs, interval_lengths[:-1])


def penalty_curv_fixedshape(
    forward_params_per_interval: JaxArray, knot_times: JaxArray, lam: float
) -> JaxArray:
    """
    Second-difference penalty, shape (interval_count-2,).
    """
    interval_lengths = knot_times[1:] - knot_times[:-1]  # (interval_count,)
    second_diff = (
        forward_params_per_interval[2:]
        - 2.0 * forward_params_per_interval[1:-1]
        + forward_params_per_interval[:-2]
    )  # (interval_count-2,)
    scale = safe_sqrt(jnp.maximum(lam, 0.0))
    denom = interval_lengths[1:-1] ** 2
    return scale * safe_div(second_diff, denom)

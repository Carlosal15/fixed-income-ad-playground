from __future__ import annotations

from fixed_income_ad_playground.kernels.penalties import (
    penalty_conv,
    penalty_level,
    penalty_slope,
)
from fixed_income_ad_playground.kernels.utils import safe_sqrt
from fixed_income_ad_playground.packing.swaps import PackedSwapBatchPooled
from fixed_income_ad_playground.types import JaxArray
import jax.numpy as jnp
import jax
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fixed_income_ad_playground.calibration.calibration import CurveGraphArrays  # type: ignore


def residuals_kernel(
    params_concat: JaxArray,
    market_par_swaps: JaxArray,  # (S bucketed,)
    knot_times: JaxArray,
    graph: CurveGraphArrays,
    batch: PackedSwapBatchPooled,
    lam_slope_per_param: JaxArray,  # (P,)
    lam_curv_per_param: JaxArray,  # (P,)
    lam_level_per_param: JaxArray,  # (P,)
) -> JaxArray:
    # yikes fix... import cycle. TODO: refactor
    from fixed_income_ad_playground.kernels.swaps import par_swap_rates_pooled

    # swaps block
    model_par = par_swap_rates_pooled(params_concat, knot_times, graph, batch)  # (S,)
    w_sqrt = safe_sqrt(jnp.maximum(batch.weights, 0.0))
    res_swaps = w_sqrt * (model_par - market_par_swaps)

    # penalties only for param curves
    N = knot_times.size - 1
    P = lam_slope_per_param.size

    def get_block(k):
        start = k * N
        return jax.lax.dynamic_slice(params_concat, (start,), (N,))

    level_blocks = jax.vmap(
        lambda k: penalty_level(get_block(k), lam_level_per_param[k]),
        in_axes=(0,),
    )(jnp.arange(P, dtype=jnp.int32))

    slope_blocks = jax.vmap(
        lambda k: penalty_slope(get_block(k), knot_times, lam_slope_per_param[k]),
        in_axes=(0,),
    )(jnp.arange(P, dtype=jnp.int32))

    curv_blocks = jax.vmap(
        lambda k: penalty_conv(get_block(k), knot_times, lam_curv_per_param[k]),
        in_axes=(0,),
    )(jnp.arange(P, dtype=jnp.int32))

    return jnp.concatenate(
        [res_swaps, level_blocks.reshape(-1), slope_blocks.reshape(-1), curv_blocks.reshape(-1)],
        axis=0,
    )


# Previous versions used jit-compiled versions of the residuals as global singletons;
# moving away for versatility and letting the calibrator decide
# RES_JIT = jax.jit(residuals_kernel)
# JAC_JIT = jax.jit(jax.jacfwd(residuals_kernel, argnums=0))

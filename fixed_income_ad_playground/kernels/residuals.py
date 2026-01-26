from fixed_income_ad_playground.kernels.swaps import par_swap_rates_from_packed
from fixed_income_ad_playground.kernels.futures import futures_quotes_from_packed
from fixed_income_ad_playground.kernels.penalties import (
    penalty_slope_fixedshape,
    penalty_curv_fixedshape,
)
from fixed_income_ad_playground.kernels.utils import safe_sqrt
from fixed_income_ad_playground.types import JaxArray
import jax.numpy as jnp
import jax


def residuals_kernel(
    forward_params_per_interval: JaxArray,
    # swaps block
    market_par_swaps: JaxArray,  # (swap_count,)
    swap_weights: JaxArray,  # (swap_count,)
    knot_times: JaxArray,
    swap_payment_times: JaxArray,  # (swap_count, max_payments)
    swap_accrual_factors: JaxArray,  # (swap_count, max_payments)
    swap_cashflow_mask: JaxArray,  # (swap_count, max_payments)
    swap_maturity_times: JaxArray,  # (swap_count,)
    # futures block (stub)
    market_quotes_futures: JaxArray,  # (future_count,)
    futures_weights: JaxArray,  # (future_count,)
    futures_expiry_times: JaxArray,  # (future_count,)
    # penalties
    lam_slope: float,
    lam_curv: float,
) -> JaxArray:
    # --- swaps block ---
    model_par_swaps = par_swap_rates_from_packed(
        forward_params_per_interval,
        knot_times,
        swap_payment_times,
        swap_accrual_factors,
        swap_cashflow_mask,
        swap_maturity_times,
    )
    swap_w_sqrt = safe_sqrt(jnp.maximum(swap_weights, 0.0))
    residuals_swaps = swap_w_sqrt * (model_par_swaps - market_par_swaps)  # (swap_count,)

    # --- futures block (stub) ---
    model_quotes_futures = futures_quotes_from_packed(
        forward_params_per_interval,
        knot_times,
        futures_expiry_times,
    )
    fut_w_sqrt = safe_sqrt(jnp.maximum(futures_weights, 0.0))
    residuals_futures = fut_w_sqrt * (
        model_quotes_futures - market_quotes_futures
    )  # (future_count,)

    # --- penalties ---
    residuals_slope = penalty_slope_fixedshape(
        forward_params_per_interval, knot_times, lam_slope
    )  # (interval_count-1,)
    residuals_curv = penalty_curv_fixedshape(
        forward_params_per_interval, knot_times, lam_curv
    )  # (interval_count-2,)

    return jnp.concatenate(
        [residuals_swaps, residuals_futures, residuals_slope, residuals_curv], axis=0
    )


# JIT-compiled, use as singletons exported from this module
RES_JIT = jax.jit(residuals_kernel)
JAC_JIT = jax.jit(jax.jacfwd(residuals_kernel, argnums=0))

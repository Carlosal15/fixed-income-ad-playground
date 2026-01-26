"""Kernels for swap pricing and calibration.

This is a single-curve par swap rate kernel, hardcoded to stepwise-constant.
It will be interesting, in the future, to generalise this to other
interpolator types. A more appropriate desing would be a par swap curve function
generator, instantiated from the scipy wrapper, and letting that go into the jitted
residuals function.
"""

from fixed_income_ad_playground.kernels.discount_factors import (
    dfs_from_stepwise_const_forwards,
)
from fixed_income_ad_playground.types import JaxArray
from fixed_income_ad_playground.kernels.utils import safe_div
import jax.numpy as jnp


# par swap rates are hardcoded to stepwise-constant instantaneous forwards
# No dynamic dispatch to other df calculations based on curve config if we want
# jit. Could be interesting to generalise later.
def par_swap_rates_from_packed(
    forward_params_per_interval: JaxArray,  # (interval_count,)
    knot_times: JaxArray,  # (interval_count+1,)
    payment_times: JaxArray,  # (swap_count, max_payments)
    accrual_factors: JaxArray,  # (swap_count, max_payments)
    cashflow_mask: JaxArray,  # (swap_count, max_payments)
    maturity_times: JaxArray,  # (swap_count,)
) -> JaxArray:
    """
    Single-curve par swap rate:
        S = (1 - P(T)) / sum_i alpha_i P(t_i)

    Hardcoded to stepwise-constant instantaneous forwards.
    """
    swap_count, max_payments = payment_times.shape

    flat_times = payment_times.reshape(-1)
    dfs_flat = dfs_from_stepwise_const_forwards(knot_times, forward_params_per_interval, flat_times)
    dfs_payments = dfs_flat.reshape(swap_count, max_payments)

    df_maturity = dfs_from_stepwise_const_forwards(
        knot_times, forward_params_per_interval, maturity_times
    )
    annuity = jnp.sum(cashflow_mask * accrual_factors * dfs_payments, axis=1)

    return safe_div(1.0 - df_maturity, annuity)

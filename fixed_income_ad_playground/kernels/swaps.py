"""Kernels for swap pricing and calibration.

This is a single-curve par swap rate kernel, hardcoded to stepwise-constant.
It will be interesting, in the future, to generalise this to other
interpolator types. A more appropriate desing would be a par swap curve function
generator, instantiated from the scipy wrapper, and letting that go into the jitted
residuals function.
"""

from fixed_income_ad_playground.calibration.calibration import (  # type: ignore
    compute_forwards_all_curves,
    CurveGraphArrays,
)
from fixed_income_ad_playground.kernels.discount_factors import (
    dfs_from_precomputed,
    # dfs_from_stepwise_const_forwards,
    precompute_curve_integrals,
)
from fixed_income_ad_playground.packing.swaps import PackedSwapBatchPooled
from fixed_income_ad_playground.types import JaxArray
from fixed_income_ad_playground.kernels.utils import safe_div
import jax.numpy as jnp
import jax


# par swap rates are hardcoded to stepwise-constant instantaneous forwards
# No dynamic dispatch to other df calculations based on curve config if we want
# jit. Could be interesting to generalise later.
# def par_swap_rates_from_packed(
#     forward_params_per_interval: JaxArray,  # (interval_count,)
#     knot_times: JaxArray,  # (interval_count+1,)
#     payment_times: JaxArray,  # (swap_count, max_payments)
#     accrual_factors: JaxArray,  # (swap_count, max_payments)
#     cashflow_mask: JaxArray,  # (swap_count, max_payments)
#     maturity_times: JaxArray,  # (swap_count,)
# ) -> JaxArray:
#     """
#     Single-curve par swap rate:
#         S = (1 - P(T)) / sum_i alpha_i P(t_i)

#     Hardcoded to stepwise-constant instantaneous forwards.
#     """
#     swap_count, max_payments = payment_times.shape

#     flat_times = payment_times.reshape(-1)
#     dfs_flat = dfs_from_stepwise_const_forwards(knot_times, forward_params_per_interval, flat_times)
#     dfs_payments = dfs_flat.reshape(swap_count, max_payments)

#     df_maturity = dfs_from_stepwise_const_forwards(
#         knot_times, forward_params_per_interval, maturity_times
#     )
#     annuity = jnp.sum(cashflow_mask * accrual_factors * dfs_payments, axis=1)

#     return safe_div(1.0 - df_maturity, annuity)


def par_swap_rates_pooled(
    params_concat: JaxArray,
    knot_times: JaxArray,
    graph: CurveGraphArrays,
    batch: PackedSwapBatchPooled,
) -> JaxArray:
    """
    Compute discount factors at pooled times once per curve, then gather for cashflows.

    NOTE: for CPU, the advanced indexing gather can be a hotspot. This is the
    correct "wiring" version; further micro-optimisation may restructure the gather.
    """
    forwards_all = compute_forwards_all_curves(params_concat, knot_times, graph)  # (C,N)
    _, cum_int_all = precompute_curve_integrals(knot_times, forwards_all)  # (C,N)

    dfs_pool = jax.vmap(
        lambda fwd, ci: dfs_from_precomputed(knot_times, fwd, ci, batch.t_pool),
        in_axes=(0, 0),
    )(forwards_all, cum_int_all)  # (C,Q)

    # gather for each swap/cashflow
    disc_idx_cf = jnp.repeat(
        batch.discount_curve_idx[:, None], repeats=batch.cashflows_bucket_size, axis=1
    )  # (S,M)
    fcast_idx_cf = jnp.repeat(
        batch.forecast_curve_idx[:, None], repeats=batch.cashflows_bucket_size, axis=1
    )  # (S,M)

    df_d_end = dfs_pool[disc_idx_cf, batch.idx_end]
    df_f_end = dfs_pool[fcast_idx_cf, batch.idx_end]
    df_f_start = dfs_pool[fcast_idx_cf, batch.idx_start]

    ratio = safe_div(df_f_start, df_f_end)
    fwd_rate = safe_div(ratio - 1.0, batch.accrual_factors)

    pv_float = jnp.sum(batch.cashflow_mask * batch.accrual_factors * fwd_rate * df_d_end, axis=1)
    annuity = jnp.sum(batch.cashflow_mask * batch.accrual_factors * df_d_end, axis=1)
    return safe_div(pv_float, annuity)


def swap_pv_components_pooled(
    params_concat: JaxArray,
    knot_times: JaxArray,
    graph: CurveGraphArrays,
    batch: PackedSwapBatchPooled,
) -> tuple[JaxArray, JaxArray]:
    """
    Returns (pv_float, annuity) per swap using pooled date table.
    """
    forwards_all = compute_forwards_all_curves(params_concat, knot_times, graph)  # (C,N)
    _, cum_int_all = precompute_curve_integrals(knot_times, forwards_all)  # (C,N)

    dfs_pool = jax.vmap(
        lambda fwd, ci: dfs_from_precomputed(knot_times, fwd, ci, batch.t_pool),
        in_axes=(0, 0),
    )(forwards_all, cum_int_all)

    # gather for each swap/cashflow
    disc_idx_cf = jnp.repeat(
        batch.discount_curve_idx[:, None], repeats=batch.cashflows_bucket_size, axis=1
    )
    fcast_idx_cf = jnp.repeat(
        batch.forecast_curve_idx[:, None], repeats=batch.cashflows_bucket_size, axis=1
    )

    df_d_end = dfs_pool[disc_idx_cf, batch.idx_end]
    df_f_end = dfs_pool[fcast_idx_cf, batch.idx_end]
    df_f_start = dfs_pool[fcast_idx_cf, batch.idx_start]

    ratio = safe_div(df_f_start, df_f_end)
    fwd_rate = safe_div(ratio - 1.0, batch.accrual_factors)

    pv_float = jnp.sum(batch.cashflow_mask * batch.accrual_factors * fwd_rate * df_d_end, axis=1)
    annuity = jnp.sum(batch.cashflow_mask * batch.accrual_factors * df_d_end, axis=1)
    return pv_float, annuity


def swap_pv_pooled(
    params_concat: JaxArray,
    knot_times: JaxArray,
    graph: CurveGraphArrays,
    batch: PackedSwapBatchPooled,
    coupon_per_swap: JaxArray,
) -> JaxArray:
    pv_float, annuity = swap_pv_components_pooled(params_concat, knot_times, graph, batch)
    return (pv_float - coupon_per_swap * annuity) * batch.notional_per_swap

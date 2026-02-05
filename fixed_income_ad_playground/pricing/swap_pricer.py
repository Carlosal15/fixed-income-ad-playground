from dataclasses import dataclass

from fixed_income_ad_playground.reference_data_container import ReferenceDataContainer
from fixed_income_ad_playground.instruments.instrument import PackContext
from fixed_income_ad_playground.instruments.swap import SwapSpec, PackedSwap
from fixed_income_ad_playground.market import Market
from fixed_income_ad_playground.kernels.utils import safe_div
import jax.numpy as jnp


@dataclass(frozen=True)
class SwapPricer:
    """
    Prices and analytics for swaps
    """

    def par_rate(
        self,
        market: Market,
        swap: SwapSpec,
        ctx: PackContext,
        reference_data: ReferenceDataContainer,
    ) -> float:
        p = swap.pack(ctx, reference_data)
        if not isinstance(p, PackedSwap):
            raise TypeError("Expected (packed) swaps")

        disc = market.curve(p.discount_curve)
        fcast = market.curve(p.forecast_curve)

        st = jnp.array(p.start_times)
        et = jnp.array(p.end_times)
        alpha = jnp.array(p.accrual_factors)

        df_d = disc.discount_factors(et)
        df_f_s = fcast.discount_factors(st)
        df_f_e = fcast.discount_factors(et)

        fwd = safe_div(safe_div(df_f_s, df_f_e) - 1.0, alpha)
        pv_float = jnp.sum(alpha * fwd * df_d)
        annuity = jnp.sum(alpha * df_d)
        return float(safe_div(pv_float, annuity))

    def pv(
        self,
        market: Market,
        swap: SwapSpec,
        fixed_rate: float,
        ctx: PackContext,
        reference_data: ReferenceDataContainer,
    ) -> float:
        p = swap.pack(ctx, reference_data)
        if not isinstance(p, PackedSwap):
            raise TypeError("Expected (packed) swaps")

        disc = market.curve(p.discount_curve)
        fcast = market.curve(p.forecast_curve)

        st = jnp.array(p.start_times)
        et = jnp.array(p.end_times)
        alpha = jnp.array(p.accrual_factors)

        df_d = disc.discount_factors(et)
        df_f_s = fcast.discount_factors(st)
        df_f_e = fcast.discount_factors(et)

        fwd = safe_div(safe_div(df_f_s, df_f_e) - 1.0, alpha)
        pv_float = jnp.sum(alpha * fwd * df_d)
        pv_fixed = float(fixed_rate) * jnp.sum(alpha * df_d)
        return float(pv_fixed - pv_float)

from dataclasses import dataclass

from fixed_income_ad_playground.instruments.instrument import PackContext
from fixed_income_ad_playground.instruments.swap import SwapSpec, PackedSwap
from fixed_income_ad_playground.market.market import Market
from fixed_income_ad_playground.kernels.utils import safe_div
import jax.numpy as jnp


@dataclass(frozen=True)
class SwapPricer:
    def par_rate(self, market: Market, swap: SwapSpec, ctx: PackContext) -> float:
        packed = swap.pack(ctx)

        if not isinstance(packed, PackedSwap):
            raise TypeError("par_rate expects SwapSpec/PackedSwap only")
        curve = market.curve(packed.discount_curve)

        payment_times = jnp.array(packed.payment_times)
        accruals = jnp.array(packed.accrual_factors)

        df_pay = curve.discount_factors(payment_times)
        df_T = curve.discount_factors(jnp.array([packed.maturity_years]))[0]

        annuity = jnp.sum(accruals * df_pay)
        par = safe_div(1.0 - df_T, annuity)
        return float(par)

    def pv(self, market: Market, swap: SwapSpec, fixed_rate: float, ctx: PackContext) -> float:
        packed = swap.pack(ctx)

        if not isinstance(packed, PackedSwap):
            raise TypeError("pv expects SwapSpec/PackedSwap only")
        curve = market.curve(packed.discount_curve)

        payment_times = jnp.array(packed.payment_times)
        accruals = jnp.array(packed.accrual_factors)

        df_pay = curve.discount_factors(payment_times)
        df_T = curve.discount_factors(jnp.array([packed.maturity_years]))[0]

        pv_float = 1.0 - df_T
        pv_fixed = float(fixed_rate) * jnp.sum(accruals * df_pay)
        pv = pv_fixed - pv_float
        return float(pv)

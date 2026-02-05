import jax.numpy as jnp
from fixed_income_ad_playground.pricing.swap_pricer import SwapPricer
from fixed_income_ad_playground.instruments.swap import SwapSpec
from fixed_income_ad_playground.instruments.instrument import PackContext
from fixed_income_ad_playground.identifiers.identifiers import InstrumentId, CurveId
from fixed_income_ad_playground.reference_data_container import ReferenceDataContainer
from fixed_income_ad_playground.curve.curve import Curve
from fixed_income_ad_playground.curve.interpolator import StepwiseConstFwdInterpolator
from fixed_income_ad_playground.market import Market


def make_const_curve(cid: str, level: float):
    return Curve(
        curve_id=CurveId(cid),
        knot_times=jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float64),
        params=jnp.array([level, level, level], dtype=jnp.float64),
        interpolator=StepwiseConstFwdInterpolator(),
    )


def test_par_rate_constant_curves_equals_forward_rate():
    disc_level = 0.02
    fcast_level = 0.02
    market = Market({
        CurveId("USD_DISC"): make_const_curve("USD_DISC", disc_level),
        CurveId("USD_FCAST"): make_const_curve("USD_FCAST", fcast_level),
    })
    ref = ReferenceDataContainer({"USD_SOFR_OIS": (CurveId("USD_DISC"), CurveId("USD_FCAST"))})

    swap = SwapSpec(
        InstrumentId("S1"),
        "USD",
        "USD_SOFR_OIS",
        maturity=2.0,
        fixed_leg_freq="A",
        float_leg_freq="A",
    )

    pricer = SwapPricer()
    rate = pricer.par_rate(market, swap, PackContext(), ref)
    # With equal constant curves, par is very close to forecast level
    assert abs(rate - fcast_level) < 5e-4


def test_pv_zero_at_par_rate():
    disc_level = 0.015
    fcast_level = 0.025
    market = Market({
        CurveId("USD_DISC"): make_const_curve("USD_DISC", disc_level),
        CurveId("USD_FCAST"): make_const_curve("USD_FCAST", fcast_level),
    })
    ref = ReferenceDataContainer({"USD_SOFR_OIS": (CurveId("USD_DISC"), CurveId("USD_FCAST"))})
    swap = SwapSpec(
        InstrumentId("S2"),
        "USD",
        "USD_SOFR_OIS",
        maturity=3.0,
        fixed_leg_freq="A",
        float_leg_freq="S",
    )

    pricer = SwapPricer()
    par = pricer.par_rate(market, swap, PackContext(), ref)
    pv = pricer.pv(market, swap, fixed_rate=par, ctx=PackContext(), reference_data=ref)
    assert abs(pv) < 1e-10

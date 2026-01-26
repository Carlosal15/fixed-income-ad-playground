import numpy as np
import jax.numpy as jnp
from fixed_income_ad_playground.identifiers.identifiers import InstrumentId, CurveId
from fixed_income_ad_playground.instruments.swap import SwapSpec
from fixed_income_ad_playground.instruments.instrument import PackContext
from fixed_income_ad_playground.curve.curve import FwdCurve
from fixed_income_ad_playground.market.market import Market
from fixed_income_ad_playground.pricing.swap_pricer import SwapPricer
from fixed_income_ad_playground.pricing.quote import Quote
from fixed_income_ad_playground.packing.swaps import pack_swaps_to_batch


def test_swap_pricer_par_rate_and_pv():
    cid = CurveId("c")
    sid = InstrumentId("s1")
    # simple piecewise forwards: 0->1 @0.0, 1->2 @0.02
    knot_times = jnp.array([0.0, 1.0, 2.0])
    forwards = jnp.array([0.0, 0.02])
    curve = FwdCurve(curve_id=cid, knot_times=knot_times, forward_params_per_interval=forwards)
    market = Market(curves={cid: curve})

    swap = SwapSpec(instrument_id=sid, maturity=2.0, discount_curve=cid)
    pricer = SwapPricer()
    ctx = PackContext()

    par = pricer.par_rate(market, swap, ctx)
    # compute reference par: annuity = sum(df_pay*accruals)
    packed = swap.pack(ctx)
    payment_times = np.asarray(packed.payment_times)
    accruals = np.asarray(packed.accrual_factors)
    df_pay = np.asarray(curve.discount_factors(jnp.array(payment_times)))
    df_T = float(np.asarray(curve.discount_factors(jnp.array([packed.maturity_years])))[0])
    ann = float(np.sum(accruals * df_pay))
    expected_par = (1.0 - df_T) / ann
    assert np.isclose(par, expected_par, rtol=1e-6)


def test_pack_swaps_to_batch():
    cid = CurveId("c")
    sid1 = InstrumentId("s1")
    sid2 = InstrumentId("s2")
    ctx = PackContext()

    s1 = SwapSpec(instrument_id=sid1, maturity=0.5, discount_curve=cid)
    s2 = SwapSpec(instrument_id=sid2, maturity=2.0, discount_curve=cid)
    q1 = Quote(instrument=s1, quote_type="par_rate", value=0.01)
    q2 = Quote(instrument=s2, quote_type="par_rate", value=0.02)

    batch = pack_swaps_to_batch([q1, q2], ctx)
    # shapes
    assert batch.payment_times.shape[0] == 2
    assert batch.maturity_times.shape[0] == 2

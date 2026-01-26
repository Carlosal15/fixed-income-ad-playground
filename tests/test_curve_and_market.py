import numpy as np
import jax.numpy as jnp
from fixed_income_ad_playground.curve.curve import FwdCurve
from fixed_income_ad_playground.identifiers.identifiers import CurveId
from fixed_income_ad_playground.market.market import Market
from fixed_income_ad_playground.kernels.discount_factors import dfs_from_stepwise_const_forwards


def test_fwd_curve_discount_and_zero():
    knot_times = jnp.array([0.0, 1.0, 2.0])
    forwards = jnp.array([0.01, 0.02])
    cid = CurveId("c")
    curve = FwdCurve(curve_id=cid, knot_times=knot_times, forward_params_per_interval=forwards)

    t = jnp.array([0.5, 1.5, 2.0])
    dfs = curve.discount_factors(t)
    # reference by calling kernel directly
    ref = dfs_from_stepwise_const_forwards(knot_times, forwards, t)
    assert np.allclose(np.asarray(dfs), np.asarray(ref))

    # zero rates: -ln(P)/t (skip t=0)
    zr = curve.zero(t)
    assert zr.shape == t.shape


def test_market_curve_lookup():
    knot_times = jnp.array([0.0, 1.0])
    forwards = jnp.array([0.01])
    cid = CurveId("c")
    curve = FwdCurve(curve_id=cid, knot_times=knot_times, forward_params_per_interval=forwards)
    m = Market(curves={cid: curve})
    assert m.curve(cid) is curve

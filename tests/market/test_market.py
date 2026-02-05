import pytest
from fixed_income_ad_playground.market import Market
from fixed_income_ad_playground.curve.curve import Curve
from fixed_income_ad_playground.curve.interpolator import StepwiseConstFwdInterpolator
from fixed_income_ad_playground.identifiers.identifiers import CurveId
import jax.numpy as jnp


def make_curve(cid: str):
    return Curve(
        curve_id=CurveId(cid),
        knot_times=jnp.array([0.0, 1.0], dtype=jnp.float64),
        params=jnp.array([0.03], dtype=jnp.float64),
        interpolator=StepwiseConstFwdInterpolator(),
    )


def test_market_curve_lookup_and_missing():
    m = Market(curves={CurveId("USD_DISC"): make_curve("USD_DISC")})
    c = m.curve(CurveId("USD_DISC"))
    assert c.curve_id.name == "USD_DISC"
    with pytest.raises(KeyError):
        m.curve(CurveId("EUR_DISC"))

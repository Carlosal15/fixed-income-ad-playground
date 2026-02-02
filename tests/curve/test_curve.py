import numpy as np
import jax.numpy as jnp
from fixed_income_ad_playground.curve.curve import Curve
from fixed_income_ad_playground.curve.interpolator import StepwiseConstFwdInterpolator
from fixed_income_ad_playground.identifiers.identifiers import CurveId


def make_simple_curve(level: float = 0.03):
    knot = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float64)
    params = jnp.array([level, level], dtype=jnp.float64)
    interp = StepwiseConstFwdInterpolator()
    return Curve(curve_id=CurveId("TEST"), knot_times=knot, params=params, interpolator=interp)


def test_discount_factors_and_zero_rate_consistency():
    c = make_simple_curve(0.04)
    t = jnp.array([0.25, 1.0, 1.75], dtype=jnp.float64)
    df = c.discount_factors(t)
    zero = c.zero_rate(t)
    # df = exp(-z * t)
    np.testing.assert_allclose(
        np.asarray(df), np.exp(-np.asarray(zero) * np.asarray(t)), rtol=1e-12
    )


def test_discount_factors_matches_constant_forward():
    c = make_simple_curve(0.03)
    t = jnp.array([0.25, 0.5, 1.5], dtype=jnp.float64)
    df = c.discount_factors(t)
    expected = jnp.exp(-0.03 * t)
    np.testing.assert_allclose(np.asarray(df), np.asarray(expected), rtol=1e-12, atol=1e-12)

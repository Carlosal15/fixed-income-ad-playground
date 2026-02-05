import numpy as np
import jax.numpy as jnp
from fixed_income_ad_playground.kernels.discount_factors import dfs_from_stepwise_const_forwards


def test_dfs_stepwise_const_constant_rate():
    # If forwards are constant r over [0,T], then P(0,t) = exp(-r*t)
    knot = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float64)  # N=2 intervals
    r = 0.03
    f = jnp.array([r, r], dtype=jnp.float64)
    t = jnp.array([0.25, 0.75, 1.5], dtype=jnp.float64)

    df = dfs_from_stepwise_const_forwards(knot, f, t)
    expected = jnp.exp(-r * t)
    np.testing.assert_allclose(np.asarray(df), np.asarray(expected), rtol=1e-12, atol=1e-12)

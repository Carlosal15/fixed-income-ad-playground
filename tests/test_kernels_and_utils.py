import numpy as np
import jax.numpy as jnp
from fixed_income_ad_playground.kernels.discount_factors import dfs_from_stepwise_const_forwards
from fixed_income_ad_playground.kernels.utils import safe_div, safe_sqrt


def test_dfs_stepwise_constant_forwards():
    knot_times = jnp.array([0.0, 1.0, 2.0])
    ifr = jnp.array([0.0, 0.02])
    qs = jnp.array([0.5, 1.5, 2.0])

    dfs = dfs_from_stepwise_const_forwards(knot_times, ifr, qs)
    # expected: for t=0.5 -> exp(-0*0.5)=1.0
    assert np.allclose(float(dfs[0]), 1.0, atol=1e-12)
    # t=1.5 -> exp(-0.02*0.5)
    assert np.allclose(float(dfs[1]), np.exp(-0.02 * 0.5), rtol=1e-6)
    # t=2.0 -> exp(-0.02*1.0)
    assert np.allclose(float(dfs[2]), np.exp(-0.02 * 1.0), rtol=1e-6)


def test_utils_safe_ops():
    a = jnp.array([1.0, 4.0])
    assert float(safe_sqrt(a)[1]) == 2.0
    # safe_div dividing by small denom yields finite
    b = jnp.array([0.0, 0.0])
    out = safe_div(jnp.array([1.0, 0.0]), b)
    assert out.shape == (2,)

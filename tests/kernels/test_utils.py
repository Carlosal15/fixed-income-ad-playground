import numpy as np
import jax.numpy as jnp
from fixed_income_ad_playground.kernels.utils import safe_div, safe_sqrt


def test_safe_div_basic_and_zero():
    a = jnp.array([2.0, 4.0, 6.0], dtype=jnp.float64)
    b = jnp.array([2.0, 0.0, 3.0], dtype=jnp.float64)
    out = safe_div(a, b)
    np.testing.assert_allclose(np.asarray(out), np.asarray([1.0, 0.0, 2.0]), rtol=1e-12, atol=1e-12)


def test_safe_sqrt():
    x = jnp.array([-1.0, 0.0, 4.0])
    out = safe_sqrt(x, eps=0.0)
    # sqrt(max(x, 0))
    np.testing.assert_allclose(np.asarray(out), np.asarray([0.0, 0.0, 2.0]))

import numpy as np
import jax.numpy as jnp
from fixed_income_ad_playground.types import FloatNDArray, JaxArray


def test_floatndarray_alias():
    arr: FloatNDArray = np.array([1.0, 2.0], dtype=np.float64)
    assert arr.dtype == np.float64
    assert arr.shape == (2,)


def test_jaxarray_alias():
    arr: JaxArray = jnp.array([1.0, 2.0])
    assert isinstance(arr, jnp.ndarray)
    assert arr.shape == (2,)

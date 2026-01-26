import jax.numpy as jnp
from fixed_income_ad_playground import enums, types


def test_enums_and_types_import():
    # ensure enums and types import and JaxArray type accepts jnp.ndarray
    assert hasattr(enums, "DayCount")
    a = jnp.array([1.0, 2.0])
    assert isinstance(a, types.JaxArray)

from fixed_income_ad_playground.types import JaxArray
import jax.numpy as jnp


def safe_sqrt(x: JaxArray, eps: float = 0.0) -> JaxArray:
    """sqrt(max(x, eps))"""
    return jnp.sqrt(jnp.maximum(x, eps))


def safe_div(numer: JaxArray, denom: JaxArray, eps: float = 1e-16) -> JaxArray:
    """
    Safe division: numer / denom, with protection against NaNs.
    """
    denom_ok = jnp.abs(denom) > eps
    denom_safe = jnp.where(denom_ok, denom, 1.0)
    out = numer / denom_safe
    return jnp.where(denom_ok, out, 0.0)

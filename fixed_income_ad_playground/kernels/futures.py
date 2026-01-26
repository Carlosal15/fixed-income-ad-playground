from fixed_income_ad_playground.types import JaxArray
import jax.numpy as jnp


def futures_quotes_from_packed(
    ifr_values: JaxArray,
    knot_times: JaxArray,
    expiry_times: JaxArray,  # (future_count,)
) -> JaxArray:
    """
    Futures pricing stub. To be implemented.
    return zeros (jit-friendly because shape is stable)
    """
    return jnp.zeros_like(expiry_times)

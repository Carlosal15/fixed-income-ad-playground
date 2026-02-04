from collections.abc import Callable
from fixed_income_ad_playground.calibration.calibration import CalibStatic
from fixed_income_ad_playground.calibration.risk import _make_swap_labels
from fixed_income_ad_playground.kernels.swaps import par_swap_rates_pooled, swap_pv_pooled
from fixed_income_ad_playground.types import FloatNDArray, JaxArray
import jax.numpy as jnp
import pandas as pd


def portfolio_pv_and_risk_df(
    params_concat: JaxArray,
    static: CalibStatic,
    coupon_per_swap_np: FloatNDArray | JaxArray,
    jac_pv_fn: Callable[[JaxArray, JaxArray], JaxArray],  # TODO: unsure about type
    dv01_bucket_fn: Callable[[JaxArray, JaxArray], JaxArray],  # TODO: unsure about type
    bucket_labels: list[str],
):
    """
    Build a pandas DataFrame with PV and bucketed DV01 per swap.
    """
    coupon = jnp.array(coupon_per_swap_np, dtype=jnp.float64)
    pv = swap_pv_pooled(
        params_concat, static.knot_times, static.graph, static.batch, coupon
    )  # (S,)
    J = jac_pv_fn(params_concat, coupon)  # (S, P*N)
    dv01_bucket = dv01_bucket_fn(params_concat, coupon)  # (S,B)

    S_true = int(static.batch.swap_count_true)
    ids = _make_swap_labels(static.batch, prefix="USD_SOFR_OIS")
    pv_true = jnp.array(pv[:S_true]).tolist()
    dv01_true = jnp.array(dv01_bucket[:S_true, :])
    df = pd.DataFrame({
        "instrument_id": ids,
        "PV": pv_true,
    })
    for b, lbl in enumerate(bucket_labels):
        df[f"DV01_{lbl}"] = dv01_true[:, b].tolist()
    return df, J


# TODO:
# - refactor for SwapPricer
# - move to demo? or restrict prefix late
def portfolio_par_df(
    params_concat: JaxArray,
    static: CalibStatic,
):
    """
    Build a pandas DataFrame with model par rate per swap.
    """
    model_par = par_swap_rates_pooled(
        params_concat, static.knot_times, static.graph, static.batch
    )  # (S,)
    import pandas as pd

    S_true = int(static.batch.swap_count_true)
    ids = _make_swap_labels(static.batch, prefix="USD_SOFR_OIS")
    par_true = jnp.array(model_par[:S_true]).tolist()
    df = pd.DataFrame({
        "instrument_id": ids,
        "Par": par_true,
    })
    return df


def portfolio_par_and_risk_df(
    params_concat: JaxArray,
    static: CalibStatic,
    coupon_per_swap_np: JaxArray | FloatNDArray,
    dv01_bucket_fn: Callable[[JaxArray, JaxArray], JaxArray],  # TODO: unsure about type
    bucket_labels: list[str],
):
    """
    Build a pandas DataFrame with Par and bucketed DV01 per swap.
    """
    # Model par
    model_par = par_swap_rates_pooled(
        params_concat, static.knot_times, static.graph, static.batch
    )  # (S,)
    # DV01 buckets use current coupon (market par)
    coupon = jnp.array(coupon_per_swap_np, dtype=jnp.float64)
    dv01_bucket = dv01_bucket_fn(params_concat, coupon)  # (S,B)
    import pandas as pd

    S_true = int(static.batch.swap_count_true)
    ids = _make_swap_labels(static.batch, prefix="USD_SOFR_OIS")
    par_true = jnp.array(model_par[:S_true]).tolist()
    dv01_true = jnp.array(dv01_bucket[:S_true, :])
    df = pd.DataFrame({
        "instrument_id": ids,
        "Par": par_true,
    })
    for b, lbl in enumerate(bucket_labels):
        df[f"DV01_{lbl}"] = dv01_true[:, b].tolist()
    return df

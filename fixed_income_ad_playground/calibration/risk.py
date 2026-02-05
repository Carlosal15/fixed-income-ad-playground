from fixed_income_ad_playground.calibration.calibration import CalibStatic
from fixed_income_ad_playground.enums import SwapIndex
from fixed_income_ad_playground.kernels.risk import (
    bucketed_dv01_jvp,
    make_bucket_shocks,
    pv_jacobian_wrt_params,
)
from fixed_income_ad_playground.packing.swaps import PackedSwapBatchPooled
from fixed_income_ad_playground.types import FloatNDArray, JaxArray
import jax
import jax.numpy as jnp
import numpy as np


def build_risk_engines(static: CalibStatic, bucket_edges: list[float]):
    """
    Pre-build jitted risk functions for a fixed (shape, graph, knot grid):
      - jacobian of PV wrt parameters
      - bucketed DV01 via JVP
    """
    jac_pv = pv_jacobian_wrt_params(static.knot_times, static.graph, static.batch)
    bucket_dirs = make_bucket_shocks(static.knot_times, static.param_curve_ids, bucket_edges)

    def dv01_bucket(params_concat: JaxArray, coupon: JaxArray) -> JaxArray:
        return bucketed_dv01_jvp(
            params_concat, static.knot_times, static.graph, static.batch, coupon, bucket_dirs
        )

    dv01_bucket_jit = jax.jit(dv01_bucket)
    return jac_pv, dv01_bucket_jit


# Helper: pad a length-S_true vector to bucketed S with zeros beyond true count
def _bucketize_vector(vec_np: FloatNDArray, batch: PackedSwapBatchPooled) -> JaxArray:
    S = int(batch.swaps_bucket_size)
    S_true = int(batch.swap_count_true)
    out = np.zeros((S,), dtype=np.float64)
    k = min(S_true, int(vec_np.size))
    if k > 0:
        out[:k] = np.asarray(vec_np[:k], dtype=np.float64)
    return jnp.array(out)


def _format_tenor_label(years: float) -> str:
    y = float(years)
    if y < 1.0:
        m = round(y * 12)
        return f"{m}M"
    else:
        yi = round(y)
        return f"{yi}Y"


# TODO: move to demo? restrict prefix later
def _make_swap_labels(
    batch: PackedSwapBatchPooled, prefix: SwapIndex | str = "USD_SOFR_OIS"
) -> list[str]:
    labels: list[str] = []
    S_true = int(batch.swap_count_true)
    start_np = np.asarray(batch.start_times, dtype=np.float64)
    end_np = np.asarray(batch.end_times, dtype=np.float64)
    mask_np = np.asarray(batch.cashflow_mask, dtype=np.float64)
    for i in range(S_true):
        k = int(np.count_nonzero(mask_np[i] > 0.0))
        if k <= 0:
            labels.append(f"{prefix}_UNKNOWN")
            continue
        start = float(start_np[i, 0])
        end_last = float(end_np[i, k - 1])
        tenor_years = max(end_last - start, 0.0)
        if start <= 1e-12:
            labels.append(f"{prefix}_{_format_tenor_label(tenor_years)}")
        else:
            labels.append(
                f"{prefix}_Fwd{_format_tenor_label(start)}x{_format_tenor_label(tenor_years)}"
            )
    return labels

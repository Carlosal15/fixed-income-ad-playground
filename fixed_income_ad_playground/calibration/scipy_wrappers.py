from fixed_income_ad_playground.types import FloatNDArray

from fixed_income_ad_playground.packing.combined import PackedMarketBatches
from fixed_income_ad_playground.kernels.residuals import RES_JIT, JAC_JIT
from typing import Callable
import numpy as np
import jax.numpy as jnp
import time
from fixed_income_ad_playground.calibration.calibrator import (
    _raise_with_block_debug,
    FitDiagnostics,
)
from fixed_income_ad_playground.curve.curve_config import (
    CurveSetConfig,
    CurveSpec,
    CalibratedCurveParams,
)

from scipy.optimize import least_squares


def make_scipy_wrappers(
    knot_times_np: FloatNDArray,
    batches: PackedMarketBatches,
    market_par_swaps_np: FloatNDArray,
    market_quotes_futures_np: FloatNDArray,
    lam_slope: float,
    lam_curv: float,
) -> tuple[Callable[[FloatNDArray], FloatNDArray], Callable[[FloatNDArray], FloatNDArray]]:
    knot_times = jnp.array(np.asarray(knot_times_np, dtype=np.float64))

    swap_batch = batches.swaps
    futures_batch = batches.futures

    market_par_swaps = jnp.array(np.asarray(market_par_swaps_np, dtype=np.float64))
    market_quotes_futures = jnp.array(np.asarray(market_quotes_futures_np, dtype=np.float64))

    swap_count = int(swap_batch.maturity_times.shape[0])
    future_count = int(futures_batch.expiry_times.shape[0])
    interval_count = int(knot_times_np.size - 1)
    slope_count = max(interval_count - 1, 0)
    curv_count = max(interval_count - 2, 0)

    block_slices: dict[str, tuple[int, int]] = {
        "swaps": (0, swap_count),
        "futures": (swap_count, swap_count + future_count),
        "slope_penalty": (swap_count + future_count, swap_count + future_count + slope_count),
        "curv_penalty": (
            swap_count + future_count + slope_count,
            swap_count + future_count + slope_count + curv_count,
        ),
    }

    def fun_np(params_np: FloatNDArray) -> FloatNDArray:
        forward_params = jnp.array(np.asarray(params_np, dtype=np.float64))
        residuals = RES_JIT(
            forward_params,
            market_par_swaps,
            swap_batch.weights,
            knot_times,
            swap_batch.payment_times,
            swap_batch.accrual_factors,
            swap_batch.cashflow_mask,
            swap_batch.maturity_times,
            market_quotes_futures,
            futures_batch.weights,
            futures_batch.expiry_times,
            lam_slope,
            lam_curv,
        )
        residuals_np = np.asarray(residuals, dtype=np.float64)
        if not np.all(np.isfinite(residuals_np)):
            _raise_with_block_debug(residuals_np, block_slices)
        return residuals_np

    def jac_np(params_np: FloatNDArray) -> FloatNDArray:
        forward_params = jnp.array(np.asarray(params_np, dtype=np.float64))
        jac = JAC_JIT(
            forward_params,
            market_par_swaps,
            swap_batch.weights,
            knot_times,
            swap_batch.payment_times,
            swap_batch.accrual_factors,
            swap_batch.cashflow_mask,
            swap_batch.maturity_times,
            market_quotes_futures,
            futures_batch.weights,
            futures_batch.expiry_times,
            lam_slope,
            lam_curv,
        )
        jac_np_out = np.asarray(jac, dtype=np.float64)
        if not np.all(np.isfinite(jac_np_out)):
            raise ValueError("Non-finite Jacobian detected (unexpected).")
        return jac_np_out

    return fun_np, jac_np


def calibrate_curve_one_date(
    label: str,
    spec: CurveSpec,
    cfg: CurveSetConfig,
    batches: PackedMarketBatches,
    market_par_swaps_np: FloatNDArray,
    market_quotes_futures_np: FloatNDArray,
    x0: FloatNDArray,
    do_warmup: bool = True,
    max_nfev: int = 80,
    bench_reps: int = 20,
) -> tuple[CalibratedCurveParams, FitDiagnostics]:
    knot_times_np = spec.knot_times
    interval_count = knot_times_np.size - 1
    swap_count = int(batches.swaps.maturity_times.shape[0])
    future_count = int(batches.futures.expiry_times.shape[0])

    fun_np, jac_np = make_scipy_wrappers(
        knot_times_np,
        batches,
        market_par_swaps_np,
        market_quotes_futures_np,
        cfg.lam_slope,
        cfg.lam_curv,
    )

    warmup_seconds = 0.0
    if do_warmup:
        t0 = time.perf_counter()
        _ = fun_np(x0)
        _ = jac_np(x0)
        warmup_seconds = time.perf_counter() - t0

    # steady-state wrapper timings
    t0 = time.perf_counter()
    for _ in range(bench_reps):
        _ = fun_np(x0)
    residual_eval_ms = 1e3 * (time.perf_counter() - t0) / bench_reps

    t1 = time.perf_counter()
    for _ in range(bench_reps):
        _ = jac_np(x0)
    jac_eval_ms = 1e3 * (time.perf_counter() - t1) / bench_reps

    t2 = time.perf_counter()
    result = least_squares(
        fun_np,
        x0,
        jac=jac_np,  # type: ignore
        method="trf",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=max_nfev,
    )
    solve_seconds = time.perf_counter() - t2

    # Diagnostics (split residual blocks)
    residual_vec = result.fun

    slope_count = max(interval_count - 1, 0)
    curv_count = max(interval_count - 2, 0)

    start_swaps = 0
    end_swaps = start_swaps + swap_count

    start_fut = end_swaps
    end_fut = start_fut + future_count

    start_slope = end_fut
    end_slope = start_slope + slope_count

    start_curv = end_slope
    end_curv = start_curv + curv_count

    residuals_swaps = residual_vec[start_swaps:end_swaps]
    residuals_slope = residual_vec[start_slope:end_slope]
    residuals_curv = residual_vec[start_curv:end_curv]

    swap_rms_bp = float(np.sqrt(np.mean(residuals_swaps**2)) * 1e4) if residuals_swaps.size else 0.0
    swap_max_bp = float(np.max(np.abs(residuals_swaps)) * 1e4) if residuals_swaps.size else 0.0
    slope_rms = float(np.sqrt(np.mean(residuals_slope**2))) if residuals_slope.size else 0.0
    curv_rms = float(np.sqrt(np.mean(residuals_curv**2))) if residuals_curv.size else 0.0

    print(f"\n=== {label} ===")
    print("success:", result.success, result.message)
    print("nfev:", result.nfev, "njev:", result.njev)
    print("warmup seconds:", warmup_seconds)
    print("avg residual eval (ms):", residual_eval_ms)
    print("avg jacobian eval (ms):", jac_eval_ms)
    print("solve seconds:", solve_seconds)
    print("swap fit: RMS(bp) =", swap_rms_bp, "max(|bp|) =", swap_max_bp)
    if cfg.lam_slope > 0.0:
        print("slope penalty RMS:", slope_rms)
    if cfg.lam_curv > 0.0:
        print("curvature penalty RMS:", curv_rms)
    if future_count > 0:
        print("NOTE: futures block exists but is currently a stub (zeros).")

    params = CalibratedCurveParams(
        forward_params_per_interval=np.asarray(result.x, dtype=np.float64)
    )
    diag = FitDiagnostics(
        swap_rms_bp=swap_rms_bp,
        swap_max_bp=swap_max_bp,
        slope_rms=slope_rms,
        curv_rms=curv_rms,
    )
    return params, diag

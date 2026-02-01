import numpy as np
from fixed_income_ad_playground.calibration.calibration import DebugStats
from fixed_income_ad_playground.types import FloatNDArray
from dataclasses import dataclass


def build_maturities_years(n: int, maxT: float) -> FloatNDArray:
    # evenly spaced maturities (demo)
    return np.linspace(maxT / n, maxT, n).astype(np.float64)


def make_true_forward(
    knot_times: FloatNDArray, level: float, slope: float, wiggle: float
) -> FloatNDArray:
    mid = 0.5 * (knot_times[:-1] + knot_times[1:])
    return (level + slope * (1.0 - np.exp(-mid)) + wiggle * np.sin(2.0 * np.pi * mid / 5.0)).astype(
        np.float64
    )


def swap_fit_bp_from_residuals(residual: np.ndarray, swap_count_true: int) -> tuple[float, float]:
    r = residual[:swap_count_true]
    rms = float(np.sqrt(np.mean(r * r)) * 1e4) if r.size else 0.0
    mx = float(np.max(np.abs(r)) * 1e4) if r.size else 0.0
    return rms, mx


@dataclass(frozen=True)
class BenchLine:
    name: str
    compile_s: float
    median_ms: float
    p90_ms: float
    max_ms: float
    rms_bp_last: float
    max_bp_last: float
    stats_first: DebugStats
    stats_last: DebugStats

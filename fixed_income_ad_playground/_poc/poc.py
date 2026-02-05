"""
JAX (+ scipy optimiser) curve calibration POC

MVP example of an E2E jax calibration just for testing performance.
Uses just swap-like toy instruments with a simple single curve.
Very simple model just to test jax kernel and calibration performance.
For ~30 or so instruments, jitted calibration is about 10ms. Jacobian eval is 0.2ms
For ~500 instruments (not shown here but easy to tweak), jacobian eval is about 5ms as-is, and calibration ~100ms.
Jacobian eval is the main cost on larger problems.
Some more optimisations and calibration settings could be tweaked to improve further.

# Just for clarity, here are the simplifications used in this POC:
- Instruments: 1W..4W, 1M..12M, 2Y..10Y
- Year fractions assumed ACT/365 and times are year-fractions from "today"
- Fixed-leg schedule model:
    * For maturities <= 1Y: single payment at maturity (bullet)
    * For maturities > 1Y: annual fixed payments
- Knots: one knot per swap maturity
- Spot lag as simple time shift by spot_lag_days/365
- Residuals = [swap par fit residuals] + [optional penalty residual blocks]
- residuals and jacobian jitted _once_ and reused across dates (assuming same shape)

"""

from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

FloatNDArray = npt.NDArray[np.float64]


# Toy instrument grid


def yearfrac_from_days(days: float) -> float:
    # using ACT/365 convention for simplicity
    return float(days) / 365.0


def build_standard_swap_maturities_years(spot_lag_days: int = 2) -> FloatNDArray:
    """
    Returns dcf to maturity for weekly, then monthly, then yearly swaps.

    Useful as a toy implicit knot config for poc
    """
    spot = yearfrac_from_days(spot_lag_days)

    weeks = np.array([1, 2, 3, 4], dtype=np.float64) * (7.0 / 365.0)
    months = np.arange(1, 13, dtype=np.float64) * (30.0 / 365.0)  # toy month = 30d
    years = np.arange(2, 11, dtype=np.float64)

    mats = np.concatenate([weeks, months, years], axis=0) + spot
    return mats.astype(np.float64)


# schedules, packed for jax


def build_swap_cashflows(
    maturities: FloatNDArray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build fixed-leg schedules for a batch of par swaps.

    NOTE: _very_ simple toy schedule generator. It assumes yearly payments,
    with a single payment for maturities <= 1Y. We don't handle business days, stubs, other
    frequencies, etc. It doesn't even properly handle something like a 18M maturity.

    TODO: Write todo's for this.

    Convention (toy):
    - If maturity <= 1Y: single payment at maturity with accrual = maturity
    - If maturity > 1Y: annual payments at 1Y,2Y,...,T with accrual=1.0
        (last payment adjusted to exactly T, accrual stays 1.0 in this toy)

    Returns (JAX arrays):
    pay_times: (n, max_p)
    accruals:  (n, max_p)
    mask:      (n, max_p)
    maturities:(n,)
    """
    # maturities = np.asarray(maturities, dtype=np.float64)
    n = maturities.size

    # number of payments per swap
    # TODO: proper schedule generator
    n_pay = np.where(maturities <= 1.0, 1, np.ceil(maturities).astype(int))
    max_p = int(n_pay.max())

    # NOTE: would it be better to use some conservatively large minimum value for max_p, so
    # that shapes don't change across dates with different instrument sets, and compilation can be reused?
    pay_times = np.zeros((n, max_p), dtype=np.float64)
    accruals = np.zeros((n, max_p), dtype=np.float64)
    mask = np.zeros((n, max_p), dtype=np.float64)

    for i in range(n):
        maturity = float(maturities[i])
        number_of_payments = int(n_pay[i])

        if maturity <= 1.0:
            pay_times[i, 0] = maturity
            accruals[i, 0] = maturity
            mask[i, 0] = 1.0
        else:
            # annual schedule 1,2,...,k-1, T
            times = np.arange(1, number_of_payments + 1, dtype=np.float64)
            times[-1] = maturity
            pay_times[i, :number_of_payments] = times
            accruals[i, :number_of_payments] = 1.0
            mask[i, :number_of_payments] = 1.0

    return (
        jnp.array(pay_times),
        jnp.array(accruals),
        jnp.array(mask),
        jnp.array(maturities),
    )


# Curve calculations


# TODO: To be owned by a curve interface that hides specific implementation,
# interpolation type, etc. Should be just .discount_factors(query_times)
# maybe have a generic one with power basis polynomials used for calibration via jax,
# and once frozen, pass it to scipy ppoly for simplicity / speed / avoid extra jit for
# pricing?
# May depend on the pricing model. Jax could also be useful with compilation, implicit vectorisation, etc.
def dfs_from_stepwise_const_forwards(
    knot_times: jnp.ndarray,
    knot_values: jnp.ndarray,  # (m,)
    query_times: jnp.ndarray,  # (Q,)
) -> jnp.ndarray:
    """
    Discount factors for stepwise-constant instantaneous forwards.

    knot_values[i] applies on [knot_times[i], knot_times[i+1]). Assumes inputs are
    sorted.
    """
    dt = jnp.diff(knot_times)
    cum_int = jnp.cumsum(knot_values * dt)

    idx = jnp.searchsorted(knot_times, query_times, side="right") - 1

    # baked-in safety, errors should be handled at api level in prod before hitting here
    idx = jnp.clip(idx, 0, knot_values.size - 1)

    int_to_prev = jnp.where(idx > 0, cum_int[idx - 1], 0.0)
    partial = knot_values[idx] * (query_times - knot_times[idx])
    return jnp.exp(-(int_to_prev + partial))


def par_swap_rates_from_knot_values(
    knot_values: jnp.ndarray,  # (m,)
    knot_times: jnp.ndarray,  # (m+1,)
    pay_times: jnp.ndarray,  # (n, max_p)
    accruals: jnp.ndarray,  # (n, max_p)
    mask: jnp.ndarray,  # (n, max_p)
    maturities: jnp.ndarray,  # (n,)
) -> jnp.ndarray:
    """
    Par swap rate (single curve):
        S = (1 - P(T)) / sum_i alpha_i P(t_i)
    """
    n, max_p = pay_times.shape
    flat_times = pay_times.reshape(-1)  # (n*max_p,)

    dfs_flat = dfs_from_stepwise_const_forwards(knot_times, knot_values, flat_times)
    dfs_pay = dfs_flat.reshape(n, max_p)

    df_T = dfs_from_stepwise_const_forwards(knot_times, knot_values, maturities)
    annuity = jnp.sum(mask * accruals * dfs_pay, axis=1)
    return (1.0 - df_T) / annuity


# penalties


def penalty_slope(f: jnp.ndarray, knot_times: jnp.ndarray, lam: float) -> jnp.ndarray:
    """
    Fixed-shape slope penalty residuals (m-1,).
    If lam == 0, this returns all zeros but shape is unchanged (jax friendly)
    """
    dt = knot_times[1:] - knot_times[:-1]  # (m,)
    df = f[1:] - f[:-1]  # (m-1,)
    scale = jnp.sqrt(jnp.maximum(lam, 0.0))
    return scale * (df / dt[:-1])  # (m-1,)


def penalty_curvature(f: jnp.ndarray, knot_times: jnp.ndarray, lam: float) -> jnp.ndarray:
    """
    Fixed-shape curvature penalty residuals (m-2,).
    If lam == 0, returns zeros but shape is unchanged (jax friendly)
    """
    dt = knot_times[1:] - knot_times[:-1]  # (m,)
    d2 = f[2:] - 2.0 * f[1:-1] + f[:-2]  # (m-2,)
    scale = jnp.sqrt(jnp.maximum(lam, 0.0))
    return scale * (d2 / (dt[1:-1] ** 2))  # (m-2,)


# TODO: may want better handling/more versatile penalties, but ok for poc
# probably wrapped too into a numpy-layer that converts to jax arrays, etc.
def residuals(
    knot_values: jnp.ndarray,
    market_par: jnp.ndarray,
    knot_times: jnp.ndarray,
    pay_times: jnp.ndarray,
    accruals: jnp.ndarray,
    mask: jnp.ndarray,
    maturities: jnp.ndarray,
    lam_slope: float,
    lam_curv: float,
) -> jnp.ndarray:
    """
    Returns a fixed-length residual vector with n_inst + (m-1) + (m-2)
    Jit-friendly recompiles due to shape changes.
    """
    model_par = par_swap_rates_from_knot_values(
        knot_values, knot_times, pay_times, accruals, mask, maturities
    )
    r_inst = model_par - market_par  # (n_inst,)

    r_s = penalty_slope(knot_values, knot_times, lam_slope)  # (m-1,)
    r_c = penalty_curvature(knot_values, knot_times, lam_curv)  # (m-2,)

    return jnp.concatenate([r_inst, r_s, r_c], axis=0)


# JIT compiled once; market_par is an argument => reuse across dates with same shapes
residuals_jitted = jax.jit(residuals)
residual_jacobian_jitted = jax.jit(jax.jacfwd(residuals, argnums=0))  # Jacobian wrt knot values


# scipy convenience wrappers


@dataclass(frozen=True)
class FitDiagnostics:
    inst_rms_bp: float
    inst_max_bp: float
    slope_rms: float
    curv_rms: float


def split_residual_blocks(
    r: FloatNDArray, n_inst: int, n_slope: int, n_curv: int
) -> tuple[FloatNDArray, FloatNDArray, FloatNDArray]:
    r_inst = r[:n_inst]
    r_slope = r[n_inst : n_inst + n_slope]
    r_curv = r[n_inst + n_slope : n_inst + n_slope + n_curv]
    return r_inst, r_slope, r_curv


def make_scipy_wrappers(
    knot_times_np: FloatNDArray,
    pay_times: jnp.ndarray,
    accruals: jnp.ndarray,
    mask: jnp.ndarray,
    maturities: jnp.ndarray,
    market_par_np: FloatNDArray,
    lam_slope: float,
    lam_curv: float,
) -> tuple[Callable[[FloatNDArray], FloatNDArray], Callable[[FloatNDArray], FloatNDArray]]:
    knot_times = jnp.array(np.asarray(knot_times_np, dtype=np.float64))
    market_par = jnp.array(np.asarray(market_par_np, dtype=np.float64))

    def fun_np(x: FloatNDArray) -> FloatNDArray:
        f = jnp.array(np.asarray(x, dtype=np.float64))
        r = residuals_jitted(
            f, market_par, knot_times, pay_times, accruals, mask, maturities, lam_slope, lam_curv
        )
        return np.asarray(r, dtype=np.float64)

    def jac_np(x: FloatNDArray) -> FloatNDArray:
        f = jnp.array(np.asarray(x, dtype=np.float64))
        J = residual_jacobian_jitted(
            f, market_par, knot_times, pay_times, accruals, mask, maturities, lam_slope, lam_curv
        )
        return np.asarray(J, dtype=np.float64)

    return fun_np, jac_np


def calibrate_one_date(
    label: str,
    knot_times: FloatNDArray,
    pay_times: jnp.ndarray,
    accruals: jnp.ndarray,
    mask: jnp.ndarray,
    maturities: jnp.ndarray,
    market_par: FloatNDArray,
    x0: FloatNDArray,
    lam_slope: float,
    lam_curv: float,
    do_warmup: bool = True,
    max_nfev: int = 80,
    bench_reps: int = 20,
) -> tuple[FloatNDArray, FitDiagnostics]:
    fun_np, jac_np = make_scipy_wrappers(
        knot_times, pay_times, accruals, mask, maturities, market_par, lam_slope, lam_curv
    )

    warmup_s = 0.0
    if do_warmup:
        t0 = time.perf_counter()
        _ = fun_np(x0)
        _ = jac_np(x0)
        warmup_s = time.perf_counter() - t0

    # benchmark evals (includes wrapper conversion overhead)
    t0 = time.perf_counter()
    for _ in range(bench_reps):
        _ = fun_np(x0)
    res_ms = 1e3 * (time.perf_counter() - t0) / bench_reps

    t1 = time.perf_counter()
    for _ in range(bench_reps):
        _ = jac_np(x0)
    jac_ms = 1e3 * (time.perf_counter() - t1) / bench_reps

    t2 = time.perf_counter()
    res = least_squares(
        fun_np,
        x0,
        jac=jac_np,  # type: ignore
        method="trf",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=max_nfev,
    )
    solve_s = time.perf_counter() - t2

    # Diagnostics: split residual blocks
    n_inst = int(maturities.shape[0])
    m = int(knot_times.size - 1)
    n_slope = m - 1
    n_curv = m - 2

    r_inst, r_slope, r_curv = split_residual_blocks(res.fun, n_inst, n_slope, n_curv)

    inst_rms_bp = float(np.sqrt(np.mean(r_inst**2)) * 1e4)
    inst_max_bp = float(np.max(np.abs(r_inst)) * 1e4)
    slope_rms = float(np.sqrt(np.mean(r_slope**2))) if r_slope.size else 0.0
    curv_rms = float(np.sqrt(np.mean(r_curv**2))) if r_curv.size else 0.0

    print(f"\n=== {label} ===")
    print("success:", res.success, res.message)
    print("nfev:", res.nfev, "njev:", res.njev)
    print("warmup seconds:", warmup_s)
    print("avg residual eval (ms):", res_ms)
    print("avg jacobian eval (ms):", jac_ms)
    print("solve seconds:", solve_s)
    print("instrument fit: RMS(bp) =", inst_rms_bp, "max(|bp|) =", inst_max_bp)
    if lam_slope > 0.0:
        print("slope penalty RMS:", slope_rms)
    if lam_curv > 0.0:
        print("curvature penalty RMS:", curv_rms)

    diag = FitDiagnostics(
        inst_rms_bp=inst_rms_bp, inst_max_bp=inst_max_bp, slope_rms=slope_rms, curv_rms=curv_rms
    )
    return np.asarray(res.x, dtype=np.float64), diag


# synthetic data generation


def make_true_forward_on_knots(
    knot_times: FloatNDArray,
    level: float,
    slope: float,
    wiggle: float,
) -> FloatNDArray:
    """
    Define a smooth-ish true forward curve sampled per interval midpoint.
    Returns f_true (m,) aligned with the knot grid.
    """
    t_mid = 0.5 * (knot_times[:-1] + knot_times[1:])
    f = level + slope * (1.0 - np.exp(-t_mid)) + wiggle * np.sin(2.0 * np.pi * t_mid / 5.0)
    return f.astype(np.float64)


def make_market_par(
    f_true: FloatNDArray,
    knot_times: FloatNDArray,
    pay_times: jnp.ndarray,
    accruals: jnp.ndarray,
    mask: jnp.ndarray,
    maturities: jnp.ndarray,
    noise_bps: float,
    seed: int,
) -> FloatNDArray:
    knot_j = jnp.array(np.asarray(knot_times, dtype=np.float64))
    f_j = jnp.array(np.asarray(f_true, dtype=np.float64))
    par = par_swap_rates_from_knot_values(f_j, knot_j, pay_times, accruals, mask, maturities)
    par_np = np.asarray(par, dtype=np.float64)

    if noise_bps > 0.0:
        rng = np.random.default_rng(seed)
        par_np = par_np + (noise_bps * 1e-4) * rng.normal(size=par_np.shape)

    return par_np


# ============================================================
# Plotting
# ============================================================


def plot_forward_and_zero(
    knot_times: FloatNDArray,
    curves_f: list[FloatNDArray],
    labels: list[str],
) -> None:
    knot_times = np.asarray(knot_times, dtype=np.float64)
    t0 = knot_times[:-1]
    t1 = knot_times[1:]
    t_step = np.column_stack([t0, t1]).reshape(-1)

    plt.figure(figsize=(10, 4))
    for f, lab in zip(curves_f, labels, strict=True):
        f = np.asarray(f, dtype=np.float64)
        f_step = np.column_stack([f, f]).reshape(-1)
        plt.plot(t_step, f_step, label=lab)
    plt.xlabel("Maturity (years)")
    plt.ylabel("Instantaneous forward f(t)")
    plt.title("Forward Curves (knots at swap maturities)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    knot_j = jnp.array(knot_times)
    t_grid = jnp.linspace(0.01, float(knot_times[-1]), 800)
    for f, lab in zip(curves_f, labels, strict=True):
        f_j = jnp.array(np.asarray(f, dtype=np.float64))
        df = dfs_from_stepwise_const_forwards(knot_j, f_j, t_grid)
        z = -jnp.log(df) / t_grid
        plt.plot(np.asarray(t_grid), np.asarray(z), label=lab)
    plt.xlabel("Maturity (years)")
    plt.ylabel("Zero rate z(t)")
    plt.title("Implied Zero Curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


def main() -> None:
    # 1) Build mock curve swap
    spot_lag_days = 2
    maturities = build_standard_swap_maturities_years(spot_lag_days=spot_lag_days)
    maturities = np.sort(maturities)

    # 2) Knots: one per swap maturity ("last dependency date") + 0
    knot_times = np.concatenate([[0.0], maturities], axis=0).astype(np.float64)
    m = knot_times.size - 1

    # 3) Build packed cashflows (same instrument set for both dates)
    pay_times, accruals, mask, mats = build_swap_cashflows(maturities)

    print("n_swaps:", int(mats.shape[0]))
    print("max fixed payments per swap:", int(pay_times.shape[1]))
    print("curve params (m):", m)
    print("spot_lag_days:", spot_lag_days)

    # 4) True curves (defined on the same knots)
    f_true_1 = make_true_forward_on_knots(knot_times, level=0.040, slope=0.010, wiggle=0.002)
    f_true_2 = make_true_forward_on_knots(knot_times, level=0.030, slope=0.010, wiggle=0.003)

    # 5) Market quotes (par rates)
    NOISE_BPS = 0.1  # try 0.0 to see perfect recovery
    mkt_1 = make_market_par(
        f_true_1, knot_times, pay_times, accruals, mask, mats, noise_bps=NOISE_BPS, seed=1
    )
    mkt_2 = make_market_par(
        f_true_2, knot_times, pay_times, accruals, mask, mats, noise_bps=NOISE_BPS, seed=2
    )

    # 6) Initial guess
    x0 = 0.05 * np.ones(m, dtype=np.float64)

    # 7) Penalty config (multiple can be on at once)
    # Start with tiny penalties; increase if you see wiggles.
    lam_slope = 1e-4
    lam_curv = 1e-6

    # 8) Calibrate two dates (should reuse compilation because shapes match)
    t0 = time.perf_counter()
    f_cal_1, _ = calibrate_one_date(
        "Date 1",
        knot_times,
        pay_times,
        accruals,
        mask,
        mats,
        mkt_1,
        x0,
        lam_slope,
        lam_curv,
        do_warmup=True,
        max_nfev=80,
        bench_reps=20,
    )

    f_cal_2, _ = calibrate_one_date(
        "Date 2",
        knot_times,
        pay_times,
        accruals,
        mask,
        mats,
        mkt_2,
        x0,
        lam_slope,
        lam_curv,
        do_warmup=True,
        max_nfev=80,
        bench_reps=20,
    )
    print("\nTotal wall time:", time.perf_counter() - t0)

    # 9) Plot
    plot_forward_and_zero(
        knot_times,
        curves_f=[f_true_1, f_cal_1, f_true_2, f_cal_2],
        labels=["TRUE date1", "CAL date1", "TRUE date2", "CAL date2"],
    )


if __name__ == "__main__":
    main()

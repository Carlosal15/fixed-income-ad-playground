import time
import jax
import jax.numpy as jnp
import numpy as np
from fixed_income_ad_playground._demo.demo_cases import (
    BenchLine,
    build_maturities_years,
    make_true_forward,
    swap_fit_bp_from_residuals,
)
from fixed_income_ad_playground.calibration.calibration import (
    CurveDef,
    CurveSetCalibrator,
    CurveSetSpec,
    build_curve_graph_arrays,
    make_calib_static,
)
from fixed_income_ad_playground.curve.curve_config import CurveConfig
from fixed_income_ad_playground.enums import SwapIndex
from fixed_income_ad_playground.identifiers.identifiers import CurveId, InstrumentId
from fixed_income_ad_playground.instruments.instrument import PackContext
from fixed_income_ad_playground.instruments.swap import SwapSpec
from fixed_income_ad_playground.kernels.swaps import par_swap_rates_pooled
from fixed_income_ad_playground.packing.swaps import pack_swaps_pooled
from fixed_income_ad_playground.pricing.pricer import Pricer
from fixed_income_ad_playground.pricing.quote import Quote
from fixed_income_ad_playground.reference_data_container import ReferenceDataContainer
from fixed_income_ad_playground.shape_policy import ShapePolicy
from fixed_income_ad_playground.types import FloatNDArray


def main(  # noqa: C901
    case: str = "usd_50",
    n_dates: int = 20,
    warm_start: bool = True,
    noise_bps: float = 0.1,
    max_steps: int = 80,
    jac_mode: str = "fwd",
) -> None:
    """
    Notebook usage:
      main(case="usd_50", n_dates=20)
      main(case="usd_100", n_dates=20)
      main(case="eur_200", n_dates=20)
    """
    ctx = PackContext()
    shape_policy = ShapePolicy()

    # --------------------------
    # Build curveset + refdata mapping for the chosen case
    # --------------------------
    if case == "usd_50":
        label = "USD-2curve-50swaps"
        n_mats = 25
        maxT = 10.0

        SOFR = CurveId("USD_SOFR")
        SPRD = CurveId("USD_FF_SPRD")
        FF = CurveId("USD_FEDFUNDS")

        curve_defs = [
            CurveDef(SOFR, "param"),
            CurveDef(SPRD, "param"),
            CurveDef(FF, "lincomb", sources=((SOFR, 1.0), (SPRD, 1.0))),
        ]
        curve_configs = {
            SOFR: CurveConfig(SOFR, interp="stepwise_const_fwd", lam_slope=1e-4, lam_curv=1e-6),
            SPRD: CurveConfig(SPRD, interp="stepwise_const_fwd", lam_slope=1e-4, lam_curv=1e-6),
        }

        # Refdata mapping: convention -> (discount, forecast)
        refdata = ReferenceDataContainer({
            "USD_SOFR_OIS": (SOFR, SOFR),
            "USD_FF": (SOFR, FF),
        })

        # pricing conventions for demo swaps
        ois_idx = SwapIndex.USD_SOFR_OIS
        ff_idx = SwapIndex.USD_FF

    elif case == "usd_100":
        label = "USD-2curve-100swaps"
        n_mats = 50
        maxT = 30.0

        SOFR = CurveId("USD_SOFR")
        SPRD = CurveId("USD_FF_SPRD")
        FF = CurveId("USD_FEDFUNDS")

        curve_defs = [
            CurveDef(SOFR, "param"),
            CurveDef(SPRD, "param"),
            CurveDef(FF, "lincomb", sources=((SOFR, 1.0), (SPRD, 1.0))),
        ]
        curve_configs = {
            SOFR: CurveConfig(SOFR, interp="stepwise_const_fwd", lam_slope=1e-4, lam_curv=1e-6),
            SPRD: CurveConfig(SPRD, interp="stepwise_const_fwd", lam_slope=1e-4, lam_curv=1e-6),
        }

        refdata = ReferenceDataContainer({
            "USD_SOFR_OIS": (SOFR, SOFR),
            "USD_FF": (SOFR, FF),
        })

        ois_idx = "USD_SOFR_OIS"
        ff_idx = "USD_FF"

    elif case == "eur_200":
        label = "EUR-4curve-200swaps"
        n_mats = 50
        maxT = 30.0

        ESTR = CurveId("EUR_ESTR")
        EUR3M = CurveId("EUR_EURIBOR3M")
        EUR6M = CurveId("EUR_EURIBOR6M")
        BASIS = CurveId("EUR_BASIS")

        curve_defs = [
            CurveDef(ESTR, "param"),
            CurveDef(EUR3M, "param"),
            CurveDef(EUR6M, "param"),
            CurveDef(BASIS, "param"),
        ]
        curve_configs = {
            ESTR: CurveConfig(ESTR, interp="stepwise_const_fwd", lam_slope=1e-4, lam_curv=1e-6),
            EUR3M: CurveConfig(EUR3M, interp="stepwise_const_fwd", lam_slope=1e-4, lam_curv=1e-6),
            EUR6M: CurveConfig(EUR6M, interp="stepwise_const_fwd", lam_slope=1e-4, lam_curv=1e-6),
            BASIS: CurveConfig(BASIS, interp="stepwise_const_fwd", lam_slope=1e-4, lam_curv=1e-6),
        }

        refdata = ReferenceDataContainer({
            "EUR_ESTR_OIS": (ESTR, ESTR),
            "EUR_EURIBOR3M": (ESTR, EUR3M),
            "EUR_EURIBOR6M": (ESTR, EUR6M),
        })

        # for EUR demo we will use multiple indices
        ois_idx = "EUR_ESTR_OIS"
        ff_idx = "EUR_EURIBOR3M"  # just reuse variable name in the demo

    else:
        raise ValueError(f"unknown case={case}")

    spec = CurveSetSpec(curve_defs=curve_defs, curve_configs=curve_configs)
    curve_id_to_idx, graph = build_curve_graph_arrays(spec.curve_defs)

    # knot grid (demo: uniform)
    knot_times = np.linspace(0.0, maxT, n_mats + 1).astype(np.float64)
    knot_times[0] = 0.0
    mats = build_maturities_years(n_mats, maxT)

    # --------------------------
    # Build calibration quotes (python layer only)
    # --------------------------
    quotes: list[Quote] = []
    if case.startswith("usd"):
        # 2 blocks of swaps (SOFR OIS + FF)
        for m in mats:
            quotes.append(
                Quote(
                    SwapSpec(
                        InstrumentId(f"SOFR_{m:.3f}"), "USD", ois_idx, float(m), float_leg_freq="S"
                    ),
                    "par_rate",
                    0.0,
                )
            )
        for m in mats:
            quotes.append(
                Quote(
                    SwapSpec(
                        InstrumentId(f"FF_{m:.3f}"), "USD", ff_idx, float(m), float_leg_freq="S"
                    ),
                    "par_rate",
                    0.0,
                )
            )
    else:
        # 4 blocks of 50 swaps: OIS, 3M, 6M, (basis curve not used directly here)
        for m in mats:
            quotes.append(
                Quote(
                    SwapSpec(
                        InstrumentId(f"OIS_{m:.3f}"),
                        "EUR",
                        "EUR_ESTR_OIS",
                        float(m),
                        float_leg_freq="S",
                    ),
                    "par_rate",
                    0.0,
                )
            )
        for m in mats:
            quotes.append(
                Quote(
                    SwapSpec(
                        InstrumentId(f"3M_{m:.3f}"),
                        "EUR",
                        "EUR_EURIBOR3M",
                        float(m),
                        float_leg_freq="Q",
                    ),
                    "par_rate",
                    0.0,
                )
            )
        for m in mats:
            quotes.append(
                Quote(
                    SwapSpec(
                        InstrumentId(f"6M_{m:.3f}"),
                        "EUR",
                        "EUR_EURIBOR6M",
                        float(m),
                        float_leg_freq="S",
                    ),
                    "par_rate",
                    0.0,
                )
            )
        for m in mats:
            quotes.append(
                Quote(
                    SwapSpec(
                        InstrumentId(f"OIS2_{m:.3f}"),
                        "EUR",
                        "EUR_ESTR_OIS",
                        float(m),
                        float(m),
                        float_leg_freq="S",
                    ),
                    "par_rate",
                    0.0,
                )
            )

    # Pack swaps with shape policy (pooled)
    batch = pack_swaps_pooled(quotes, ctx, refdata, curve_id_to_idx, shape_policy)

    # Build static calibration payload
    static = make_calib_static(knot_times, graph, batch, spec)

    # initial guess + "true" params
    P = len(static.param_curve_ids)
    N = int(knot_times.size - 1)
    x0 = np.zeros((P * N,), dtype=np.float64) + 0.02

    true_blocks = []
    for k in range(P):
        true_blocks.append(
            make_true_forward(knot_times, level=0.03 + 0.002 * k, slope=0.01, wiggle=0.001)
        )
    params_true = np.concatenate(true_blocks, axis=0).astype(np.float64)

    print(f"\n=== Demo: {label} ===")
    print(f"case={case}  n_dates={n_dates}  warm_start={warm_start}")
    print(f"solver max_steps={max_steps}  jac_mode={jac_mode}")
    print(
        f"true shapes: swaps={batch.swap_count_true} max_cf={batch.max_cf_true} pooled_Q={batch.pooled_time_count_true}"
    )
    print(
        f"bucketed shape key: S={batch.swaps_bucket_size} M={batch.cashflows_bucket_size} Q={batch.unique_times_bucket_size} | curves={len(curve_defs)} param_curves={P} N={N} maxT={maxT}"
    )

    # --------------------------
    # Market par generation (compile once, reuse)
    # --------------------------
    def par_kernel(pvec):
        return par_swap_rates_pooled(pvec, static.knot_times, static.graph, static.batch)

    par_jit = jax.jit(par_kernel)

    t0 = time.perf_counter()
    base = np.asarray(par_jit(jnp.array(params_true)).block_until_ready(), dtype=np.float64)
    warm_par = time.perf_counter() - t0

    t1 = time.perf_counter()
    rng = np.random.default_rng(123)
    market_par_dates: list[FloatNDArray] = []
    for _ in range(n_dates):
        noise = (noise_bps * 1e-4) * rng.normal(size=base.shape)
        market_par_dates.append(base + noise)
    gen_t = time.perf_counter() - t1
    print(f"market generation: {gen_t:.3f}s (kernel warmup={warm_par:.3f}s)")

    # --------------------------
    # Calibrator warmup
    # --------------------------
    calib = CurveSetCalibrator(static)
    x0_j = jnp.array(x0)
    mp0_j = jnp.array(market_par_dates[0])
    warm_rj = calib.warmup_residuals_jacobian(x0_j, mp0_j)
    print(f"residual+jac warmup: {warm_rj:.3f}s")

    # --------------------------
    # Benchmark solvers
    # --------------------------
    benches: list[BenchLine] = []

    def run_bench_least_squares(name: str, solver_kind: str):
        run_jit, (solver_obj, fn) = calib.build_optx_least_squares_value_only(
            solver_kind, max_steps=max_steps, jac_mode=jac_mode
        )

        t0 = time.perf_counter()
        _ = run_jit(x0_j, mp0_j).block_until_ready()
        compile_s = time.perf_counter() - t0

        st_first = calib.debug_optx_least_squares_stats(
            solver_obj, fn, x0_j, mp0_j, max_steps=max_steps, jac_mode=jac_mode
        )

        timings = []
        x_curr = x0_j
        last_x = x0_j
        last_mp = mp0_j

        for i in range(n_dates):
            mp = jnp.array(market_par_dates[i])
            t = time.perf_counter()
            x_star = run_jit(x_curr, mp)
            x_star.block_until_ready()
            timings.append((time.perf_counter() - t) * 1e3)
            x_curr = x_star if warm_start else x0_j
            last_x = x_star
            last_mp = mp

        st_last = calib.debug_optx_least_squares_stats(
            solver_obj, fn, last_x, last_mp, max_steps=max_steps, jac_mode=jac_mode
        )

        res_last = np.asarray(calib.res_jit(last_x, last_mp).block_until_ready(), dtype=np.float64)
        rms_bp, mx_bp = swap_fit_bp_from_residuals(res_last, static.batch.swap_count_true)

        arr = np.asarray(timings, dtype=np.float64)
        benches.append(
            BenchLine(
                name=name,
                compile_s=compile_s,
                median_ms=float(np.median(arr)),
                p90_ms=float(np.quantile(arr, 0.9)),
                max_ms=float(np.max(arr)),
                rms_bp_last=rms_bp,
                max_bp_last=mx_bp,
                stats_first=st_first,
                stats_last=st_last,
            )
        )

    def run_bench_scalar(name: str, solver_kind: str):
        run_jit, (solver_obj, fn) = calib.build_optx_scalar_value_only(
            solver_kind, max_steps=max_steps
        )

        t0 = time.perf_counter()
        _ = run_jit(x0_j, mp0_j).block_until_ready()
        compile_s = time.perf_counter() - t0

        st_first = calib.debug_optx_scalar_stats(solver_obj, fn, x0_j, mp0_j, max_steps=max_steps)

        timings = []
        x_curr = x0_j
        last_x = x0_j
        last_mp = mp0_j

        for i in range(n_dates):
            mp = jnp.array(market_par_dates[i])
            t = time.perf_counter()
            x_star = run_jit(x_curr, mp)
            x_star.block_until_ready()
            timings.append((time.perf_counter() - t) * 1e3)
            x_curr = x_star if warm_start else x0_j
            last_x = x_star
            last_mp = mp

        st_last = calib.debug_optx_scalar_stats(
            solver_obj, fn, last_x, last_mp, max_steps=max_steps
        )

        res_last = np.asarray(calib.res_jit(last_x, last_mp).block_until_ready(), dtype=np.float64)
        rms_bp, mx_bp = swap_fit_bp_from_residuals(res_last, static.batch.swap_count_true)

        arr = np.asarray(timings, dtype=np.float64)
        benches.append(
            BenchLine(
                name=name,
                compile_s=compile_s,
                median_ms=float(np.median(arr)),
                p90_ms=float(np.quantile(arr, 0.9)),
                max_ms=float(np.max(arr)),
                rms_bp_last=rms_bp,
                max_bp_last=mx_bp,
                stats_first=st_first,
                stats_last=st_last,
            )
        )

    # Least squares (uses J internally)
    run_bench_least_squares("Optimistix LS-GN (uses J)", "gn")
    run_bench_least_squares("Optimistix LS-DOGLEG (uses J)", "dogleg")
    run_bench_least_squares("Optimistix LS-LM (uses J)", "lm")

    # Scalar (matrix-free-ish)
    run_bench_scalar("Optimistix NONLINEAR_CG (scalar)", "nonlinear_cg")
    run_bench_scalar("Optimistix LBFGS (scalar)", "lbfgs")

    # Print table
    print("\n--- Benchmark (compile excluded from steady-state; per-date solve times) ---")
    for b in benches:
        print(
            f"{b.name:35s} | compile={b.compile_s:7.3f}s | median={b.median_ms:9.3f}ms | "
            f"p90={b.p90_ms:9.3f}ms | max={b.max_ms:9.3f}ms | "
            f"fit(last): RMS={b.rms_bp_last:.4f}bp max={b.max_bp_last:.4f}bp"
        )
        print(
            f"    stats(first): steps={b.stats_first.steps} result={b.stats_first.result_code} | "
            f"stats(last): steps={b.stats_last.steps} result={b.stats_last.result_code}"
        )

    best = min(benches, key=lambda x: x.median_ms)
    print(f"\nBest median solver: {best.name} ({best.median_ms:.3f}ms)")

    # --------------------------
    # Build a Market from the best solver's LAST calibration (for pricing demo)
    # --------------------------
    # For simplicity: re-run the best solver once on the last date to get params,
    # then build the Market and price a couple of swaps.
    # (In your production workflow you'd keep the last_x from the bench loop.)
    last_mp = jnp.array(market_par_dates[-1])

    if "LS-" in best.name:
        # map back to solver kind
        if "GN" in best.name:
            kind = "gn"
        elif "DOGLEG" in best.name:
            kind = "dogleg"
        else:
            kind = "lm"
        run_jit, _ = calib.build_optx_least_squares_value_only(
            kind, max_steps=max_steps, jac_mode=jac_mode
        )
    else:
        if "NONLINEAR_CG" in best.name:
            run_jit, _ = calib.build_optx_scalar_value_only("nonlinear_cg", max_steps=max_steps)
        else:
            run_jit, _ = calib.build_optx_scalar_value_only("lbfgs", max_steps=max_steps)

    x_star = run_jit(x0_j, last_mp).block_until_ready()
    market = calib.build_market(x_star)

    pricer = Pricer()

    # Pricing demo instruments (python layer)
    if case.startswith("usd"):
        test_swaps = [
            SwapSpec(InstrumentId("TEST_SOFR_5Y"), "USD", "USD_SOFR_OIS", 5.0, float_leg_freq="S"),
            SwapSpec(InstrumentId("TEST_FF_5Y"), "USD", "USD_FF", 5.0, float_leg_freq="S"),
        ]
    else:
        test_swaps = [
            SwapSpec(InstrumentId("TEST_ESTR_7Y"), "EUR", "EUR_ESTR_OIS", 7.0, float_leg_freq="S"),
            SwapSpec(InstrumentId("TEST_3M_7Y"), "EUR", "EUR_EURIBOR3M", 7.0, float_leg_freq="Q"),
        ]

    print("\n--- Pricing demo off calibrated Market (python layer) ---")
    for s in test_swaps:
        par = pricer.par_rate(market, s, ctx, refdata)
        print(f"{s.instrument_id.name:16s} | index={s.index:14s} | par={par * 1e4:9.3f} bp")

    # Optional: show a small sanity check on curve outputs
    # (No plotting to keep this minimal and fast; add matplotlib plot if you want.)
    t_grid = jnp.array(np.linspace(0.5, float(maxT), 6), dtype=jnp.float64)
    first_curve = next(iter(market.curves.values()))
    z = np.asarray(first_curve.zero_rate(t_grid))
    print(f"\nSample zeros for curve {first_curve.curve_id.name}:")
    for t, zz in zip(np.asarray(t_grid), z, strict=False):
        print(f"  t={t:5.2f}y  z={zz * 100:8.4f}%")


if __name__ == "__main__":
    main(case="usd_50", n_dates=20, warm_start=True, noise_bps=0.1, max_steps=80, jac_mode="fwd")

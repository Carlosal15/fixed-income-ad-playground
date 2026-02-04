from fixed_income_ad_playground.calibration.calibration import (
    CalibStatic,
    CurveDef,
    CurveSetCalibrator,
    CurveSetSpec,
    build_curve_graph_arrays,
    make_calib_static,
)
from fixed_income_ad_playground.calibration.risk import _bucketize_vector, build_risk_engines
from fixed_income_ad_playground.curve.curve_config import CurveConfig
from fixed_income_ad_playground.identifiers.identifiers import CurveId, InstrumentId
from fixed_income_ad_playground.instruments.instrument import PackContext
from fixed_income_ad_playground.instruments.swap import SwapSpec
from fixed_income_ad_playground.packing.swaps import pack_swaps_pooled
from fixed_income_ad_playground.pricing.quote import Quote
from fixed_income_ad_playground.pricing.risk_pricer import (
    portfolio_par_and_risk_df,
    portfolio_pv_and_risk_df,
)
from fixed_income_ad_playground.reference_data_container import ReferenceDataContainer
from fixed_income_ad_playground.shape_policy import ShapePolicy
from fixed_income_ad_playground.types import JaxArray
from fixed_income_ad_playground.types import FloatNDArray
from collections.abc import Sequence
import time
import numpy as np

from IPython.display import display, clear_output
import jax.numpy as jnp


def demo_streaming_risk_from_static(
    static: CalibStatic,
    x0_np: FloatNDArray,
    market_par_base_np: FloatNDArray,
    coupon_per_swap_np: FloatNDArray,
    max_steps: int = 50,
    jac_mode: str = "fwd",
    noise_bps: float = 0.1,
    iterations: int = 20,
    sleep_seconds: float = 1.0,
    bucket_edges: Sequence[float] = (0.0, 1.0, 3.0, 5.0, 10.0, 30.0),
    display_mode: str = "pv",
):
    """
    Streaming calibration loop using a fixed static payload. At each tick:
      - perturb market par rates by small noise
      - solve calibration (warm-start)
      - compute PV and bucketed DV01 and print a compact DataFrame
    """

    calib = CurveSetCalibrator(static)
    # Use Levenberg-Marquardt for stability in streaming
    run_jit, _ = calib.build_optx_least_squares_value_only(
        solver_kind="lm", max_steps=max_steps, jac_mode=jac_mode
    )

    # Build risk engines (coupon passed per tick)
    jac_pv_fn, dv01_bucket_fn = build_risk_engines(static, list(bucket_edges))
    bucket_labels = [
        f"{bucket_edges[i]}-{bucket_edges[i + 1]}Y" for i in range(len(bucket_edges) - 1)
    ]

    x = jnp.array(np.asarray(x0_np, dtype=np.float64))
    # Pad baseline market pars to bucketed S
    market_par_base = _bucketize_vector(market_par_base_np, static.batch)

    # Warm up JITs to avoid first-iteration compilation effects
    _ = calib.warmup_residuals_jacobian(x, market_par_base)

    for t in range(int(iterations)):
        noise = (float(noise_bps) / 1e4) * jnp.array(
            np.random.randn(static.batch.swaps_bucket_size), dtype=jnp.float64
        )
        market_par: JaxArray = market_par_base + noise

        # Solve for params (value-only jitted function returns params)
        x = run_jit(x, market_par)

        # Use current market par as coupon so par swaps have PV≈0
        coupon_current: JaxArray = market_par
        if display_mode == "par":
            df = portfolio_par_and_risk_df(x, static, coupon_current, dv01_bucket_fn, bucket_labels)
        else:
            df, _ = portfolio_pv_and_risk_df(
                x, static, coupon_current, jac_pv_fn, dv01_bucket_fn, bucket_labels
            )

        # Compute objective for reporting
        # r = calib.res_jit(x, market_par)
        # obj = 0.5 * jnp.sum(r * r)
        # Stream update in-place: add tick index and re-render
        df["tick index"] = int(t)
        # (objective hidden per request)
        clear_output(wait=True)
        display(df.round(6))
        time.sleep(float(sleep_seconds))


def main_risk(
    static: CalibStatic | None = None,
    quotes: Sequence[Quote] | None = None,
    x0_np: FloatNDArray | None = None,
    iterations: int = 5,
    noise_bps: float = 0.2,
    max_steps: int = 50,
    jac_mode: str = "fwd",
    bucket_edges: Sequence[float] = (0.0, 1.0, 3.0, 5.0, 10.0, 30.0),
    n_mats: int = 20,
    maxT: float = 10.0,
    sleep_seconds: float = 0.25,
    # case: str = "usd_default", # TODO: add others
) -> None:
    """
    Simple risk demo entrypoint.

    Expects a pre-built static payload and quotes (same order as packed batch).
    Builds coupons from quotes and runs a short streaming calibration loop
    while printing PV + bucketed DV01 per swap.

    Usage (in a notebook, after running benchmark setup):
      from fixed_income_ad_playground.scratch.curveset_dev.curvesets_attempt_with_risk import main_risk
      main_risk(static, quotes, x0_np, iterations=5)
    """

    # If no inputs provided, build a minimal USD OIS curveset and quotes
    if static is None or quotes is None:
        # Single curve for USD SOFR OIS (discount = forecast = SOFR)
        sofr = CurveId("USD_SOFR")

        # Curve defs (single param curve)
        curve_defs = (CurveDef(curve_id=sofr, kind="param"),)

        # Curve configs: small regularisation for slope/curv; no level penalty
        curve_configs = {
            sofr: CurveConfig(
                curve_id=sofr,
                interp="stepwise_const_fwd",
                lam_slope=1e-4,
                lam_curv=1e-6,
                lam_level=0.0,
            ),
        }

        # Refdata mapping (USD SOFR OIS -> (disc, fcast) both SOFR)
        refdata = ReferenceDataContainer({"USD_SOFR_OIS": (sofr, sofr)})

        # Spec
        spec = CurveSetSpec(curve_defs=list(curve_defs), curve_configs=curve_configs)
        curve_id_to_idx, graph = build_curve_graph_arrays(spec.curve_defs)

        # Knot grid
        knot_times = np.linspace(0.0, float(maxT), int(n_mats) + 1).astype(np.float64)
        knot_times[0] = 0.0

        # Quotes: mix of months, years, and forward-start swaps.
        # Use today's SOFR OIS market par for spot maturities; disable forwards by weight.
        qts: list[Quote] = []
        spot_months = [0.25, 0.5, 0.75]
        spot_years = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
        forwards: list[tuple[float, float]] = [
            (1.0, 1.0),
            (1.0, 3.0),
            (1.0, 5.0),
            (2.0, 2.0),
            (2.0, 5.0),
        ]
        month_par = {3: 0.03667, 6: 0.03624, 9: 0.03558}
        year_par = {1: 0.03498, 2: 0.03404, 3: 0.03432, 5: 0.03553, 7: 0.03689, 10: 0.03867}
        for m in spot_months:
            inst = SwapSpec(
                instrument_id=InstrumentId(f"USD_SOFR_OIS_Spot_{round(m * 12)}M"),
                currency="USD",
                index="USD_SOFR_OIS",
                maturity=float(m),
                forward_start_years=0.0,
                fixed_leg_freq="A",
                float_leg_freq="S",
            )
            val = month_par.get(round(m * 12), 0.035)
            qts.append(Quote(inst, "par_rate", float(val), 1.0))
        for y in spot_years:
            inst = SwapSpec(
                instrument_id=InstrumentId(f"USD_SOFR_OIS_Spot_{round(y)}Y"),
                currency="USD",
                index="USD_SOFR_OIS",
                maturity=float(y),
                forward_start_years=0.0,
                fixed_leg_freq="A",
                float_leg_freq="S",
            )
            val = year_par.get(round(y), 0.035)
            qts.append(Quote(inst, "par_rate", float(val), 1.0))
        for fs, ten in forwards:
            inst = SwapSpec(
                instrument_id=InstrumentId(f"USD_SOFR_OIS_Fwd{round(fs)}Yx{round(ten)}Y"),
                currency="USD",
                index="USD_SOFR_OIS",
                maturity=float(ten),
                forward_start_years=float(fs),
                fixed_leg_freq="A",
                float_leg_freq="S",
            )
            qts.append(Quote(inst, "par_rate", 0.0, 0.0))

        # Pack
        ctx = PackContext()
        shape_policy = ShapePolicy()
        batch = pack_swaps_pooled(qts, ctx, refdata, curve_id_to_idx, shape_policy)
        static = make_calib_static(knot_times, graph, batch, spec)
        # Use provided market par rates directly
        quotes = qts

    # Initial guess if not provided
    P = len(static.param_curve_ids)
    N = int(static.knot_times.size - 1)
    if x0_np is None:
        x0_np = np.zeros((P * N,), dtype=np.float64) + 0.02

    # Coupons from quotes (par) and baseline market par (same for par instruments)
    coupon_np = np.array([q.value for q in quotes], dtype=np.float64)
    market_par_base_np = coupon_np.copy()
    # Pad to bucketed S
    coupon_bucket = _bucketize_vector(coupon_np, static.batch)
    market_par_base_bucket = _bucketize_vector(market_par_base_np, static.batch)

    return demo_streaming_risk_from_static(
        static=static,
        x0_np=x0_np,
        market_par_base_np=np.asarray(market_par_base_bucket, dtype=np.float64),
        coupon_per_swap_np=np.asarray(coupon_bucket, dtype=np.float64),
        iterations=iterations,
        noise_bps=noise_bps,
        max_steps=max_steps,
        jac_mode=jac_mode,
        bucket_edges=bucket_edges,
        display_mode="par",
        sleep_seconds=sleep_seconds,
    )


def risk_snapshot(
    static: CalibStatic | None = None,
    quotes: Sequence[Quote] | None = None,
    x_np: FloatNDArray | None = None,
    bucket_edges: Sequence[float] = (0.0, 1.0, 3.0, 5.0, 10.0, 30.0),
    return_jacobian: bool = False,
    snapshot_mode: str = "pv",
    n_mats: int = 20,
    maxT: float = 10.0,
):
    """
    Build a PV + bucketed DV01 snapshot as a pandas DataFrame.

    When `static` or `quotes` are omitted, constructs a minimal USD curveset with two
    param curves (discount/forecast) and `n_mats` USD SOFR OIS quotes over [0, maxT].

    Parameters:
      - static: prebuilt calibration payload; if None, a default is constructed
      - quotes: list of `Quote`; if None, defaults are constructed
      - x_np: parameter vector (concat of param blocks); if None, defaults to 0.02
      - bucket_edges: term bucket edges in years
      - return_jacobian: when True, also returns the full per-knot Jacobian (S, P*N)
      - n_mats, maxT: used only when building defaults

        Returns:
            - When snapshot_mode == "pv":
                    df: DataFrame [instrument_id, PV, DV01_<bucket>...]
                    J (optional): numpy array (S, P*N) of per-knot PV deltas
                When snapshot_mode == "par":
                    df: DataFrame [instrument_id, Par]
    """

    # Default build if needed (same pattern as `main_risk`)
    if static is None or quotes is None:
        sofr = CurveId("USD_SOFR")

        curve_defs = (CurveDef(curve_id=sofr, kind="param"),)
        curve_configs = {
            sofr: CurveConfig(
                curve_id=sofr,
                interp="stepwise_const_fwd",
                lam_slope=1e-4,
                lam_curv=1e-6,
                lam_level=0.0,
            ),
        }
        refdata = ReferenceDataContainer({"USD_SOFR_OIS": (sofr, sofr)})
        spec = CurveSetSpec(curve_defs=curve_defs, curve_configs=curve_configs)
        curve_id_to_idx, graph = build_curve_graph_arrays(spec.curve_defs)

        knot_times = np.linspace(0.0, float(maxT), int(n_mats) + 1).astype(np.float64)
        knot_times[0] = 0.0

        # Default quotes: same curated set as main_risk
        qts: list[Quote] = []
        spot_months = [0.25, 0.5, 0.75]
        spot_years = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
        forwards: list[tuple[float, float]] = [
            (1.0, 1.0),
            (1.0, 3.0),
            (1.0, 5.0),
            (2.0, 2.0),
            (2.0, 5.0),
        ]
        month_par = {3: 0.03667, 6: 0.03624, 9: 0.03558}
        year_par = {1: 0.03498, 2: 0.03404, 3: 0.03432, 5: 0.03553, 7: 0.03689, 10: 0.03867}
        for m in spot_months:
            inst = SwapSpec(
                instrument_id=InstrumentId(f"USD_SOFR_OIS_Spot_{round(m * 12)}M"),
                currency="USD",
                index="USD_SOFR_OIS",
                maturity=float(m),
                forward_start_years=0.0,
                fixed_leg_freq="A",
                float_leg_freq="S",
            )
            val = month_par.get(round(m * 12), 0.035)
            qts.append(Quote(inst, "par_rate", float(val), 1.0))
        for y in spot_years:
            inst = SwapSpec(
                instrument_id=InstrumentId(f"USD_SOFR_OIS_Spot_{round(y)}Y"),
                currency="USD",
                index="USD_SOFR_OIS",
                maturity=float(y),
                forward_start_years=0.0,
                fixed_leg_freq="A",
                float_leg_freq="S",
            )
            val = year_par.get(round(y), 0.035)
            qts.append(Quote(inst, "par_rate", float(val), 1.0))
        for fs, ten in forwards:
            inst = SwapSpec(
                instrument_id=InstrumentId(f"USD_SOFR_OIS_Fwd{round(fs)}Yx{round(ten)}Y"),
                currency="USD",
                index="USD_SOFR_OIS",
                maturity=float(ten),
                forward_start_years=float(fs),
                fixed_leg_freq="A",
                float_leg_freq="S",
            )
            qts.append(Quote(inst, "par_rate", 0.0, 0.0))

        ctx = PackContext()
        shape_policy = ShapePolicy()
        batch = pack_swaps_pooled(qts, ctx, refdata, curve_id_to_idx, shape_policy)
        static = make_calib_static(knot_times, graph, batch, spec)
        quotes = qts

    # Params default
    P = len(static.param_curve_ids)
    N = int(static.knot_times.size - 1)
    if x_np is None:
        x_np = np.zeros((P * N,), dtype=np.float64) + 0.02

    if snapshot_mode == "par":
        # Build Par + DV01 snapshot (coupon from quotes)
        coupon_np = np.array([q.value for q in quotes], dtype=np.float64)
        coupon_bucket = _bucketize_vector(coupon_np, static.batch)
        _, dv01_bucket_fn = build_risk_engines(static, bucket_edges)
        bucket_labels = [
            f"{bucket_edges[i]}-{bucket_edges[i + 1]}Y" for i in range(len(bucket_edges) - 1)
        ]
        df = portfolio_par_and_risk_df(
            params_concat=jnp.array(x_np),
            static=static,
            coupon_per_swap_np=coupon_bucket,
            dv01_bucket_fn=dv01_bucket_fn,
            bucket_labels=bucket_labels,
        )
        return df
    else:
        coupon_np = np.array([q.value for q in quotes], dtype=np.float64)
        # Pad coupons to bucketed S for PV/risk engines
        coupon_bucket = _bucketize_vector(coupon_np, static.batch)
        jac_pv_fn, dv01_bucket_fn = build_risk_engines(static, bucket_edges)
        bucket_labels = [
            f"{bucket_edges[i]}-{bucket_edges[i + 1]}Y" for i in range(len(bucket_edges) - 1)
        ]

        df, J = portfolio_pv_and_risk_df(
            params_concat=jnp.array(x_np),
            static=static,
            coupon_per_swap_np=coupon_bucket,
            jac_pv_fn=jac_pv_fn,
            dv01_bucket_fn=dv01_bucket_fn,
            bucket_labels=bucket_labels,
        )

        if return_jacobian:
            return df, J
        return df


if __name__ == "__main__":
    main_risk(iterations=10, noise_bps=0.2, max_steps=50, jac_mode="fwd")

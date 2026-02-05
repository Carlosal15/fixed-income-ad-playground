from fixed_income_ad_playground.types import FloatNDArray
import numpy as np
from fixed_income_ad_playground.packing.swaps import PackedSwapQuoteBatch
from fixed_income_ad_playground.kernels.swaps import par_swap_rates_from_packed
import jax.numpy as jnp
import matplotlib.pyplot as plt
from fixed_income_ad_playground.kernels.discount_factors import dfs_from_stepwise_const_forwards
from fixed_income_ad_playground.identifiers.identifiers import CurveId, InstrumentId
from fixed_income_ad_playground.instruments.instrument import PackContext
from fixed_income_ad_playground.pricing.swap_pricer import SwapPricer
from fixed_income_ad_playground.pricing.quote import Quote
from fixed_income_ad_playground.instruments.swap import SwapSpec
from fixed_income_ad_playground.curve.curve_config import CurveSetConfig, CurveSpec
from fixed_income_ad_playground.calibration.scipy_wrappers import calibrate_curve_one_date
from fixed_income_ad_playground.curve.curve import FwdCurve
from fixed_income_ad_playground.packing.futures import (
    PackedFuturesQuoteBatch,
    pack_futures_to_batch,
)
from fixed_income_ad_playground.packing.combined import PackedMarketBatches
from fixed_income_ad_playground.packing.swaps import pack_swaps_to_batch
import time
from fixed_income_ad_playground.market.market import Market


def yearfrac_from_days(days: float) -> float:
    return float(days) / 365.0  # ACT/365F toy


def build_standard_swap_maturities_years(spot_lag_days: int = 2) -> FloatNDArray:
    """
    Weekly 1W..4W, monthly 1M..12M (30d months), yearly 2Y..10Y.
    Shifted by spot lag (toy: +spot/365).
    """
    spot = yearfrac_from_days(spot_lag_days)
    weeks = np.array([1, 2, 3, 4], dtype=np.float64) * (7.0 / 365.0)
    months = np.arange(1, 13, dtype=np.float64) * (30.0 / 365.0)
    years = np.arange(2, 11, dtype=np.float64)
    return (np.concatenate([weeks, months, years], axis=0) + spot).astype(np.float64)


def make_true_forward_on_knots(
    knot_times: FloatNDArray, level: float, slope: float, wiggle: float
) -> FloatNDArray:
    interval_midpoints = 0.5 * (knot_times[:-1] + knot_times[1:])
    forward = (
        level
        + slope * (1.0 - np.exp(-interval_midpoints))
        + wiggle * np.sin(2.0 * np.pi * interval_midpoints / 5.0)
    )
    return forward.astype(np.float64)


def make_market_par_from_true(
    true_forward_params: FloatNDArray,
    knot_times: FloatNDArray,
    swap_batch: PackedSwapQuoteBatch,
    noise_bps: float,
    seed: int,
) -> FloatNDArray:
    knot_jax = jnp.array(knot_times)
    f_jax = jnp.array(true_forward_params)
    par = par_swap_rates_from_packed(
        f_jax,
        knot_jax,
        swap_batch.payment_times,
        swap_batch.accrual_factors,
        swap_batch.cashflow_mask,
        swap_batch.maturity_times,
    )
    par_np = np.asarray(par, dtype=np.float64)
    if noise_bps > 0.0:
        rng = np.random.default_rng(seed)
        par_np = par_np + (noise_bps * 1e-4) * rng.normal(size=par_np.shape)
    return par_np


def plot_forward_and_zero(
    knot_times: FloatNDArray, curves_forward: list[FloatNDArray], labels: list[str]
) -> None:
    knot_times = np.asarray(knot_times, dtype=np.float64)
    left = knot_times[:-1]
    right = knot_times[1:]
    step_times = np.column_stack([left, right]).reshape(-1)

    plt.figure(figsize=(10, 4))
    for forward_params, label in zip(curves_forward, labels, strict=True):
        step_values = np.column_stack([forward_params, forward_params]).reshape(-1)
        plt.plot(step_times, step_values, label=label)
    plt.xlabel("Maturity (years)")
    plt.ylabel("Instantaneous forward f(t)")
    plt.title("Forward Curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    knot_jax = jnp.array(knot_times)
    t_grid = jnp.linspace(0.01, float(knot_times[-1]), 800)
    for forward_params, label in zip(curves_forward, labels, strict=True):
        f_jax = jnp.array(forward_params)
        df = dfs_from_stepwise_const_forwards(knot_jax, f_jax, t_grid)
        zero = -jnp.log(df) / t_grid
        plt.plot(np.asarray(t_grid), np.asarray(zero), label=label)
    plt.xlabel("Maturity (years)")
    plt.ylabel("Zero rate z(t)")
    plt.title("Implied Zero Curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    ctx = PackContext()
    curve_id = CurveId("USD_SOFR_OIS_TOY")

    # ------------------------------------------------------------
    # Build swap instrument set (same across dates)
    # ------------------------------------------------------------
    spot_lag_days = 2
    maturities = np.sort(build_standard_swap_maturities_years(spot_lag_days=spot_lag_days))

    swap_quotes: list[Quote] = []
    for maturity_years in maturities:
        inst = SwapSpec(
            instrument_id=InstrumentId(f"SWAP_{maturity_years:.6f}Y"),
            maturity=float(maturity_years),
            discount_curve=curve_id,
        )
        swap_quotes.append(Quote(instrument=inst, quote_type="par_rate", value=0.0, weight=1.0))

    # ------------------------------------------------------------
    # Futures block: STUB (empty by default)
    # ------------------------------------------------------------
    futures_quotes: list[Quote] = []
    # Example of how you'd add a stub future later:
    # fut_inst = FuturesSpec(InstrumentId("FUT_1"), expiry_years=0.25, discount_curve=curve_id)
    # futures_quotes.append(Quote(instrument=fut_inst, qtype="par_rate", value=0.0, weight=1.0))

    # ------------------------------------------------------------
    # Knots: one per swap maturity + 0
    # ------------------------------------------------------------
    knot_times = np.concatenate([[0.0], maturities], axis=0).astype(np.float64)
    spec = CurveSpec(curve_id=curve_id, knot_times=knot_times)

    # Pack swaps + futures into fixed-shape arrays
    swap_batch = pack_swaps_to_batch(swap_quotes, ctx)

    # For empty futures block, create batch with shape (0,) and the same curve_id
    futures_batch = pack_futures_to_batch(futures_quotes, ctx)
    if futures_batch.expiry_times.shape[0] == 0:
        futures_batch = PackedFuturesQuoteBatch(
            curve_id=curve_id,
            instrument_ids=[],
            expiry_times=jnp.zeros((0,), dtype=jnp.float64),
            weights=jnp.zeros((0,), dtype=jnp.float64),
        )

    batches = PackedMarketBatches(swaps=swap_batch, futures=futures_batch)

    print("swap_count:", int(swap_batch.maturity_times.shape[0]))
    print("max fixed payments per swap:", int(swap_batch.payment_times.shape[1]))
    print("curve intervals:", int(knot_times.size - 1))
    print("spot_lag_days:", spot_lag_days)
    print("futures_count (stub):", int(futures_batch.expiry_times.shape[0]))

    # Config (penalties optional)
    cfg = CurveSetConfig(curve_specs=[spec], quotes=swap_quotes, lam_slope=1e-4, lam_curv=1e-6)

    # True curves + synthetic market for 2 dates
    true_forward_1 = make_true_forward_on_knots(knot_times, level=0.040, slope=0.010, wiggle=0.002)
    true_forward_2 = make_true_forward_on_knots(knot_times, level=0.030, slope=0.010, wiggle=0.003)

    noise_bps = 0.1
    market_par_swaps_1 = make_market_par_from_true(
        true_forward_1, knot_times, swap_batch, noise_bps=noise_bps, seed=4
    )
    market_par_swaps_2 = make_market_par_from_true(
        true_forward_2, knot_times, swap_batch, noise_bps=noise_bps, seed=5
    )

    # Futures market quotes (stub, empty)
    market_quotes_futures_1 = np.zeros(
        (int(futures_batch.expiry_times.shape[0]),), dtype=np.float64
    )
    market_quotes_futures_2 = np.zeros(
        (int(futures_batch.expiry_times.shape[0]),), dtype=np.float64
    )

    # Initial guess
    interval_count = knot_times.size - 1
    x0 = 0.05 * np.ones(interval_count, dtype=np.float64)

    # Calibrate two dates (should reuse compiled code; Date2 warmup should be tiny)
    t_all = time.perf_counter()
    params_1, _ = calibrate_curve_one_date(
        "Date 1",
        spec,
        cfg,
        batches,
        market_par_swaps_1,
        market_quotes_futures_1,
        x0,
        do_warmup=True,
        max_nfev=80,
    )
    params_2, _ = calibrate_curve_one_date(
        "Date 2",
        spec,
        cfg,
        batches,
        market_par_swaps_2,
        market_quotes_futures_2,
        x0,
        do_warmup=True,
        max_nfev=80,
    )
    print("\nTotal wall time:", time.perf_counter() - t_all)

    # Build a frozen Market for Date 1 and price something
    curve_date1 = FwdCurve(
        curve_id=curve_id,
        knot_times=jnp.array(knot_times),
        forward_params_per_interval=jnp.array(params_1.forward_params_per_interval),
    )
    market_date1 = Market(curves={curve_id: curve_date1})

    pricer = SwapPricer()

    # Pricing example: pick maturity closest to 5Y (+ spot shift already in the grid)
    target = 5.0 + yearfrac_from_days(spot_lag_days)
    picked_maturity = float(maturities[np.argmin(np.abs(maturities - target))])

    example_swap = SwapSpec(
        instrument_id=InstrumentId("EXAMPLE_SWAP"),
        maturity=picked_maturity,
        discount_curve=curve_id,
    )

    par_rate = pricer.par_rate(market_date1, example_swap, ctx)
    pv_at_par = pricer.pv(market_date1, example_swap, fixed_rate=par_rate, ctx=ctx)
    pv_bumped = pricer.pv(market_date1, example_swap, fixed_rate=par_rate + 10e-4, ctx=ctx)  # +10bp

    print("\n--- Pricing example (Date 1 calibrated market) ---")
    print("Example maturity (years):", picked_maturity)
    print("Par rate:", par_rate)
    print("PV at par (should be ~0):", pv_at_par)
    print("PV at par+10bp:", pv_bumped)

    # Plot (true vs calibrated)
    plot_forward_and_zero(
        knot_times,
        curves_forward=[
            true_forward_1,
            params_1.forward_params_per_interval,
            true_forward_2,
            params_2.forward_params_per_interval,
        ],
        labels=["TRUE date1", "CAL date1", "TRUE date2", "CAL date2"],
    )


if __name__ == "__main__":
    main()

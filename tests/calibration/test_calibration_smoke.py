import numpy as np
import jax.numpy as jnp
from fixed_income_ad_playground.calibration.calibration import (
    CurveDef,
    CurveSetSpec,
    build_curve_graph_arrays,
    make_calib_static,
    CurveSetCalibrator,
)
from fixed_income_ad_playground.identifiers.identifiers import CurveId, InstrumentId
from fixed_income_ad_playground.curve.curve_config import CurveConfig
from fixed_income_ad_playground.packing.swaps import pack_swaps_pooled
from fixed_income_ad_playground.shape_policy import ShapePolicy
from fixed_income_ad_playground.instruments.swap import SwapSpec
from fixed_income_ad_playground.instruments.instrument import PackContext
from fixed_income_ad_playground.pricing.quote import Quote
from fixed_income_ad_playground.reference_data_container import ReferenceDataContainer


def test_calibration_pipeline_smoke():
    # Build a simple curveset: two param curves
    disc = CurveId("USD_DISC")
    fcast = CurveId("USD_FCAST")
    curve_defs = [CurveDef(disc, "param"), CurveDef(fcast, "param")]
    spec = CurveSetSpec(
        curve_defs=curve_defs,
        curve_configs={
            disc: CurveConfig(
                curve_id=disc, interp="stepwise_const_fwd", lam_slope=1e-4, lam_curv=1e-6
            ),
            fcast: CurveConfig(
                curve_id=fcast, interp="stepwise_const_fwd", lam_slope=1e-4, lam_curv=1e-6
            ),
        },
    )

    curve_id_to_idx, graph = build_curve_graph_arrays(curve_defs)

    # Quotes: small set
    ref = ReferenceDataContainer({"USD_SOFR_OIS": (disc, fcast)})
    swaps = [
        SwapSpec(
            instrument_id=InstrumentId("S1"), currency="USD", index="USD_SOFR_OIS", maturity=1.0
        ),
        SwapSpec(
            instrument_id=InstrumentId("S2"), currency="USD", index="USD_SOFR_OIS", maturity=2.0
        ),
    ]
    quotes = [Quote(instrument=s, quote_type="par_rate", value=0.0) for s in swaps]

    batch = pack_swaps_pooled(quotes, PackContext(), ref, curve_id_to_idx, ShapePolicy())

    # Knot grid
    knot = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    static = make_calib_static(knot, graph, batch, spec)
    calib = CurveSetCalibrator(static)

    # Warmup jitted functions
    N = static.knot_times.size - 1
    P = static.graph.num_param
    x0 = jnp.full((P * N,), 0.02)
    mp0 = jnp.zeros((batch.swaps_bucket_size,))
    warm = calib.warmup_residuals_jacobian(x0, mp0)
    assert warm >= 0.0

    # Build market from x0
    mkt = calib.build_market(x0)
    assert disc in mkt.curves and fcast in mkt.curves

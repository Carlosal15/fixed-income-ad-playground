from fixed_income_ad_playground.packing.swaps import pack_swaps_pooled
from fixed_income_ad_playground.shape_policy import ShapePolicy
from fixed_income_ad_playground.instruments.swap import SwapSpec
from fixed_income_ad_playground.instruments.instrument import PackContext
from fixed_income_ad_playground.pricing.quote import Quote
from fixed_income_ad_playground.identifiers.identifiers import CurveId, InstrumentId
from fixed_income_ad_playground.reference_data_container import ReferenceDataContainer


def test_pack_swaps_pooled_shapes_and_indices():
    ref = ReferenceDataContainer({"USD_SOFR_OIS": (CurveId("USD_DISC"), CurveId("USD_FCAST"))})
    # Two swaps with different maturities
    swaps = [
        SwapSpec(
            InstrumentId("S1"),
            "USD",
            "USD_SOFR_OIS",
            maturity=1.0,
            fixed_leg_freq="A",
            float_leg_freq="Q",
        ),
        SwapSpec(
            InstrumentId("S2"),
            "USD",
            "USD_SOFR_OIS",
            maturity=2.0,
            fixed_leg_freq="A",
            float_leg_freq="S",
        ),
    ]
    quotes = [Quote(instrument=s, quote_type="par_rate", value=0.0, weight=1.0) for s in swaps]

    curve_id_to_idx = {CurveId("USD_DISC"): 0, CurveId("USD_FCAST"): 1}
    batch = pack_swaps_pooled(quotes, PackContext(), ref, curve_id_to_idx, ShapePolicy())

    # True counts
    assert batch.swap_count_true == 2
    assert batch.max_cf_true >= 2
    assert batch.Q_true >= 3

    # Bucket sizes are powers of two or predefined buckets
    assert batch.swaps_bucket_size >= batch.swap_count_true
    assert batch.cashflows_bucket_size >= batch.max_cf_true
    assert batch.unique_times_bucket_size >= batch.Q_true

    # Indices refer into pooled times correctly
    for i in range(batch.swap_count_true):
        for j in range(batch.max_cf_true):
            if batch.cashflow_mask[i, j] == 1.0:
                s_idx = int(batch.idx_start[i, j])
                e_idx = int(batch.idx_end[i, j])
                assert 0 <= s_idx < batch.t_pool.size
                assert 0 <= e_idx < batch.t_pool.size

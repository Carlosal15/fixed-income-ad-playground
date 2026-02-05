import numpy as np
from fixed_income_ad_playground.instruments.swap import SwapSpec
from fixed_income_ad_playground.instruments.instrument import PackContext
from fixed_income_ad_playground.identifiers.identifiers import InstrumentId, CurveId
from fixed_income_ad_playground.reference_data_container import ReferenceDataContainer


def test_swap_pack_generates_schedule_and_curves():
    ref = ReferenceDataContainer({"USD_SOFR_OIS": (CurveId("USD_DISC"), CurveId("USD_FCAST"))})
    s = SwapSpec(
        instrument_id=InstrumentId("SWAP1"),
        currency="USD",
        index="USD_SOFR_OIS",
        maturity=3.25,
        fixed_leg_freq="A",
        float_leg_freq="S",
    )
    p = s.pack(PackContext(), ref)
    assert p.discount_curve.name == "USD_DISC"
    assert p.forecast_curve.name == "USD_FCAST"
    assert p.end_times[-1] == np.float64(3.25)
    # Semiannual schedule → ~0.5 steps
    assert p.end_times.size >= 6

from __future__ import annotations
from dataclasses import dataclass

from fixed_income_ad_playground.identifiers.identifiers import InstrumentId, CurveId
from fixed_income_ad_playground.instruments.instrument import (
    Instrument,
    PackContext,
    PackedInstrument,
)
from fixed_income_ad_playground.enums import Currency, SwapIndex, PaymentFrequency
from fixed_income_ad_playground.types import FloatNDArray
from fixed_income_ad_playground.reference_data_container import ReferenceDataContainer
import numpy as np

_freq_to_step = {"A": 1.0, "S": 0.5, "Q": 0.25}


@dataclass(frozen=True)
class SwapSpec(Instrument):
    """
    Minimal vanilla swap spec:
    - maturity provided directly in year fractions
    - fixed leg schedule rule (toy):
        * if maturity <= 1Y -> bullet payment at maturity
        * else -> annual payments
    """

    instrument_id: InstrumentId
    currency: Currency
    index: SwapIndex
    maturity: float  # years (tenor length)
    notional: float = 1_000_000.0
    forward_start_years: float = 0.0  # forward start
    fixed_leg_freq: PaymentFrequency = PaymentFrequency.A
    float_leg_freq: PaymentFrequency = PaymentFrequency.S

    def pack(self, ctx: PackContext, reference_data: ReferenceDataContainer) -> PackedSwap:
        tenor = float(self.maturity)
        fwd_start = float(self.forward_start_years)
        disc_curve, fcast_curve = reference_data.swap_curves(self.index)

        # Demo schedule: by leg freq (keep it simple)
        freq_to_step = {"A": 1.0, "S": 0.5, "Q": 0.25}
        step = float(freq_to_step[self.float_leg_freq])

        n_pay = int(np.ceil(tenor / step))
        # Forward-start schedule: shift by F, ensure last payment hits F+T
        end_times = fwd_start + np.arange(1, n_pay + 1, dtype=np.float64) * step
        end_times[-1] = fwd_start + tenor
        start_times = np.concatenate([[fwd_start], end_times[:-1]], axis=0)
        accrual = (end_times - start_times).astype(np.float64)

        return PackedSwap(
            instrument_id=self.instrument_id,
            discount_curve=disc_curve,
            forecast_curve=fcast_curve,
            start_times=start_times,
            end_times=end_times,
            accrual_factors=accrual,
            maturity_years=tenor,
            notional=self.notional,
        )


@dataclass(frozen=True)
class PackedSwap(PackedInstrument):
    instrument_id: InstrumentId
    discount_curve: CurveId
    forecast_curve: CurveId
    start_times: FloatNDArray
    end_times: FloatNDArray
    accrual_factors: FloatNDArray
    maturity_years: float
    notional: float

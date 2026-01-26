from __future__ import annotations
from dataclasses import dataclass

from fixed_income_ad_playground.identifiers.identifiers import InstrumentId, CurveId
from fixed_income_ad_playground.instruments.instrument import (
    Instrument,
    PackContext,
    PackedInstrument,
)
from fixed_income_ad_playground.enums import DayCount
from fixed_income_ad_playground.types import FloatNDArray
import numpy as np


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
    maturity: float
    discount_curve: CurveId
    daycount: DayCount = "ACT/365"  # TODO: mock value- no datetime handling implemented

    def pack(self, ctx: PackContext) -> PackedSwap:
        # Packing here produces a *ragged* schedule; batching/padding happens later.
        maturity_years = float(self.maturity)
        if maturity_years <= 1.0:
            payment_times = np.array([maturity_years], dtype=np.float64)
            accrual_factors = np.array(
                [maturity_years], dtype=np.float64
            )  # toy: accrual = maturity
        else:
            payment_count = int(np.ceil(maturity_years))
            payment_times = np.arange(1, payment_count + 1, dtype=np.float64)
            payment_times[-1] = maturity_years
            accrual_factors = np.ones_like(payment_times, dtype=np.float64)

        return PackedSwap(
            instrument_id=self.instrument_id,
            discount_curve=self.discount_curve,
            payment_times=payment_times,
            accrual_factors=accrual_factors,
            maturity_years=maturity_years,
        )


@dataclass(frozen=True)
class PackedSwap(PackedInstrument):
    instrument_id: InstrumentId
    discount_curve: CurveId
    payment_times: FloatNDArray  # (k,) ragged
    accrual_factors: FloatNDArray  # (k,) ragged
    maturity_years: float

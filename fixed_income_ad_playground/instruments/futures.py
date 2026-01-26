from __future__ import annotations
from dataclasses import dataclass
from fixed_income_ad_playground.identifiers.identifiers import InstrumentId, CurveId
from fixed_income_ad_playground.instruments.instrument import (
    Instrument,
    PackContext,
    PackedInstrument,
)


@dataclass(frozen=True)
class FuturesSpec(Instrument):
    """
    STUB for Futures instrument.

    Futures aren't implemented currently, but it was important to have a placeholder
    in order to test a design with multiple instrument types for jitted calibration.

    v1 stub:
        - expiry_years: year-fraction to expiry
        - discount_curve: which curve affects its pricing
    """

    instrument_id: InstrumentId
    expiry_years: float
    discount_curve: CurveId

    def pack(self, ctx: PackContext) -> PackedFuture:
        return PackedFuture(
            instrument_id=self.instrument_id,
            discount_curve=self.discount_curve,
            expiry_years=float(self.expiry_years),
        )


@dataclass(frozen=True)
class PackedFuture(PackedInstrument):
    instrument_id: InstrumentId
    discount_curve: CurveId
    expiry_years: float

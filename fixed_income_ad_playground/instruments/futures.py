from __future__ import annotations
from dataclasses import dataclass
from fixed_income_ad_playground.identifiers.identifiers import InstrumentId
from fixed_income_ad_playground.instruments.instrument import (
    Instrument,
    PackContext,
    PackedInstrument,
)
from fixed_income_ad_playground.enums import Currency, FuturesIndex
from fixed_income_ad_playground.reference_data_container import ReferenceDataContainer


@dataclass(frozen=True)
class StIRFutureSpec(Instrument):
    """
    Stub instrument for STIR futures (SOFR/ESTR/etc).
    Not implemented; wiring only.
    """

    instrument_id: InstrumentId
    currency: Currency
    index: FuturesIndex
    expiry: float  # years

    def pack(self, ctx: PackContext, reference_data: ReferenceDataContainer) -> PackedStIRFuture:
        raise NotImplementedError("STIR futures packing is a stub in this POC.")


@dataclass(frozen=True)
class PackedStIRFuture(PackedInstrument):
    instrument_id: InstrumentId
    # etc...
    pass

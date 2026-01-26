from __future__ import annotations
from dataclasses import dataclass
from fixed_income_ad_playground.identifiers.identifiers import InstrumentId, CurveId
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PackContext:
    """
    Acting as a context for packing instruments. Can contain market data references,
    configuration flags, etc. Empty for now...
    """

    pass


@runtime_checkable
class Instrument(Protocol):
    """Lightweight instrument protocol

    Instruments currently only hold their own name, and particularly important: implement a pack
    method for calibration.
    """

    instrument_id: InstrumentId

    def pack(self, ctx: PackContext) -> PackedInstrument: ...


@runtime_checkable
class PackedInstrument(Protocol):
    # assume same forecast and discount curve
    instrument_id: InstrumentId
    discount_curve: CurveId

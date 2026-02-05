from __future__ import annotations
from dataclasses import dataclass
from fixed_income_ad_playground.identifiers.identifiers import InstrumentId
from typing import Protocol, runtime_checkable
from fixed_income_ad_playground.reference_data_container import ReferenceDataContainer


@dataclass(frozen=True)
class PackContext:
    """
    Acting as a context for packing instruments.
    Real case would include things like      ]
    - valuation date
    - calendars, conventions
    - fixings
    - daycounts
    - etc
    """

    pass


@runtime_checkable
class Instrument(Protocol):
    """Lightweight instrument protocol

    Instruments currently only hold their own name, and particularly important: implement a pack
    method for calibration.
    """

    instrument_id: InstrumentId

    def pack(
        self, ctx: PackContext, reference_data: ReferenceDataContainer
    ) -> PackedInstrument: ...


@runtime_checkable
class PackedInstrument(Protocol):
    instrument_id: InstrumentId

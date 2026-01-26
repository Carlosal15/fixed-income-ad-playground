from dataclasses import dataclass


# simple str-like identifiers for now. Would require more structures and types


@dataclass(frozen=True)
class CurveId:
    name: str


@dataclass(frozen=True)
class InstrumentId:
    name: str

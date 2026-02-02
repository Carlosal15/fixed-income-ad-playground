from dataclasses import FrozenInstanceError
import pytest
from fixed_income_ad_playground.identifiers.identifiers import CurveId, InstrumentId


def test_curve_id_equality_and_hash():
    a = CurveId("USD_DISC")
    b = CurveId("USD_DISC")
    c = CurveId("EUR_DISC")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_instrument_id_immutable():
    ins = InstrumentId("SWAP1")
    with pytest.raises(FrozenInstanceError):
        # dataclass frozen: assignment not allowed
        ins.name = "SWAP2"  # type: ignore

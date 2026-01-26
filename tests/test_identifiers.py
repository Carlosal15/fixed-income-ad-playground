from fixed_income_ad_playground.identifiers.identifiers import CurveId, InstrumentId


def test_identifier_dataclasses():
    c = CurveId(name="c1")
    i = InstrumentId(name="s1")
    assert c.name == "c1"
    assert i.name == "s1"
    # dataclass equality
    assert CurveId(name="c1") == c

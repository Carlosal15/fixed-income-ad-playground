from fixed_income_ad_playground.enums import (
    DayCount,
    QuoteType,
    Currency,
    SwapIndex,
    FuturesIndex,
    InterpType,
    PaymentFrequency,
    CurveDefKind,
)


def test_enums_values_exist():
    # Basic membership checks using typing.Literal-compatible values
    assert "ACT/365" in DayCount.__args__
    assert "par_rate" in QuoteType.__args__
    assert "USD" in Currency.__args__
    assert "USD_SOFR_OIS" in SwapIndex.__args__
    assert "USD_SOFR_FUT" in FuturesIndex.__args__
    assert "stepwise_const_fwd" in InterpType.__args__
    assert "A" in PaymentFrequency.__args__
    assert "param" in CurveDefKind.__args__

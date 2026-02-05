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
    assert QuoteType.PAR_RATE
    assert Currency.USD
    assert SwapIndex.USD_SOFR_OIS
    assert FuturesIndex.USD_SOFR_FUT
    assert InterpType.STEPWISE_CONST_FWD
    assert PaymentFrequency.A
    assert CurveDefKind.PARAM

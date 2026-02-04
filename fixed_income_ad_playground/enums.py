from typing import Literal
from enum import StrEnum

# Acting as "enums"/stubs for supported enumerations

DayCount = Literal["ACT/360", "ACT/365", "30/360"]
# QuoteType = Literal["par_rate"]


class QuoteType(StrEnum):
    PAR_RATE = "par_rate"
    PRICE = "price"
    SPREAD = "spread"


class Currency(StrEnum):
    USD = "USD"
    EUR = "EUR"


class SwapIndex(StrEnum):
    USD_SOFR_OIS = "USD_SOFR_OIS"
    USD_FF = "USD_FF"
    EUR_ESTR_OIS = "EUR_ESTR_OIS"
    EUR_EURIBOR3M = "EUR_EURIBOR3M"
    EUR_EURIBOR6M = "EUR_EURIBOR6M"


class FuturesIndex(StrEnum):
    USD_SOFR_FUT = "USD_SOFR_FUT"
    EUR_ESTR_FUT = "EUR_ESTR_FUT"


class InterpType(StrEnum):
    STEPWISE_CONST_FWD = "stepwise_const_fwd"
    MIXED_CONST_QUADRATIC = "mixed_const_quadratic"
    QUADRATIC = "quadratic"


class PaymentFrequency(StrEnum):
    A = "A"
    S = "S"
    Q = "Q"


class CurveDefKind(StrEnum):
    PARAM = "param"
    LINEAR_COMB = "linear_comb"

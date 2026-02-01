from typing import Literal

# Acting as "enums"/stubs for supported enumerations. Not used for demo purposes.

DayCount = Literal["ACT/360", "ACT/365", "30/360"]
QuoteType = Literal["par_rate"]

Currency = Literal["USD", "EUR"]

SwapIndex = Literal[
    "USD_SOFR_OIS",
    "USD_FF",
    "EUR_ESTR_OIS",
    "EUR_EURIBOR3M",
    "EUR_EURIBOR6M",
]

FuturesIndex = Literal[
    "USD_SOFR_FUT",
    "EUR_ESTR_FUT",
]

InterpType = Literal[
    "stepwise_const_fwd",
    "mixed_const_quadratic",
    "quadratic",
]

PaymentFrequency = Literal["A", "S", "Q"]

CurveDefKind = Literal["param", "lincomb"]

from dataclasses import dataclass
from fixed_income_ad_playground.enums import Currency, FuturesIndex


@dataclass(frozen=True)
class FuturesConvention:
    name: FuturesIndex
    currency: Currency

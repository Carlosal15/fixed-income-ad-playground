from dataclasses import dataclass
from fixed_income_ad_playground.enums import Currency, SwapIndex


@dataclass(frozen=True)
class SwapConvention:
    """
    Minimal convention object.
    discount_curve_id and forecast_curve_id are resolved by "RefData"
    for a given curveset.
    """

    name: SwapIndex
    currency: Currency

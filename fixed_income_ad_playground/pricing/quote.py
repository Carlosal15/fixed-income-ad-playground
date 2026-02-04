"""The intention behind this module is to define a standard way to represent market quotes
for calibration purposes.

We could, in principle, definee multiple quote types (e.g., par rates, prices, spreads),
as well as having weights for the different quotes, and bid/ask spreads, and use this in a
calibration framework that would allow, e.g., weighted calibration, or constrained to bid/ask
ranges, or other capabilities.

It would be dictated by the config, but the quotes would be passed around.
"""

from dataclasses import dataclass
from fixed_income_ad_playground.enums import QuoteType
from fixed_income_ad_playground.instruments.instrument import Instrument


@dataclass(frozen=True)
class Quote:
    instrument: Instrument
    quote_type: QuoteType
    value: float
    bid: float | None = None  # stub
    ask: float | None = None  # stub
    weight: float = 1.0

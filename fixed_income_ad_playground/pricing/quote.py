"""The intention behind this module is to define a standard way to represent market quotes
for calibration purposes.

We could, in principle, definee multiple quote types (e.g., par rates, prices, spreads),
as well as having weights for the different quotes, and bid/ask spreads, and use this in a
calibration framework that would allow, e.g., weighted calibration, or constrained to bid/ask
ranges, or other capabilities.

It would be dictated by the config, but the quotes would be passed around.
"""

from dataclasses import dataclass
from typing import Literal
from fixed_income_ad_playground.instruments.instrument import Instrument


# more of a stub for demo purposes, should be properly enumerated later
QuoteType = Literal["par_rate"]


@dataclass(frozen=True)
class Quote:
    instrument: Instrument
    quote_type: QuoteType
    value: float
    bid: float | None = None
    ask: float | None = None
    weight: float = 1.0

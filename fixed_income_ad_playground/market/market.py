from dataclasses import dataclass

from collections.abc import Mapping
from fixed_income_ad_playground.identifiers.identifiers import CurveId
from fixed_income_ad_playground.curve.curve import Curve

"""
'Market' object that encapsulates the market state. For now, it only contains curves.
Moreover, the first demo version contains only a single curve and all instruments (swaps)
use it as discounting and forecasting.
"""


@dataclass(frozen=True)
class Market:
    curves: Mapping[CurveId, Curve]

    def curve(self, cid: CurveId) -> Curve:
        return self.curves[cid]

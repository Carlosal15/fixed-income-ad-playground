from dataclasses import dataclass
from fixed_income_ad_playground.curve.curve import Curve
from fixed_income_ad_playground.identifiers.identifiers import CurveId


@dataclass(frozen=True)
class Market:
    """
    Market is a mapping from CurveId -> Curve.
    In real life: also includes FX spots, vol surfaces, fixings, etc.

    It is essentially a container of calibrated/usable models for pricing. Only
    curves in this POC.
    """

    curves: dict[CurveId, Curve]

    def curve(self, cid: CurveId) -> Curve:
        if cid not in self.curves:
            raise KeyError(f"Curve not found in market: {cid.name}")
        return self.curves[cid]

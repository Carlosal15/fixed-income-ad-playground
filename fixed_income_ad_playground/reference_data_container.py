from fixed_income_ad_playground.enums import SwapIndex
from fixed_income_ad_playground.identifiers.identifiers import CurveId


class ReferenceDataContainer:
    """
    Tiny mock "refdata" container/database

    """

    def __init__(self, swap_map: dict[SwapIndex, tuple[CurveId, CurveId]]):
        # swap_map[index] = (discount_curve_id, forecast_curve_id)
        self._swap_map = dict(swap_map)

    def swap_curves(self, idx: SwapIndex) -> tuple[CurveId, CurveId]:
        if idx not in self._swap_map:
            raise KeyError(f"Unknown SwapIndex={idx}")
        return self._swap_map[idx]

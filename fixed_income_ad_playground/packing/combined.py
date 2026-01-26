from dataclasses import dataclass
from fixed_income_ad_playground.packing.swaps import PackedSwapQuoteBatch
from fixed_income_ad_playground.packing.futures import PackedFuturesQuoteBatch


@dataclass(frozen=True)
class PackedMarketBatches:
    """
    Container of all instrument blocks used in calibration.
    """

    swaps: PackedSwapQuoteBatch
    futures: PackedFuturesQuoteBatch  # Ostensibly zero for initial demo purposes

from fixed_income_ad_playground.reference_data_container import ReferenceDataContainer
from fixed_income_ad_playground.instruments.futures import StIRFutureSpec
from fixed_income_ad_playground.instruments.instrument import Instrument, PackContext
from fixed_income_ad_playground.instruments.swap import SwapSpec
from fixed_income_ad_playground.market import Market
from fixed_income_ad_playground.pricing.swap_pricer import SwapPricer


class Pricer:
    """
    Generic pricer that can dispatch per instrument.
    """

    swap_pricer: SwapPricer = SwapPricer()

    # TODO: only par-rate currently implemented
    # In a more complete implementation, we'd have have a dynamic dispatcher
    # for different metrics per instrument type.
    def par_rate(
        self,
        market: Market,
        inst: Instrument,
        ctx: PackContext,
        reference_data: ReferenceDataContainer,
    ) -> float:
        if isinstance(inst, SwapSpec):
            return self.swap_pricer.par_rate(market, inst, ctx, reference_data)
        if isinstance(inst, StIRFutureSpec):
            raise NotImplementedError("STIR future pricing stub")
        raise NotImplementedError(f"Pricing not implemented for instrument type: {type(inst)}")

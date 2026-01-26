from dataclasses import dataclass
from fixed_income_ad_playground.identifiers.identifiers import InstrumentId, CurveId
from fixed_income_ad_playground.instruments.instrument import PackContext
from collections.abc import Sequence
from fixed_income_ad_playground.pricing.quote import Quote
from fixed_income_ad_playground.types import JaxArray
from fixed_income_ad_playground.instruments.swap import PackedSwap
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PackedSwapQuoteBatch:
    """
    Fixed-shape arrays for swaps that feed the JAX kernel.
    """

    curve_id: CurveId
    instrument_ids: list[InstrumentId]  # reporting only
    payment_times: JaxArray  # (swap_count, max_payments)
    accrual_factors: JaxArray  # (swap_count, max_payments)
    cashflow_mask: JaxArray  # (swap_count, max_payments)
    maturity_times: JaxArray  # (swap_count,)
    weights: JaxArray  # (swap_count,)


def pack_swaps_to_batch(quotes: Sequence[Quote], ctx: PackContext) -> PackedSwapQuoteBatch:
    packed_swaps: list[PackedSwap] = []
    instrument_ids: list[InstrumentId] = []
    weights_list: list[float] = []

    curve_id: CurveId | None = None

    for quote in quotes:
        if quote.quote_type != "par_rate":
            raise ValueError(f"Unsupported quote_type in swaps: {quote.quote_type}")
        packed = quote.instrument.pack(ctx)
        if not isinstance(packed, PackedSwap):
            raise TypeError("pack_swaps_to_batch expects SwapSpec/PackedSwap only")

        if curve_id is None:
            curve_id = packed.discount_curve
        elif packed.discount_curve != curve_id:
            raise ValueError("v1 swap batch expects a single discount_curve for all swap quotes")

        packed_swaps.append(packed)
        instrument_ids.append(packed.instrument_id)
        weights_list.append(float(quote.weight))

    if curve_id is None:
        raise ValueError("No swaps to pack in pack_swaps_to_batch")
    swap_count = len(packed_swaps)
    max_payments = max(s.payment_times.size for s in packed_swaps) if swap_count > 0 else 0

    payment_times_np = np.zeros((swap_count, max_payments), dtype=np.float64)
    accrual_factors_np = np.zeros((swap_count, max_payments), dtype=np.float64)
    cashflow_mask_np = np.zeros((swap_count, max_payments), dtype=np.float64)
    maturity_times_np = np.zeros((swap_count,), dtype=np.float64)

    for i, packed_swap in enumerate(packed_swaps):
        payment_count = packed_swap.payment_times.size
        payment_times_np[i, :payment_count] = packed_swap.payment_times
        accrual_factors_np[i, :payment_count] = packed_swap.accrual_factors
        cashflow_mask_np[i, :payment_count] = 1.0
        maturity_times_np[i] = packed_swap.maturity_years

    return PackedSwapQuoteBatch(
        curve_id=curve_id,
        instrument_ids=instrument_ids,
        payment_times=jnp.array(payment_times_np),
        accrual_factors=jnp.array(accrual_factors_np),
        cashflow_mask=jnp.array(cashflow_mask_np),
        maturity_times=jnp.array(maturity_times_np),
        weights=jnp.array(np.asarray(weights_list, dtype=np.float64)),
    )

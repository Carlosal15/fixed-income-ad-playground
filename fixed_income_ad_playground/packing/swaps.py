from dataclasses import dataclass
from fixed_income_ad_playground.identifiers.identifiers import CurveId
from fixed_income_ad_playground.instruments.instrument import PackContext
from fixed_income_ad_playground.pricing.quote import Quote
from fixed_income_ad_playground.types import JaxArray
from fixed_income_ad_playground.instruments.swap import PackedSwap
from fixed_income_ad_playground.reference_data_container import ReferenceDataContainer
from fixed_income_ad_playground.shape_policy import ShapePolicy
import jax.numpy as jnp
import numpy as np


# @dataclass(frozen=True)
# class PackedSwapQuoteBatch:
#     """
#     Fixed-shape arrays for swaps that feed the JAX kernel.
#     """

#     curve_id: CurveId
#     instrument_ids: list[InstrumentId]  # reporting only
#     payment_times: JaxJaxArray  # (swap_count, max_payments)
#     accrual_factors: JaxJaxArray  # (swap_count, max_payments)
#     cashflow_mask: JaxJaxArray  # (swap_count, max_payments)
#     maturity_times: JaxJaxArray  # (swap_count,)
#     weights: JaxJaxArray  # (swap_count,)


# def pack_swaps_to_batch(quotes: Sequence[Quote], ctx: PackContext) -> PackedSwapQuoteBatch:
#     packed_swaps: list[PackedSwap] = []
#     instrument_ids: list[InstrumentId] = []
#     weights_list: list[float] = []

#     curve_id: CurveId | None = None

#     for quote in quotes:
#         if quote.quote_type != "par_rate":
#             raise ValueError(f"Unsupported quote_type in swaps: {quote.quote_type}")
#         packed = quote.instrument.pack(ctx)
#         if not isinstance(packed, PackedSwap):
#             raise TypeError("pack_swaps_to_batch expects SwapSpec/PackedSwap only")

#         if curve_id is None:
#             curve_id = packed.discount_curve
#         elif packed.discount_curve != curve_id:
#             raise ValueError("v1 swap batch expects a single discount_curve for all swap quotes")

#         packed_swaps.append(packed)
#         instrument_ids.append(packed.instrument_id)
#         weights_list.append(float(quote.weight))

#     if curve_id is None:
#         raise ValueError("No swaps to pack in pack_swaps_to_batch")
#     swap_count = len(packed_swaps)
#     max_payments = max(s.payment_times.size for s in packed_swaps) if swap_count > 0 else 0

#     payment_times_np = np.zeros((swap_count, max_payments), dtype=np.float64)
#     accrual_factors_np = np.zeros((swap_count, max_payments), dtype=np.float64)
#     cashflow_mask_np = np.zeros((swap_count, max_payments), dtype=np.float64)
#     maturity_times_np = np.zeros((swap_count,), dtype=np.float64)

#     for i, packed_swap in enumerate(packed_swaps):
#         payment_count = packed_swap.payment_times.size
#         payment_times_np[i, :payment_count] = packed_swap.payment_times
#         accrual_factors_np[i, :payment_count] = packed_swap.accrual_factors
#         cashflow_mask_np[i, :payment_count] = 1.0
#         maturity_times_np[i] = packed_swap.maturity_years

#     return PackedSwapQuoteBatch(
#         curve_id=curve_id,
#         instrument_ids=instrument_ids,
#         payment_times=jnp.array(payment_times_np),
#         accrual_factors=jnp.array(accrual_factors_np),
#         cashflow_mask=jnp.array(cashflow_mask_np),
#         maturity_times=jnp.array(maturity_times_np),
#         weights=jnp.array(np.asarray(weights_list, dtype=np.float64)),
#     )


@dataclass(frozen=True)
class PackedSwapBatchPooled:
    # bucketed sizes
    swaps_bucket_size: int
    cashflows_bucket_size: int
    unique_times_bucket_size: int

    # per-swap/cf arrays (bucketed)
    start_times: JaxArray  # (S,M)
    end_times: JaxArray  # (S,M)
    accrual_factors: JaxArray  # (S,M)
    cashflow_mask: JaxArray  # (S,M)
    discount_curve_idx: JaxArray  # (S,)
    forecast_curve_idx: JaxArray  # (S,)
    weights: JaxArray  # (S,)
    notional_per_swap: JaxArray  # (S,)

    # pooled date table and indices
    t_pool: JaxArray  # (Q,)
    idx_start: JaxArray  # (S,M) int32
    idx_end: JaxArray  # (S,M) int32

    # true counts (for reporting)
    swap_count_true: int
    max_cf_true: int
    pooled_time_count_true: int


def pack_swaps_pooled(
    quotes: list[Quote],
    ctx: PackContext,
    refdata: ReferenceDataContainer,
    curve_id_to_idx: dict[CurveId, int],
    shape_policy: ShapePolicy,
) -> PackedSwapBatchPooled:
    packed: list[PackedSwap] = []
    weights: list[float] = []
    disc: list[int] = []
    fcast: list[int] = []
    notionals: list[float] = []

    for q in quotes:
        if q.quote_type != "par_rate":
            raise ValueError("Only par_rate supported in this demo.")
        p = q.instrument.pack(ctx, refdata)
        if not isinstance(p, PackedSwap):
            raise NotImplementedError("Only swaps are implemented in pooled packing.")
        packed.append(p)
        weights.append(float(q.weight))
        disc.append(int(curve_id_to_idx[p.discount_curve]))
        fcast.append(int(curve_id_to_idx[p.forecast_curve]))
        notionals.append(float(p.notional))

    S_true = len(packed)
    M_true = max((p.end_times.size for p in packed), default=0)

    # pool times from true schedules
    # pool times from true schedules
    times = [0.0]
    for s in packed:
        times.extend(s.start_times.tolist())
        times.extend(s.end_times.tolist())
    uniq = np.unique(np.asarray(times, dtype=np.float64))
    uniq.sort()
    Q_true = int(uniq.size)

    pos = {float(t): i for i, t in enumerate(uniq.tolist())}

    shape = shape_policy.key(S_true, M_true, Q_true)

    start_np = np.zeros((shape.swaps_bucket_size, shape.cashflows_bucket_size), dtype=np.float64)
    end_np = np.zeros((shape.swaps_bucket_size, shape.cashflows_bucket_size), dtype=np.float64)
    accrual_np = np.zeros((shape.swaps_bucket_size, shape.cashflows_bucket_size), dtype=np.float64)
    mask_np = np.zeros((shape.swaps_bucket_size, shape.cashflows_bucket_size), dtype=np.float64)

    idx_s_np = np.zeros((shape.swaps_bucket_size, shape.cashflows_bucket_size), dtype=np.int32)
    idx_e_np = np.zeros((shape.swaps_bucket_size, shape.cashflows_bucket_size), dtype=np.int32)

    for i, s in enumerate(packed):
        k = s.end_times.size
        start_np[i, :k] = s.start_times
        end_np[i, :k] = s.end_times
        accrual_np[i, :k] = s.accrual_factors
        mask_np[i, :k] = 1.0
        for j in range(k):
            idx_s_np[i, j] = pos[float(s.start_times[j])]
            idx_e_np[i, j] = pos[float(s.end_times[j])]

    t_pool_np = np.zeros((shape.unique_times_bucket_size,), dtype=np.float64)
    t_pool_np[:Q_true] = uniq

    w_np = np.zeros((shape.swaps_bucket_size,), dtype=np.float64)
    d_np = np.zeros((shape.swaps_bucket_size,), dtype=np.int32)
    f_np = np.zeros((shape.swaps_bucket_size,), dtype=np.int32)
    notional_np = np.zeros((shape.swaps_bucket_size,), dtype=np.float64)
    w_np[:S_true] = np.asarray(weights, dtype=np.float64)
    d_np[:S_true] = np.asarray(disc, dtype=np.int32)
    f_np[:S_true] = np.asarray(fcast, dtype=np.int32)
    notional_np[:S_true] = np.asarray(notionals, dtype=np.float64)

    return PackedSwapBatchPooled(
        swaps_bucket_size=shape.swaps_bucket_size,
        cashflows_bucket_size=shape.cashflows_bucket_size,
        unique_times_bucket_size=shape.unique_times_bucket_size,
        start_times=jnp.array(start_np),
        end_times=jnp.array(end_np),
        accrual_factors=jnp.array(accrual_np),
        cashflow_mask=jnp.array(mask_np),
        discount_curve_idx=jnp.array(d_np),
        forecast_curve_idx=jnp.array(f_np),
        weights=jnp.array(w_np),
        notional_per_swap=jnp.array(notional_np),
        t_pool=jnp.array(t_pool_np),
        idx_start=jnp.array(idx_s_np),
        idx_end=jnp.array(idx_e_np),
        swap_count_true=S_true,
        max_cf_true=M_true,
        pooled_time_count_true=Q_true,
    )

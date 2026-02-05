# from dataclasses import dataclass
# from fixed_income_ad_playground.identifiers.identifiers import InstrumentId, CurveId
# from fixed_income_ad_playground.instruments.instrument import PackContext
# from fixed_income_ad_playground.instruments.futures import PackedFuture
# from fixed_income_ad_playground.pricing.quote import Quote
# from fixed_income_ad_playground.types import JaxArray
# from collections.abc import Sequence
# import numpy as np
# import jax.numpy as jnp


# @dataclass(frozen=True)
# class PackedFuturesQuoteBatch:
#     """
#     Fixed-shape arrays for futures (STUB).
#     """

#     curve_id: CurveId
#     instrument_ids: list[InstrumentId]  # reporting only
#     expiry_times: JaxArray  # (future_count,)
#     weights: JaxArray  # (future_count,)


# def pack_futures_to_batch(quotes: Sequence[Quote], ctx: PackContext) -> PackedFuturesQuoteBatch:
#     packed_futures: list[PackedFuture] = []
#     instrument_ids: list[InstrumentId] = []
#     weights_list: list[float] = []

#     curve_id: CurveId | None = None

#     for quote in quotes:
#         if quote.quote_type != "par_rate":
#             raise ValueError(f"Unsupported qtype in futures stub: {quote.quote_type}")
#         packed = quote.instrument.pack(ctx)
#         if not isinstance(packed, PackedFuture):
#             raise TypeError("pack_futures_to_batch expects FuturesSpec/PackedFuture only")

#         if curve_id is None:
#             curve_id = packed.discount_curve
#         elif packed.discount_curve != curve_id:
#             raise ValueError(
#                 "v1 futures batch expects a single discount_curve for all futures quotes"
#             )

#         packed_futures.append(packed)
#         instrument_ids.append(packed.instrument_id)
#         weights_list.append(float(quote.weight))

#     if curve_id is None and len(packed_futures) != 0:
#         raise ValueError("curve_id must not be None for non-empty futures batch")
#     if curve_id is None:
#         # empty batch needs a curve_id to exist for now, dummy value for now
#         curve_id = CurveId("EMPTY_FUTURES_BLOCK")

#     future_count = len(packed_futures)
#     expiry_times_np = np.zeros((future_count,), dtype=np.float64)
#     for i, packed_future in enumerate(packed_futures):
#         expiry_times_np[i] = packed_future.expiry_years

#     return PackedFuturesQuoteBatch(
#         curve_id=curve_id,
#         instrument_ids=instrument_ids,
#         expiry_times=jnp.array(expiry_times_np),
#         weights=jnp.array(np.asarray(weights_list, dtype=np.float64)),
#     )

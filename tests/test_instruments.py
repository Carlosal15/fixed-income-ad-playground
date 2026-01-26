from fixed_income_ad_playground.identifiers.identifiers import InstrumentId, CurveId
from fixed_income_ad_playground.instruments.swap import SwapSpec, PackedSwap
from fixed_income_ad_playground.instruments.futures import FuturesSpec, PackedFuture
from fixed_income_ad_playground.instruments.instrument import PackContext


def test_swap_pack_short_and_long():
    sid = InstrumentId("swap1")
    cid = CurveId("c")
    ctx = PackContext()

    short = SwapSpec(instrument_id=sid, maturity=0.5, discount_curve=cid)
    packed_short = short.pack(ctx)
    assert isinstance(packed_short, PackedSwap)
    assert packed_short.payment_times.size == 1

    long = SwapSpec(instrument_id=sid, maturity=2.3, discount_curve=cid)
    packed_long = long.pack(ctx)
    assert isinstance(packed_long, PackedSwap)
    assert packed_long.payment_times.size >= 2


def test_futures_pack():
    fid = InstrumentId("fut1")
    cid = CurveId("c")
    ctx = PackContext()

    fut = FuturesSpec(instrument_id=fid, expiry_years=0.25, discount_curve=cid)
    packed = fut.pack(ctx)
    assert isinstance(packed, PackedFuture)
    assert packed.expiry_years == 0.25

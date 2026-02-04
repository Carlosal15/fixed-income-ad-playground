from fixed_income_ad_playground.calibration.calibration import CurveGraphArrays
from fixed_income_ad_playground.identifiers.identifiers import CurveId
from fixed_income_ad_playground.kernels.swaps import swap_pv_pooled
from fixed_income_ad_playground.packing.swaps import PackedSwapBatchPooled
from fixed_income_ad_playground.types import JaxArray
import jax
import jax.numpy as jnp


def pv_jacobian_wrt_params(
    knot_times: JaxArray,
    graph: CurveGraphArrays,
    batch: PackedSwapBatchPooled,
):
    """
    Returns a jitted jac(params_concat, coupon)
    computing dPV/dparams for the given coupon vector.

    To be reused.

    TODO: explore fwd/backward mode for efficiency?
    """

    def jac(params_concat: JaxArray, coupon: JaxArray) -> JaxArray:
        return jax.jacrev(
            lambda x, c: swap_pv_pooled(x, knot_times, graph, batch, c),
            argnums=0,
        )(params_concat, coupon)

    return jax.jit(jac)


def make_bucket_shocks(
    knot_times: JaxArray,
    param_curve_ids: tuple[CurveId, ...],
    bucket_edges: list[float],
) -> JaxArray:
    """
    Build shock directions D of shape (B, P*N):
    For each bucket [e_k, e_{k+1}), set +1e-4 (1bp) on intervals whose midpoints
    fall in the bucket, for all param curves.
    """
    N = knot_times.size - 1
    mids = 0.5 * (knot_times[:-1] + knot_times[1:])
    B = len(bucket_edges) - 1
    P = len(param_curve_ids)
    D = jnp.zeros((B, P * N), dtype=jnp.float64)

    for b in range(B):
        left, right = bucket_edges[b], bucket_edges[b + 1]
        in_bucket = (mids >= left) & (mids < right)  # (N,)
        mask_block = (in_bucket.astype(jnp.float64)) * 1e-4
        for k in range(P):
            start = k * N
            D = D.at[b, start : start + N].set(mask_block)
    return D


def bucketed_dv01_jvp(
    params_concat: JaxArray,
    knot_times: JaxArray,
    graph: CurveGraphArrays,
    batch: PackedSwapBatchPooled,
    coupon_per_swap: JaxArray,
    bucket_dirs: JaxArray,
) -> JaxArray:
    """
    Returns DV01 per swap per bucket: (S, B)
    """
    pv_fn_jit = jax.jit(lambda x, c: swap_pv_pooled(x, knot_times, graph, batch, c))
    _y, lin = jax.linearize(lambda x: pv_fn_jit(x, coupon_per_swap), params_concat)

    def one_dir(d):
        return lin(d)

    return jax.vmap(one_dir, in_axes=0)(bucket_dirs).T

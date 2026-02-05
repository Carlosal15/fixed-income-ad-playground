import jax
import time
from dataclasses import dataclass
from fixed_income_ad_playground.curve.interpolator import Interpolator, make_interpolator
from fixed_income_ad_playground.market import Market
from fixed_income_ad_playground.packing.swaps import PackedSwapBatchPooled
from fixed_income_ad_playground.types import FloatNDArray
from fixed_income_ad_playground.curve.curve import Curve
import numpy as np
import jax.numpy as jnp
from fixed_income_ad_playground.identifiers.identifiers import CurveId
from fixed_income_ad_playground.enums import CurveDefKind
from fixed_income_ad_playground.curve.curve_config import CurveConfig
from fixed_income_ad_playground.types import JaxArray
import jax.tree_util as jtu
from fixed_income_ad_playground.kernels.residuals import residuals_kernel
from fixed_income_ad_playground.enums import InterpType


@dataclass(frozen=True)
class FitDiagnostics:
    swap_rms_bp: float
    swap_max_bp: float
    slope_rms: float
    curv_rms: float


def _raise_with_block_debug(
    residual_vector_np: FloatNDArray,
    block_slices: dict[str, tuple[int, int]],
) -> None:
    bad_indices = np.where(~np.isfinite(residual_vector_np))[0]
    if bad_indices.size == 0:
        return

    affected_blocks: list[str] = []
    for block_name, (start, end) in block_slices.items():
        if np.any((bad_indices >= start) & (bad_indices < end)):
            affected_blocks.append(block_name)

    sample_indices = bad_indices[:10]
    sample_values = residual_vector_np[sample_indices]

    raise ValueError(
        "Non-finite residuals detected. "
        f"affected_blocks={affected_blocks} "
        f"sample_indices={sample_indices.tolist()} "
        f"sample_values={sample_values.tolist()} "
        f"block_slices={block_slices}"
    )


@dataclass(frozen=True)
class CurveDef:
    curve_id: CurveId
    kind: CurveDefKind
    sources: tuple[tuple[CurveId, float], ...] = ()


@dataclass(frozen=True)
class CurveSetSpec:
    """
    - curve_defs: topologically ordered
    - curve_configs: for param curves (and possibly for others in a fuller design)
    """

    curve_defs: list[CurveDef]
    curve_configs: dict[CurveId, CurveConfig]


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class CurveGraphArrays:
    param_pos: JaxArray
    src_idx: JaxArray
    src_w: JaxArray
    src_mask: JaxArray
    is_param: JaxArray
    num_param: int

    def tree_flatten(self) -> tuple[tuple[JaxArray, ...], dict[str, int]]:
        children = (self.param_pos, self.src_idx, self.src_w, self.src_mask, self.is_param)
        aux = {"num_param": int(self.num_param)}
        return children, aux

    @classmethod
    def tree_unflatten(
        cls, aux: dict[str, int], children: tuple[JaxArray, ...]
    ) -> "CurveGraphArrays":
        (param_pos, src_idx, src_w, src_mask, is_param) = children
        return cls(param_pos, src_idx, src_w, src_mask, is_param, int(aux["num_param"]))


def build_curve_graph_arrays(
    curve_defs: list[CurveDef],
) -> tuple[dict[CurveId, int], CurveGraphArrays]:
    curve_id_to_idx: dict[CurveId, int] = {cd.curve_id: i for i, cd in enumerate(curve_defs)}
    n_curves = len(curve_defs)

    param_curve_ids = [cd.curve_id for cd in curve_defs if cd.kind == CurveDefKind.PARAM]
    param_index = {cid: i for i, cid in enumerate(param_curve_ids)}
    num_param = len(param_curve_ids)

    max_sources = max(
        (len(cd.sources) for cd in curve_defs if cd.kind == CurveDefKind.LINEAR_COMB), default=1
    )
    max_sources = max(max_sources, 1)

    param_pos_np = -np.ones((n_curves,), dtype=np.int32)
    is_param_np = np.zeros((n_curves,), dtype=np.int32)
    src_idx_np = np.zeros((n_curves, max_sources), dtype=np.int32)
    src_w_np = np.zeros((n_curves, max_sources), dtype=np.float64)
    src_mask_np = np.zeros((n_curves, max_sources), dtype=np.float64)

    for i, cd in enumerate(curve_defs):
        match cd.kind:
            case CurveDefKind.PARAM:
                is_param_np[i] = 1
                param_pos_np[i] = param_index[cd.curve_id]
            case _:
                for j, (src_cid, w) in enumerate(cd.sources):
                    if src_cid not in curve_id_to_idx:
                        raise ValueError(
                            f"Curve {cd.curve_id.name} depends on unknown {src_cid.name}"
                        )
                    src_idx_np[i, j] = curve_id_to_idx[src_cid]
                    src_w_np[i, j] = float(w)
                    src_mask_np[i, j] = 1.0

    arrays = CurveGraphArrays(
        param_pos=jnp.array(param_pos_np),
        src_idx=jnp.array(src_idx_np),
        src_w=jnp.array(src_w_np),
        src_mask=jnp.array(src_mask_np),
        is_param=jnp.array(is_param_np),
        num_param=num_param,
    )
    return curve_id_to_idx, arrays


# TODO: remove
# @dataclass(frozen=True)
# class StaticCalibrationData:
#     knot_times: JaxArray
#     graph: CurveGraphArrays
#     batch: PackedSwapBatchPooled
#     lam_slope: JaxArray
#     lam_curv: JaxArray
#     spec: CurveSetSpec
#     param_curve_ids: tuple[CurveId, ...]  # order of param blocks
#     curve_id_order: tuple[CurveId, ...]  # curve_defs order (for building full curves)


@dataclass(frozen=True)
class CalibStatic:
    knot_times: JaxArray
    graph: CurveGraphArrays
    batch: PackedSwapBatchPooled
    lam_slope: JaxArray
    lam_curv: JaxArray
    lam_level: JaxArray
    spec: CurveSetSpec
    param_curve_ids: tuple[CurveId, ...]  # order of param blocks
    curve_id_order: tuple[CurveId, ...]  # curve_defs order (for building full curves)


def make_calib_static(
    knot_times_np: FloatNDArray,
    graph: CurveGraphArrays,
    batch: PackedSwapBatchPooled,
    spec: CurveSetSpec,
) -> CalibStatic:
    # param curve order is the order in curve_defs where kind == param
    param_curve_ids = tuple(cd.curve_id for cd in spec.curve_defs if cd.kind == CurveDefKind.PARAM)
    lam_slope_np = np.array(
        [spec.curve_configs[cid].lam_slope for cid in param_curve_ids], dtype=np.float64
    )
    lam_curv_np = np.array(
        [spec.curve_configs[cid].lam_curv for cid in param_curve_ids], dtype=np.float64
    )
    lam_level_np = np.array(
        [spec.curve_configs[cid].lam_level for cid in param_curve_ids], dtype=np.float64
    )

    return CalibStatic(
        knot_times=jnp.array(np.asarray(knot_times_np, dtype=np.float64)),
        graph=graph,
        batch=batch,
        lam_slope=jnp.array(lam_slope_np),
        lam_curv=jnp.array(lam_curv_np),
        lam_level=jnp.array(lam_level_np),
        spec=spec,
        param_curve_ids=param_curve_ids,
        curve_id_order=tuple(cd.curve_id for cd in spec.curve_defs),
    )


@dataclass(frozen=True)
class DebugStats:
    steps: int
    result_code: int


@dataclass(frozen=True)
class CalibrationResult:
    """
    Returned from calibrate_one_date.
    """

    params_concat: JaxArray
    market: Market
    residual_last: np.ndarray | None = None


def compute_forwards_all_curves(
    params_concat: JaxArray, knot_times: JaxArray, graph: CurveGraphArrays
) -> JaxArray:
    """
    params_concat: concatenated blocks for param curves, each block length N
    Returns forwards_all (C,N) for all curves in curve_defs order.
    """
    C = graph.param_pos.shape[0]
    N = knot_times.size - 1
    forwards_all = jnp.zeros((C, N), dtype=jnp.float64)

    def body(i, fa):
        is_param_i = graph.is_param[i]
        pos = graph.param_pos[i]

        def compute_param(fa_in):  # type: ignore
            start = pos * N
            block = jax.lax.dynamic_slice(params_concat, (start,), (N,))
            return fa_in.at[i].set(block)

        def compute_lincomb(fa_in):  # type: ignore
            idxs = graph.src_idx[i]
            ws = graph.src_w[i]
            ms = graph.src_mask[i]
            src_fwds = fa_in[idxs]  # (K,N)
            combo = jnp.sum((ms[:, None] * ws[:, None]) * src_fwds, axis=0)
            return fa_in.at[i].set(combo)

        return jax.lax.cond(is_param_i == 1, compute_param, compute_lincomb, fa)

    return jax.lax.fori_loop(0, C, body, forwards_all)


class CurveSetCalibrator:
    """
    Production-like pattern:
      - keep jitted path pure-array and stable-shape
      - compile once per (ShapeKey, knot grid, curveset graph)
      - build Market from calibrated params for pricing
    """

    def __init__(self, static: CalibStatic):
        self.static = static

        # Pre-build interpolators per curve_id for Market construction
        self._curve_interp: dict[CurveId, Interpolator] = {}
        for cid, cfg in static.spec.curve_configs.items():
            self._curve_interp[cid] = make_interpolator(cfg)

        def res_fn(x, market_par):
            return residuals_kernel(
                x,
                market_par,
                static.knot_times,
                static.graph,
                static.batch,
                static.lam_slope,
                static.lam_curv,
                static.lam_level,
            )

        self.res_fn = res_fn
        self.res_jit = jax.jit(res_fn)
        self.jac_jit = jax.jit(jax.jacfwd(res_fn, argnums=0))

        # JIT helper to compute all curves forwards (for building Market)
        def forwards_all_fn(x):
            return compute_forwards_all_curves(x, static.knot_times, static.graph)  # (C,N)

        self.forwards_all_jit = jax.jit(forwards_all_fn)

    def warmup_residuals_jacobian(self, x0: JaxArray, mp0: JaxArray) -> float:
        t0 = time.perf_counter()
        _ = self.res_jit(x0, mp0).block_until_ready()
        _ = self.jac_jit(x0, mp0).block_until_ready()
        _ = self.forwards_all_jit(x0).block_until_ready()
        return time.perf_counter() - t0

    # LS solvers
    def build_optx_least_squares_value_only(self, solver_kind: str, max_steps: int, jac_mode: str):
        import optimistix as optx

        kind = solver_kind.lower()
        if kind == "gn":
            solver = optx.GaussNewton(rtol=1e-12, atol=1e-12)
        elif kind == "dogleg":
            solver = optx.Dogleg(rtol=1e-12, atol=1e-12)
        elif kind == "lm":
            solver = optx.LevenbergMarquardt(rtol=1e-12, atol=1e-12)
        else:
            raise ValueError(f"unknown LS solver_kind={solver_kind}")

        def fn(y, market_par):
            return self.res_fn(y, market_par)

        def run_value_only(x0, market_par):
            sol = optx.least_squares(
                fn,
                solver,
                x0,
                args=market_par,
                max_steps=max_steps,
                throw=False,
                options={"jac": jac_mode},
            )
            return sol.value  # array only

        return jax.jit(run_value_only), (solver, fn)

    def debug_optx_least_squares_stats(
        self, solver, fn, x0: JaxArray, market_par: JaxArray, max_steps: int, jac_mode: str
    ) -> DebugStats:
        import optimistix as optx

        sol = optx.least_squares(
            fn,
            solver,
            x0,
            args=market_par,
            max_steps=max_steps,
            throw=False,
            options={"jac": jac_mode},
        )
        steps = int(sol.stats.get("num_steps", sol.stats.get("steps", -1)))
        code = int(getattr(sol.result, "_value", 999))
        return DebugStats(steps=steps, result_code=code)

    # Scalar objective solvers
    def build_optx_scalar_value_only(self, solver_kind: str, max_steps: int):
        import optimistix as optx

        kind = solver_kind.lower()
        if kind == "nonlinear_cg":
            solver = optx.NonlinearCG(rtol=1e-12, atol=1e-12)
        elif kind == "lbfgs":
            solver = optx.LBFGS(rtol=1e-12, atol=1e-12)
        else:
            raise ValueError(f"unknown scalar solver_kind={solver_kind}")

        def obj(x, market_par):
            r = self.res_fn(x, market_par)
            return 0.5 * jnp.sum(r * r)

        def fn(y, market_par):
            return obj(y, market_par)

        def run_value_only(x0, market_par):
            sol = optx.minimise(
                fn,
                solver,
                x0,
                args=market_par,
                max_steps=max_steps,
                throw=False,
            )
            return sol.value  # params array

        return jax.jit(run_value_only), (solver, fn)

    def debug_optx_scalar_stats(
        self, solver, fn, x0: JaxArray, market_par: JaxArray, max_steps: int
    ) -> DebugStats:
        import optimistix as optx

        sol = optx.minimise(
            fn,
            solver,
            x0,
            args=market_par,
            max_steps=max_steps,
            throw=False,
        )
        steps = int(sol.stats.get("num_steps", sol.stats.get("steps", -1)))
        code = int(getattr(sol.result, "_value", 999))
        return DebugStats(steps=steps, result_code=code)

    # ---------- Build Market from calibrated parameters ----------
    def build_market(self, params_concat: JaxArray) -> Market:
        """
        Uses the same curveset graph as calibration to produce full curve forwards.
        Then builds python-layer Curve objects with interpolators from CurveConfig.

        NOTE: we support stepwise const forwards fully; other interpolators stub.
        """
        forwards_all = self.forwards_all_jit(params_concat).block_until_ready()  # (C,N)
        forwards_all_np = np.asarray(forwards_all, dtype=np.float64)

        knot_times = self.static.knot_times
        curve_id_to_idx = {cid: i for i, cid in enumerate(self.static.curve_id_order)}

        curves: dict[CurveId, Curve] = {}
        for cid, idx in curve_id_to_idx.items():
            # For derived curves (lincomb), there's no CurveConfig in this POC,
            # so default to stepwise const for discounting.
            cfg = self.static.spec.curve_configs.get(
                cid, CurveConfig(curve_id=cid, interp=InterpType.STEPWISE_CONST_FWD)
            )
            interp = make_interpolator(cfg)
            params_curve = jnp.array(forwards_all_np[idx], dtype=jnp.float64)
            curves[cid] = Curve(
                curve_id=cid, knot_times=knot_times, params=params_curve, interpolator=interp
            )

        return Market(curves=curves)

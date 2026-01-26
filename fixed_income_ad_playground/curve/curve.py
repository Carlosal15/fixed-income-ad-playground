from dataclasses import dataclass
from fixed_income_ad_playground.types import JaxArray
from typing import Literal, Protocol, runtime_checkable
from fixed_income_ad_playground.identifiers.identifiers import CurveId
from fixed_income_ad_playground.kernels.discount_factors import dfs_from_stepwise_const_forwards
import jax.numpy as jnp

# mock values to properly implement later
CurveKind = Literal["piecewise_const_fwd", "quadratic_stub"]


@runtime_checkable
class Curve(Protocol):
    curve_id: CurveId

    def discount_factors(self, t: JaxArray) -> JaxArray: ...
    def zero(self, t: JaxArray) -> JaxArray: ...
    def fwd_inst(self, t: JaxArray) -> JaxArray: ...
    def fwd(self, t1: JaxArray, t2: JaxArray) -> JaxArray: ...


# @dataclass(frozen=True)
# class PiecewiseConstFwdCurve(Curve):
#     """
#     Frozen calibrated curve (OO facade). This wraps the kernel.
#     Not required to be jitted; still JAX-compatible if called with jnp arrays.
#     """
#     curve_id: CurveId
#     kind: CurveKind
#     knot_times: JaxArray                  # (interval_count+1,)
#     interval_forwards: JaxArray           # (interval_count,)

#     def discount_factors(self, t: JaxArray) -> JaxArray:
#         return dfs_from_stepwise_const_forwards(self.knot_times, self.interval_forwards, t)

#     def zero_rates(self, t: JaxArray) -> JaxArray:
#         t = jnp.asarray(t)
#         return -jnp.log(self.discount_factors(t)) / t

#     def instantaneous_forwards(self, t: JaxArray) -> JaxArray:
#         # stepwise constant: choose interval by searchsorted
#         idx = jnp.searchsorted(self.knot_times, t, side="right") - 1
#         idx = jnp.clip(idx, 0, self.interval_forwards.size - 1)
#         return self.interval_forwards[idx]


@dataclass(frozen=True)
class FwdCurve(Curve):
    """
    Instantaneous forward curve with piecewise-constant forwards on knot intervals.
    """

    curve_id: CurveId
    knot_times: JaxArray  # (interval_count+1,)
    forward_params_per_interval: JaxArray  # (interval_count,)

    def discount_factors(self, t: JaxArray) -> JaxArray:
        return dfs_from_stepwise_const_forwards(
            self.knot_times, self.forward_params_per_interval, t
        )

    def zero(self, t: JaxArray) -> JaxArray:
        t = jnp.asarray(t)
        return -jnp.log(self.discount_factors(t)) / t

    def fwd_inst(self, t: JaxArray) -> JaxArray:
        interval_index = jnp.clip(
            jnp.searchsorted(self.knot_times, t, side="right") - 1,
            0,
            self.forward_params_per_interval.size - 1,
        )
        return self.forward_params_per_interval[interval_index]

    def fwd(self, t1: JaxArray, t2: JaxArray) -> JaxArray:
        p1 = self.discount_factors(t1)
        p2 = self.discount_factors(t2)
        return -jnp.log(p2 / p1) / (t2 - t1)

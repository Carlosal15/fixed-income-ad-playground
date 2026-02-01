from dataclasses import dataclass
from fixed_income_ad_playground.curve.interpolator import Interpolator
from fixed_income_ad_playground.types import JaxArray
from fixed_income_ad_playground.identifiers.identifiers import CurveId
import jax.numpy as jnp


@dataclass(frozen=True)
class Curve:
    """
    Python-layer curve object (user-facing).
    """

    curve_id: CurveId
    knot_times: JaxArray  # (N+1,)
    params: JaxArray  # (N,) for stepwise const fwd
    interpolator: Interpolator

    def discount_factors(self, t: JaxArray) -> JaxArray:
        return self.interpolator.discount_factors(self.knot_times, self.params, t)

    def zero_rate(self, t: JaxArray) -> JaxArray:
        t = jnp.asarray(t)
        df = self.discount_factors(t)
        return -jnp.log(df) / t

    def ifr(self, t: JaxArray) -> JaxArray:
        """
        Instantaneous forward for stepwise-const-forward.
        For other interpolators, implement later.
        """
        match self.interpolator.interpolator_type:
            case "stepwise_const_fwd":
                idx = jnp.clip(
                    jnp.searchsorted(self.knot_times, t, side="right") - 1, 0, self.params.size - 1
                )
                return self.params[idx]
            case _:
                raise NotImplementedError(
                    f"fwd_inst not implemented for interpolator type {self.interpolator.interpolator_type}"
                )


# @dataclass(frozen=True)
# class FwdCurve(Curve):
#     """
#     Instantaneous forward curve with piecewise-constant forwards on knot intervals.
#     """

#     curve_id: CurveId
#     knot_times: JaxArray  # (interval_count+1,)
#     forward_params_per_interval: JaxArray  # (interval_count,)

#     def discount_factors(self, t: JaxArray) -> JaxArray:
#         return dfs_from_stepwise_const_forwards(
#             self.knot_times, self.forward_params_per_interval, t
#         )

#     def zero(self, t: JaxArray) -> JaxArray:
#         t = jnp.asarray(t)
#         return -jnp.log(self.discount_factors(t)) / t

#     def ifr(self, t: JaxArray) -> JaxArray:
#         interval_index = jnp.clip(
#             jnp.searchsorted(self.knot_times, t, side="right") - 1,
#             0,
#             self.forward_params_per_interval.size - 1,
#         )
#         return self.forward_params_per_interval[interval_index]

#     def fwd(self, t1: JaxArray, t2: JaxArray) -> JaxArray:
#         p1 = self.discount_factors(t1)
#         p2 = self.discount_factors(t2)
#         return -jnp.log(p2 / p1) / (t2 - t1)

from abc import ABC, abstractmethod
from fixed_income_ad_playground.curve.curve_config import CurveConfig
from fixed_income_ad_playground.kernels.discount_factors import dfs_from_stepwise_const_forwards
from fixed_income_ad_playground.types import JaxArray
from fixed_income_ad_playground.enums import InterpType
from dataclasses import dataclass


class Interpolator(ABC):
    """
    Python layer interpolator interface for the JAX kernels.
    Additional interpolators can be added behind this interface.
    """

    interpolator_type: InterpType

    @abstractmethod
    def discount_factors(self, knot_times: JaxArray, params: JaxArray, t: JaxArray) -> JaxArray: ...


# only one currently implemented
@dataclass(frozen=True)
class StepwiseConstFwdInterpolator(Interpolator):
    interp = InterpType.STEPWISE_CONST_FWD

    def discount_factors(self, knot_times: JaxArray, params: JaxArray, t: JaxArray) -> JaxArray:
        return dfs_from_stepwise_const_forwards(knot_times, params, t)


@dataclass(frozen=True)
class MixedConstQuadraticInterpolator(Interpolator):
    """
    Stub: implement later.
    """

    # In production this should take either a date time or a knot index
    switch_at: float = 0.0
    interp = InterpType.MIXED_CONST_QUADRATIC

    def discount_factors(self, knot_times: JaxArray, params: JaxArray, t: JaxArray) -> JaxArray:
        raise NotImplementedError("mixed_const_quadratic interpolator stub")


def make_interpolator(cfg: CurveConfig) -> Interpolator:
    match cfg.interp:
        case InterpType.STEPWISE_CONST_FWD:
            return StepwiseConstFwdInterpolator()
        case InterpType.MIXED_CONST_QUADRATIC:
            return MixedConstQuadraticInterpolator(switch_at=cfg.mixed_switch_time)
        case InterpType.QUADRATIC:
            raise NotImplementedError("quadratic interpolator stub")
        case _:
            raise ValueError(f"Unknown interpolator={cfg.interp}")

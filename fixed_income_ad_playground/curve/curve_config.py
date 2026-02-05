"""Contains curve config logic.
Defines curve specifications, calibrated parameters, and curve set configuration.

Currently, akin to a stub for demo purposes.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from fixed_income_ad_playground.enums import InterpType

# from fixed_income_ad_playground.prod_interfaces import Quote
from fixed_income_ad_playground.types import FloatNDArray
from typing import Literal
from fixed_income_ad_playground.identifiers.identifiers import CurveId
from fixed_income_ad_playground.pricing.quote import Quote


@dataclass(frozen=True)
class CurveSpec:
    curve_id: CurveId
    knot_times: FloatNDArray  # (interval_count+1,) year fractions, sorted, starting at 0.0
    interp: Literal["piecewise_const_fwd"] = "piecewise_const_fwd"


@dataclass(frozen=True)
class CalibratedCurveParams:
    """Raw calibrated parameters for a curve (v1: instantaneous forwards per interval)."""

    forward_params_per_interval: FloatNDArray  # (interval_count,)


# mock stub for now
@dataclass(frozen=True)
class CurveSetConfig:
    curve_specs: Sequence[CurveSpec]
    quotes: Sequence[Quote]
    solver: Literal["scipy_least_squares"] = "scipy_least_squares"
    lam_slope: float = 0.0
    lam_curv: float = 0.0


@dataclass(frozen=True)
class CurveConfig:
    """
    User-facing curve config. In a real package this would include:
    - interpolator type + params
    - penalties
    - bounds, constraints
    - etc
    """

    curve_id: CurveId
    interp: InterpType = InterpType.STEPWISE_CONST_FWD
    lam_slope: float = 0.0
    lam_curv: float = 0.0
    lam_level: float = 0.0
    # stub params for mixed interpolator
    mixed_switch_time: float = 0.0


# @dataclass(frozen=True)
# class CurveSetConfig:
#     curve_specs: Sequence[CurveSpec]
#     quotes: Sequence[Quote]
#     solver: Literal["scipy_least_squares"] = "scipy_least_squares"
#     lam_slope: float = 0.0
#     lam_curv: float = 0.0

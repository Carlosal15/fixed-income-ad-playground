from dataclasses import dataclass
from fixed_income_ad_playground.types import FloatNDArray
import numpy as np


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

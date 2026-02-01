"""
This module defines a shape policy for bucketing array shapes to reduce JIT recompilations.

We're limiting the number of unique shapes to swaps, cashflows and unique times (common coupon
payment dates across instruments are computed once and reused)

This is a key step in going from the 'user' api to the kernels that 'speak' in JAX arrays.

It uses simple logic with some pre-defined shapes and is only currently implemented for swaps, but can be extended.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeKey:
    swaps_bucket_size: int
    cashflows_bucket_size: int
    unique_times_bucket_size: int


# _BUCKETS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)


@dataclass(frozen=True)
class ShapePolicy:
    """
    Buckets array shapes to reduce JIT recompiles.
    Policy: smallest bucket >= n, from fixed list, else power-of-two.
    """

    def bucket(self, n: int) -> int:
        # just return the next power of bigger than n
        return 1 << (n - 1).bit_length()

    def key(self, swaps_true_size: int, cashflows_true_size: int, times_true_size: int) -> ShapeKey:
        return ShapeKey(
            swaps_bucket_size=self.bucket(swaps_true_size),
            cashflows_bucket_size=self.bucket(cashflows_true_size),
            unique_times_bucket_size=self.bucket(times_true_size),
        )

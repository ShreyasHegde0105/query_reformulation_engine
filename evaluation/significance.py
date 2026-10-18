"""Paired significance tests for comparing two retrieval configurations.

Both tests work on per-query metric values for the *same* queries under two
systems (A = baseline, B = candidate), which is the standard setup for IR
evaluation on a fixed query set.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class PairedComparison:
    """Result of comparing system B against baseline A on one metric."""

    mean_a: float
    mean_b: float
    mean_diff: float        # mean(B - A)
    ci_low: float           # bootstrap 95% CI of mean(B - A)
    ci_high: float
    p_value: float          # two-sided paired randomization test
    n: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def to_dict(self) -> Dict[str, float]:
        return {**dataclasses.asdict(self), "significant": self.significant}


def paired_comparison(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 0,
) -> PairedComparison:
    """Compare per-query scores ``b`` against baseline ``a``.

    * p-value: two-sided paired randomization (sign-flip) test on the mean
      difference — under the null hypothesis each query's A/B labels are
      exchangeable, so each difference's sign is flipped at random.
    * CI: percentile bootstrap (queries resampled with replacement) of the
      mean difference.

    Deterministic for a given ``seed``.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape or a_arr.ndim != 1:
        raise ValueError("a and b must be 1-D sequences of equal length (one value per query).")
    if a_arr.size == 0:
        raise ValueError("At least one query is required.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1.")

    diff = b_arr - a_arr
    n = diff.size
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)

    signs = rng.choice((-1.0, 1.0), size=(n_resamples, n))
    null_means = (signs * diff).mean(axis=1)
    # +1 smoothing: the observed assignment is itself one valid permutation.
    p_value = float((np.sum(np.abs(null_means) >= abs(observed) - 1e-12) + 1) / (n_resamples + 1))

    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diff[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return PairedComparison(
        mean_a=float(a_arr.mean()),
        mean_b=float(b_arr.mean()),
        mean_diff=observed,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=min(p_value, 1.0),
        n=n,
    )


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Use when several comparisons are reported together: an adjusted p below
    0.05 keeps the family-wise error rate at 5% across all of them. Holm is
    uniformly more powerful than plain Bonferroni and assumes nothing about
    dependence between the tests.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[i])
        adjusted[i] = min(running_max, 1.0)
    return adjusted

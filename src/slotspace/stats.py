"""Statistics for SlotSpace reports (DEVELOPMENT.md M8)."""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from scipy import stats as st


def paired_t(a, b) -> float:
    """Two-sided paired t-test p-value for per-query metric vectors a, b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 1.0
    return float(st.ttest_rel(a, b).pvalue)


def randomization_test(a, b, n_perm: int = 10000, seed: int = 13) -> float:
    """Two-sided paired randomization (sign-flip) test on the mean difference."""
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(perm >= obs - 1e-12)) / (n_perm + 1))


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


def bootstrap_ci(a, b, n_boot: int = 1000, seed: int = 13, level: float = 0.95):
    """Percentile bootstrap interval for mean(a - b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(d.mean()), float(lo), float(hi)


def mde_points(n: int, p_discordant: float, alpha: float = 0.05, power: float = 0.8, m: int = 1) -> float:
    """Minimum detectable difference (in points) for a paired binary metric such as Hit@1."""
    z = st.norm.ppf(1 - alpha / (2 * m)) + st.norm.ppf(power)
    return 100.0 * z * math.sqrt(p_discordant / n)


def discordant_rate(a, b) -> float:
    a, b = np.asarray(a) > 0.5, np.asarray(b) > 0.5
    return float(np.mean(a != b))

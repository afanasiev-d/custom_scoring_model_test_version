"""Unit tests for bootstrap.py — the BCa confidence-interval engine.

The strategy is to pin every numerically non-trivial piece to an independent
ground truth:
  * the point statistics against scikit-learn / a direct CDF computation,
  * the closed-form jackknives against brute-force leave-one-out,
  * the BCa machinery against its defining invariants,
  * the public API against statistical invariants (Gini = image of AUC,
    estimate ∈ interval, reproducibility) and a coverage sanity check.
"""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import bootstrap as bs


# ── point statistics ──────────────────────────────────────────────────────
def test_auc_matches_sklearn_exactly(binary_scores):
    score, y = binary_scores
    auc, _ = bs._metrics(score[y == 0], score[y == 1])
    # App convention: positive class = bad, predictor = -score.
    assert auc == pytest.approx(roc_auc_score(y, -score), abs=1e-12)


def test_ks_matches_direct_cdf(binary_scores):
    score, y = binary_scores
    _, ks = bs._metrics(score[y == 0], score[y == 1])
    thr = np.unique(score)
    Fg = np.array([(score[y == 0] <= t).mean() for t in thr])
    Fb = np.array([(score[y == 1] <= t).mean() for t in thr])
    assert ks == pytest.approx(np.abs(Fb - Fg).max(), abs=1e-12)


def test_auc_handles_ties_as_half():
    # 2 goods, 2 bads, one exact tie at 5 -> P(good>bad)=... computed with 0.5 ties.
    good = np.array([5.0, 7.0])
    bad = np.array([3.0, 5.0])
    auc, _ = bs._metrics(good, bad)
    assert auc == pytest.approx(roc_auc_score([0, 0, 1, 1],
                                              -np.array([5.0, 7.0, 3.0, 5.0])), abs=1e-12)


def test_perfect_separation_gives_auc_one_ks_one():
    good = np.array([10.0, 11.0, 12.0])
    bad = np.array([1.0, 2.0, 3.0])
    auc, ks = bs._metrics(good, bad)
    assert auc == pytest.approx(1.0)
    assert ks == pytest.approx(1.0)


# ── exact closed-form jackknife vs brute force ────────────────────────────
def _brute_jackknife(good, bad, which):
    g, b = np.sort(good), np.sort(bad)
    out = []
    idx = 0 if which == "auc" else 1
    for i in range(len(g)):
        out.append(bs._metrics(np.delete(g, i), b)[idx])
    for j in range(len(b)):
        out.append(bs._metrics(g, np.delete(b, j))[idx])
    return np.array(out)


def test_jackknife_auc_is_exact(binary_scores):
    score, y = binary_scores
    g, b = score[y == 0], score[y == 1]
    assert np.allclose(bs._jackknife_auc(g, b), _brute_jackknife(g, b, "auc"), atol=1e-12)


def test_jackknife_ks_is_exact(binary_scores):
    score, y = binary_scores
    g, b = score[y == 0], score[y == 1]
    assert np.allclose(bs._jackknife_ks(g, b), _brute_jackknife(g, b, "ks"), atol=1e-10)


# ── BCa interval mechanics ────────────────────────────────────────────────
def test_bca_interval_contains_estimate_and_is_ordered():
    rng = np.random.default_rng(0)
    boot = rng.normal(0.7, 0.03, 4000)
    jack = rng.normal(0.7, 0.01, 400)
    lo, hi = bs._bca_ci(0.7, boot, jack, alpha=0.05)
    assert lo < 0.7 < hi
    assert lo < hi


def test_bca_zero_acceleration_recovers_percentile_when_unbiased():
    # Symmetric bootstrap, jackknife with zero skew, theta at the median ->
    # z0 ≈ 0, a ≈ 0  ->  BCa collapses to the plain percentile interval.
    rng = np.random.default_rng(1)
    boot = rng.normal(0.5, 0.05, 20000)
    jack = 0.5 + rng.normal(0, 1e-6, 500)        # negligible, symmetric -> a≈0
    lo, hi = bs._bca_ci(0.5, boot, jack, alpha=0.05)
    plo, phi = bs._percentile_ci(boot, 0.05)
    assert lo == pytest.approx(plo, abs=2e-3)
    assert hi == pytest.approx(phi, abs=2e-3)


def test_bca_degenerate_constant_bootstrap_falls_back():
    boot = np.full(100, 0.4)
    lo, hi = bs._bca_ci(0.4, boot, jack=None, alpha=0.05)
    assert lo == hi == pytest.approx(0.4)


# ── public API invariants ─────────────────────────────────────────────────
def test_confidence_intervals_estimates_lie_inside_their_cis(binary_scores):
    score, y = binary_scores
    out = bs.confidence_intervals(score, y, n_boot=500, ci_level=0.95, seed=3)
    m = out["metrics"].set_index("Metric")
    for k in ("KS", "AUC ROC", "Gini"):
        assert m.loc[k, "ci_low"] <= m.loc[k, "estimate"] <= m.loc[k, "ci_high"]
        assert m.loc[k, "ci_low"] < m.loc[k, "ci_high"]
        assert m.loc[k, "se"] > 0


def test_gini_interval_is_exact_image_of_auc_interval(binary_scores):
    score, y = binary_scores
    out = bs.confidence_intervals(score, y, n_boot=400, ci_level=0.95, seed=4)
    m = out["metrics"].set_index("Metric")
    a, g = m.loc["AUC ROC"], m.loc["Gini"]
    # Gini = 2·AUC − 1, scaled to 0–100.
    assert g["estimate"] == pytest.approx((2 * a["estimate"] - 1) * 100)
    assert g["ci_low"] == pytest.approx((2 * a["ci_low"] - 1) * 100)
    assert g["ci_high"] == pytest.approx((2 * a["ci_high"] - 1) * 100)


def test_confidence_intervals_are_reproducible_with_seed(binary_scores):
    score, y = binary_scores
    a = bs.confidence_intervals(score, y, n_boot=300, seed=7)["metrics"]
    b = bs.confidence_intervals(score, y, n_boot=300, seed=7)["metrics"]
    assert np.allclose(a[["estimate", "ci_low", "ci_high", "se"]].to_numpy(),
                       b[["estimate", "ci_low", "ci_high", "se"]].to_numpy())


def test_confidence_intervals_reports_class_counts(binary_scores):
    score, y = binary_scores
    out = bs.confidence_intervals(score, y, n_boot=200, seed=1)
    assert out["n_good"] == int((y == 0).sum())
    assert out["n_bad"] == int((y == 1).sum())
    assert out["n_good"] + out["n_bad"] == len(y)


@pytest.mark.parametrize("y", [np.array([0, 0, 0, 0]), np.array([1, 1, 0])])
def test_confidence_intervals_requires_both_classes(y):
    with pytest.raises(ValueError):
        bs.confidence_intervals(np.arange(len(y), dtype=float), y, n_boot=50)


@pytest.mark.slow
def test_bca_coverage_is_approximately_nominal():
    """Monte-Carlo: the 95% BCa AUC interval should cover the true AUC ≈ 95% of
    the time. Goods ~ N(1,1), bads ~ N(0,1)  ->  true AUC = Φ(1/√2)."""
    from scipy.stats import norm
    true_auc = norm.cdf(1 / np.sqrt(2))
    reps, covered = 200, 0
    rng = np.random.default_rng(99)
    for r in range(reps):
        yy = (rng.random(400) < 0.3).astype(int)
        ss = np.where(yy == 0, rng.normal(1, 1, 400), rng.normal(0, 1, 400))
        a = bs.confidence_intervals(ss, yy, n_boot=500, ci_level=0.95, seed=r)["metrics"]
        a = a.set_index("Metric").loc["AUC ROC"]
        covered += a["ci_low"] <= true_auc <= a["ci_high"]
    # Allow generous Monte-Carlo slack (SE ≈ 1.5% over 200 reps).
    assert 0.90 <= covered / reps <= 0.99

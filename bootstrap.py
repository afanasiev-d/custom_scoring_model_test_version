"""
bootstrap.py — rigorous bootstrap confidence intervals for the rank-based
discrimination metrics (AUC, Gini, KS).

Methodology
-----------
* **Statistics.** For scores ``s`` and binary outcome ``y`` (1 = bad/default),
  with goods ``G = {s : y==0}`` and bads ``B = {s : y==1}``:
    - ``AUC  = P(s_good > s_bad) + ½·P(tie)`` — the Mann–Whitney form; under the
      app's "higher score = better" convention this equals
      ``roc_auc_score(y, −s)``.
    - ``KS   = max_t |F_bad(t) − F_good(t)|`` — the maximum separation of the two
      score CDFs.
    - ``Gini = 2·AUC − 1`` — a strictly increasing transform of AUC.

* **Resampling — stratified.** Goods and bads are resampled with replacement
  *independently*, preserving the original class counts ``(n_good, n_bad)``.
  This is the correct scheme for two-sample rank statistics: it conditions on
  the class design exactly as the metrics do (it is also ``pROC``'s default for
  AUC) and avoids the degenerate, unbalanced resamples a pooled bootstrap can
  produce.

* **Interval — BCa.** Bias-corrected and accelerated (Efron, 1987): second-order
  accurate and *transformation-respecting*, the gold standard. The adjusted
  percentiles are
      α₁ = Φ( z₀ + (z₀ + z_{α/2})   / (1 − a·(z₀ + z_{α/2})) )
      α₂ = Φ( z₀ + (z₀ + z_{1−α/2}) / (1 − a·(z₀ + z_{1−α/2})) )
  with the bias-correction ``z₀`` read from the bootstrap distribution and the
  acceleration ``a`` from the empirical jackknife. The jackknife is computed in
  **closed form** (exact, O(n log n)) rather than by ``n`` re-fits:
    - AUC: leave-one-out via the per-point Mann–Whitney placements —
      ``AUC₋gᵢ = (U − uᵢ)/((n_g−1)·n_b)`` and ``AUC₋bⱼ = (U − vⱼ)/(n_g·(n_b−1))``.
    - KS : dropping one observation shifts one CDF by ``1/(n−1)`` above that
      point; the new maximum gap follows from prefix/suffix running maxima of
      the (shifted) gap curve.

* **Gini.** Because ``Gini = 2·AUC − 1`` is strictly monotone increasing, its
  BCa interval is the exact image of the AUC interval (confidence intervals are
  equivariant under monotone transforms) — so no separate resampling is needed
  and the AUC and Gini intervals are guaranteed mutually consistent.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ──────────────────────────────────────────────────────────────────────────
# Point statistics
# ──────────────────────────────────────────────────────────────────────────
def _metrics(good, bad):
    """``(AUC, KS)`` for separated good/bad score arrays. ``AUC ∈ [0, 1]`` is
    ``P(good > bad) + ½·ties``; ``KS ∈ [0, 1]`` is ``max_t |F_bad − F_good|``."""
    g = np.sort(np.asarray(good, dtype=float))
    b = np.sort(np.asarray(bad, dtype=float))
    ng, nb = len(g), len(b)
    # AUC via the Mann–Whitney placements of the bads against the goods.
    r = np.searchsorted(g, b, side='right')   # goods ≤ bad_j
    l = np.searchsorted(g, b, side='left')    # goods < bad_j
    auc = float(((ng - r) + 0.5 * (r - l)).sum() / (ng * nb))
    # KS over the union of score thresholds.
    thr = np.union1d(g, b)
    Fg = np.searchsorted(g, thr, side='right') / ng
    Fb = np.searchsorted(b, thr, side='right') / nb
    ks = float(np.abs(Fb - Fg).max())
    return auc, ks


# ──────────────────────────────────────────────────────────────────────────
# Exact closed-form jackknife (for the BCa acceleration constant)
# ──────────────────────────────────────────────────────────────────────────
def _jackknife_auc(g, b):
    """Exact leave-one-out AUC values, length ``n_g + n_b`` — O(n log n)."""
    g = np.sort(g); b = np.sort(b)
    ng, nb = len(g), len(b)
    # uᵢ = Σⱼ ψ(gᵢ, bⱼ) = (#bads < gᵢ) + ½(#bads == gᵢ) = ½(l + r)
    lb = np.searchsorted(b, g, side='left')
    rb = np.searchsorted(b, g, side='right')
    u = 0.5 * (lb + rb)
    # vⱼ = Σᵢ ψ(gᵢ, bⱼ) = (#goods > bⱼ) + ½(#goods == bⱼ) = n_g − ½(l + r)
    lg = np.searchsorted(g, b, side='left')
    rg = np.searchsorted(g, b, side='right')
    v = ng - 0.5 * (lg + rg)
    U = float(v.sum())                         # = n_g·n_b·AUC  (= Σ u, by symmetry)
    jg = (U - u) / ((ng - 1) * nb)             # drop good i
    jb = (U - v) / (ng * (nb - 1))             # drop bad j
    return np.concatenate([jg, jb])


def _suffix_max(x):
    """Running maximum from the right: ``out[k] = max(x[k:])``, with a trailing
    ``-inf`` sentinel so ``out[len(x)] = -inf``."""
    out = np.empty(len(x) + 1)
    out[-1] = -np.inf
    out[:-1] = np.maximum.accumulate(x[::-1])[::-1]
    return out


def _jackknife_ks(g, b):
    """Exact leave-one-out KS values, length ``n_g + n_b`` — O(n log n).

    Dropping a good of value ``x`` leaves ``F_bad`` unchanged and shifts the gap
    ``D(t) = F_bad − F_good`` by ``+1/(n_g−1)`` for ``t ≥ x``; the new KS is the
    max of ``|D|`` below ``x`` and ``|D + shift|`` at/above ``x``, obtained from
    prefix/suffix running maxima.  Symmetric construction for a dropped bad.
    """
    g = np.sort(g); b = np.sort(b)
    ng, nb = len(g), len(b)
    thr = np.union1d(g, b)
    cg = np.searchsorted(g, thr, side='right')   # #goods ≤ thr
    cb = np.searchsorted(b, thr, side='right')   # #bads  ≤ thr

    # ── drop a good of value x:  D' = A           for t < x
    #                             D' = A + c       for t ≥ x      (c = 1/(n_g−1))
    A = cb / nb - cg / (ng - 1)
    c = 1.0 / (ng - 1)
    pre = np.concatenate([[-np.inf], np.maximum.accumulate(np.abs(A))])  # pre[k]=max|A[:k]|
    suf = _suffix_max(np.abs(A + c))                                     # suf[k]=max|A+c|[k:]
    kx = np.searchsorted(thr, g, side='left')    # thr[kx] == x
    ks_g = np.maximum(pre[kx], suf[kx])

    # ── drop a bad of value y:   D' = A2          for t < y
    #                             D' = A2 − c2     for t ≥ y      (c2 = 1/(n_b−1))
    A2 = cb / (nb - 1) - cg / ng
    c2 = 1.0 / (nb - 1)
    pre2 = np.concatenate([[-np.inf], np.maximum.accumulate(np.abs(A2))])
    suf2 = _suffix_max(np.abs(A2 - c2))
    ky = np.searchsorted(thr, b, side='left')
    ks_b = np.maximum(pre2[ky], suf2[ky])

    return np.concatenate([ks_g, ks_b])


# ──────────────────────────────────────────────────────────────────────────
# BCa interval
# ──────────────────────────────────────────────────────────────────────────
def _percentile_ci(boot, alpha):
    lo, hi = np.percentile(boot, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    return float(lo), float(hi)


def _bca_ci(theta_hat, boot, jack, alpha):
    """BCa interval. Falls back to the percentile interval when the jackknife is
    unavailable (degenerate class) or the bootstrap distribution is constant."""
    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    B = len(boot)
    if B == 0 or np.ptp(boot) == 0 or jack is None or len(jack) < 2:
        return _percentile_ci(boot, alpha) if B else (theta_hat, theta_hat)

    # Bias correction z0 (with a continuity term for ties), clamped off ±inf.
    prop = (np.sum(boot < theta_hat) + 0.5 * np.sum(boot == theta_hat)) / B
    prop = min(max(prop, 0.5 / B), 1 - 0.5 / B)
    z0 = norm.ppf(prop)

    # Acceleration a = Σ(θ̄ − θ₍ᵢ₎)³ / (6·[Σ(θ̄ − θ₍ᵢ₎)²]^{3/2}).
    d = jack.mean() - jack
    s2 = np.sum(d ** 2)
    a = float(np.sum(d ** 3) / (6.0 * s2 ** 1.5)) if s2 > 0 else 0.0

    zL, zU = norm.ppf(alpha / 2.0), norm.ppf(1 - alpha / 2.0)

    def _adj(z):
        num = z0 + z
        denom = 1.0 - a * num
        if denom <= 0:                       # guard the rare blow-up
            denom = 1e-12
        return float(norm.cdf(z0 + num / denom))

    q1 = min(max(_adj(zL), 0.0), 1.0) * 100.0
    q2 = min(max(_adj(zU), 0.0), 1.0) * 100.0
    return float(np.percentile(boot, q1)), float(np.percentile(boot, q2))


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────
def confidence_intervals(scores, y, n_boot=2000, ci_level=0.95, seed=42, progress=None):
    """Stratified-bootstrap BCa confidence intervals for KS, AUC and Gini.

    Parameters
    ----------
    scores : array-like         model scores (higher = better / lower risk)
    y      : array-like         binary outcome, 1 = bad/default
    n_boot : int                number of stratified bootstrap resamples
    ci_level : float            e.g. 0.95 for a 95% interval
    seed   : int                RNG seed (reproducible)
    progress : callable | None  optional iterable-wrapper (e.g. ``stqdm``) used
                                to render a progress bar over the resamples

    Returns
    -------
    dict with:
      'metrics'  : DataFrame[Metric, estimate, ci_low, ci_high, se] in *display*
                   units — KS and Gini on 0–100, AUC on 0–1.
      'ci_level', 'n_boot', 'n_good', 'n_bad'
    """
    s = np.asarray(scores, dtype=float)
    yv = np.asarray(y).astype(int)
    good = s[yv == 0]
    bad = s[yv == 1]
    ng, nb = len(good), len(bad)
    if ng < 2 or nb < 2:
        raise ValueError('Bootstrap CIs need at least two goods and two bads.')
    alpha = 1.0 - float(ci_level)

    auc_hat, ks_hat = _metrics(good, bad)

    rng = np.random.default_rng(int(seed))
    gs = np.sort(good)
    bs = np.sort(bad)
    boot_auc = np.empty(n_boot)
    boot_ks = np.empty(n_boot)
    it = range(int(n_boot))
    if progress is not None:
        it = progress(it)
    for k in it:
        gg = gs[rng.integers(0, ng, ng)]
        bb = bs[rng.integers(0, nb, nb)]
        boot_auc[k], boot_ks[k] = _metrics(gg, bb)

    jauc = _jackknife_auc(gs, bs)
    jks = _jackknife_ks(gs, bs)

    auc_lo, auc_hi = _bca_ci(auc_hat, boot_auc, jauc, alpha)
    ks_lo, ks_hi = _bca_ci(ks_hat, boot_ks, jks, alpha)

    # Gini = 2·AUC − 1 (monotone): transform the AUC endpoints directly.
    def gini(x):
        return 2.0 * x - 1.0

    se_auc = float(boot_auc.std(ddof=1))
    se_ks = float(boot_ks.std(ddof=1))

    rows = [
        # Metric,   estimate,            ci_low,            ci_high,           se
        ('KS',       ks_hat * 100.0,     ks_lo * 100.0,     ks_hi * 100.0,     se_ks * 100.0),
        ('AUC ROC',  auc_hat,            auc_lo,            auc_hi,            se_auc),
        ('Gini',     gini(auc_hat) * 100, gini(auc_lo) * 100, gini(auc_hi) * 100, se_auc * 2 * 100),
    ]
    metrics = pd.DataFrame(rows, columns=['Metric', 'estimate', 'ci_low', 'ci_high', 'se'])
    return {'metrics': metrics, 'ci_level': float(ci_level), 'n_boot': int(n_boot),
            'n_good': int(ng), 'n_bad': int(nb)}

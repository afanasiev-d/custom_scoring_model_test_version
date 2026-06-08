"""Unit tests for correlation.py — Cramér's V association and the two redundancy
filters (Pearson on WoE features, Cramér's V on binned categoricals)."""
import numpy as np
import pandas as pd
import pytest

import correlation as corr


# ── Cramér's V ─────────────────────────────────────────────────────────────
def test_cramers_v_zero_for_independent(rng):
    x = rng.choice(list("abc"), 2000)
    y = rng.choice(list("xy"), 2000)
    assert corr.cramers_v(pd.Series(x), pd.Series(y)) < 0.1


def test_cramers_v_high_for_perfect_dependence():
    s = pd.Series(["a"] * 100 + ["b"] * 100)
    assert corr.cramers_v(s, s.copy()) > 0.9


def test_cramers_v_degenerate_single_level_is_zero():
    x = pd.Series(["a"] * 50)
    y = pd.Series(["p", "q"] * 25)
    assert corr.cramers_v(x, y) == 0.0


def test_cramers_v_in_unit_interval(rng):
    x = pd.Series(rng.choice(list("abcd"), 500))
    y = pd.Series(rng.choice(list("pqr"), 500))
    v = corr.cramers_v(x, y)
    assert 0.0 <= v <= 1.0


# ── Pearson filter on WoE features ────────────────────────────────────────
def test_filtering_drops_redundant_keeps_stronger(stub_streamlit, rng):
    # `strong` and `dup` share a latent component (collinear > 0.67) while each
    # stays modestly correlated with the target (< 0.67) — so the collinear pair
    # triggers, not the feature↔target pairs, and the weaker member is dropped.
    n = 800
    common = rng.normal(0, 1, n)
    strong = common + rng.normal(0, 0.30, n)
    dup = common + rng.normal(0, 0.55, n)          # noisier copy -> weaker on the target
    indep = rng.normal(0, 1, n)                    # uncorrelated
    y = (common + rng.normal(0, 1.7, n) > 0).astype(float)
    df = pd.DataFrame({"strong": strong, "dup": dup, "indep": indep, "PI": y})
    out = corr.filtering(df, "PI", threshold=0.67)
    cols = out.columns.tolist()
    assert "dup" not in cols                       # the weaker of the collinear pair is dropped
    assert "strong" in cols and "indep" in cols
    assert cols[-1] == "PI"                        # target retained, last


def test_filtering_keeps_everything_when_uncorrelated(stub_streamlit, rng):
    n = 600
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n),
        "c": rng.normal(0, 1, n), "PI": (rng.random(n) < 0.5).astype(float),
    })
    out = corr.filtering(df, "PI", threshold=0.67)
    assert set(out.columns) == {"a", "b", "c", "PI"}


# ── Cramér's V filter on binned categoricals ──────────────────────────────
def test_filtering_categorical_drops_redundant(stub_streamlit, rng):
    n = 1000
    y = (rng.random(n) < 0.5).astype(int)
    cat_a = np.where(y == 1, "hi", "lo")
    flip = rng.random(n) < 0.05
    cat_b = np.where(flip, np.where(cat_a == "hi", "lo", "hi"), cat_a)  # ~= cat_a, weaker on y
    cat_indep = rng.choice(["p", "q"], n)
    df = pd.DataFrame({"cat_a": cat_a, "cat_b": cat_b, "cat_indep": cat_indep, "PI": y})
    out = corr.filtering_categorical(df, "PI", threshold=0.7)
    assert "cat_b" not in out.columns              # redundant, weaker association with target
    assert "cat_a" in out.columns and "cat_indep" in out.columns
    assert "PI" in out.columns


def test_filtering_categorical_noop_with_single_feature(stub_streamlit):
    df = pd.DataFrame({"only": ["a", "b"] * 10, "PI": [0, 1] * 10})
    out = corr.filtering_categorical(df, "PI", threshold=0.7)
    assert set(out.columns) == {"only", "PI"}

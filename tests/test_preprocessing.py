"""Unit tests for preprocessing.py — the data-curation layer (filtering, numeric
coercion, num/cat split, Information Value, and the *Match business guard).
All functions here are pure (no Streamlit)."""
import numpy as np
import pandas as pd
import pytest

import preprocessing as pp


# ── numeric coercion ──────────────────────────────────────────────────────
def test_coerce_numeric_converts_stringy_numbers():
    # Placeholders like '.'/'NULL' are turned into NaN upstream (initial_filtering),
    # so coercion sees numeric strings + NaN.
    df = pd.DataFrame({
        "PI": [0, 1, 0, 1],
        "amount": ["100.5", "200", np.nan, "350.25"],    # numeric-as-string
        "name": ["a", "b", "c", "d"],                    # genuine text
    })
    out, converted = pp.coerce_numeric_columns(df.copy(), "PI")
    assert "amount" in converted and "name" not in converted
    assert pd.api.types.is_numeric_dtype(out["amount"])
    assert out["amount"].tolist() == pytest.approx([100.5, 200.0, np.nan, 350.25], nan_ok=True)


def test_coerce_numeric_leaves_target_and_real_numbers_alone():
    df = pd.DataFrame({"PI": ["0", "1", "0"], "x": [1.0, 2.0, 3.0]})
    out, converted = pp.coerce_numeric_columns(df.copy(), "PI")
    assert "PI" not in converted          # target never coerced
    assert "x" not in converted           # already numeric, untouched


# ── high cardinality ──────────────────────────────────────────────────────
def test_filter_high_cardinality_drops_free_text_keeps_small():
    df = pd.DataFrame({
        "PI": [0, 1] * 10,
        "free_text": [f"v{i}" for i in range(20)],     # 20 unique -> drop
        "small_cat": (["A", "B", "C"] * 7)[:20],       # 3 unique -> keep
        "num": np.arange(20.0),                         # numeric -> never dropped
    })
    out, dropped = pp.filter_high_cardinality(df, "PI", max_cardinality=5)
    assert dropped == ["free_text"]
    assert "small_cat" in out.columns and "num" in out.columns


# ── num / cat split ───────────────────────────────────────────────────────
def test_num_cat_split_routes_binary_numeric_to_categorical():
    df = pd.DataFrame({
        "cont": np.linspace(0, 100, 50),
        "binary01": np.tile([0, 1], 25),     # 0/1 numeric -> treated as categorical
        "text": ["a", "b"] * 25,
    })
    df_num, df_cat = pp.num_cat_split(df)
    assert "cont" in df_num.columns
    assert "binary01" not in df_num.columns and "binary01" in df_cat.columns
    assert "text" in df_cat.columns


# ── missing rate ──────────────────────────────────────────────────────────
def test_missing_rate_is_percent():
    df = pd.DataFrame({"a": [1, np.nan, np.nan, np.nan], "b": [1, 2, 3, 4]})
    mr = pp.missing_rate(df).set_index("Feature")["Missing rate"]
    assert mr["a"] == pytest.approx(75.0)
    assert mr["b"] == pytest.approx(0.0)


# ── Information Value ──────────────────────────────────────────────────────
def test_calc_iv_sum_matches_and_ranks_signal_above_noise(rng):
    n = 2000
    y = (rng.random(n) < 0.4).astype(int)
    # Strong but NOT perfectly separating bins (both bins keep some good & bad, so
    # WoE stays finite — perfect separation would give ±inf, replaced by 0).
    signal = np.where(rng.random(n) < np.where(y == 1, 0.8, 0.25), "A", "B")
    df = pd.DataFrame({"PI": y, "signal": signal, "noise": rng.choice(["p", "q", "r"], n)})
    iv_sig, data = pp.calc_iv(df, "signal", "PI")
    iv_noise, _ = pp.calc_iv(df, "noise", "PI")
    assert iv_sig == pytest.approx(data["IV"].sum())
    assert iv_sig > iv_noise
    assert iv_noise < 0.1                                        # noise is unpredictive


def test_calc_iv_handles_pure_bins_without_inf():
    # A bin with zero bads would give WoE = +inf; the code must replace it with 0.
    df = pd.DataFrame({"PI": [0, 0, 0, 1, 1], "f": ["x", "x", "x", "y", "y"]})
    iv, data = pp.calc_iv(df, "f", "PI")
    assert np.isfinite(iv)
    assert np.isfinite(data["WoE"]).all()


def test_get_init_iv_excludes_target_and_indexes_by_feature(rng):
    df = pd.DataFrame({
        "PI": (rng.random(300) < 0.5).astype(int),
        "a": rng.choice(["x", "y"], 300),
        "b": rng.choice(["m", "n"], 300),
    })
    iv = pp.get_init_iv(df, "PI")
    assert set(iv.index) == {"a", "b"}            # target excluded
    assert list(iv.columns) == ["IV"]


# ── *Match business guard ─────────────────────────────────────────────────
def test_drop_illogical_match_feature():
    # Match ('Y') has the HIGHER bad rate -> contradicts business logic -> drop.
    df = pd.DataFrame({
        "PI": [1] * 30 + [0] * 20 + [1] * 10 + [0] * 40,
        "idMatch_cat": ["['Y']"] * 50 + ["['N']"] * 50,
    })
    out, dropped = pp.drop_illogical_match_features(df, "PI")
    assert dropped == ["idMatch"]
    assert "idMatch_cat" not in out.columns


def test_keep_logical_match_feature():
    # Match ('Y') has the LOWER bad rate -> logical -> keep.
    df = pd.DataFrame({
        "PI": [1] * 10 + [0] * 40 + [1] * 30 + [0] * 20,
        "idMatch_cat": ["['Y']"] * 50 + ["['N']"] * 50,
    })
    out, dropped = pp.drop_illogical_match_features(df, "PI")
    assert dropped == []
    assert "idMatch_cat" in out.columns


def test_non_match_features_are_untouched():
    df = pd.DataFrame({"PI": [0, 1, 0, 1], "income_cat": ["['low']", "['high']"] * 2})
    out, dropped = pp.drop_illogical_match_features(df, "PI")
    assert dropped == [] and "income_cat" in out.columns


# ── initial filtering ─────────────────────────────────────────────────────
def test_initial_filtering_drops_geographic_score_sparse_and_moves_target_last():
    n = 200
    df = pd.DataFrame({
        "PI": np.tile([0, 1], n // 2),
        "good_feature": np.arange(n, dtype=float),
        "credit_score": np.arange(n, dtype=float),        # 'score' -> dropped
        "borrower_state": np.tile([1.0, 2.0], n // 2),    # geographic -> dropped
        "sparse_col": np.where(np.arange(n) < 3, 1.0, np.nan),  # >95% missing -> dropped
    })
    out = pp.initial_filtering(df.copy(), sparse_threshold=0.95, target="PI")
    cols = out.columns.tolist()
    assert "good_feature" in cols
    assert "credit_score" not in cols
    assert "borrower_state" not in cols
    assert "sparse_col" not in cols
    assert cols[-1] == "PI"          # target always moved to the last column

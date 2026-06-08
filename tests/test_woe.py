"""Unit tests for woe.py — Laplace-smoothed Weight-of-Evidence encoding with
neutral treatment of missing values."""
import numpy as np
import pandas as pd
import pytest

import woe


def _make_df():
    """Binned frame with three categories and a known good/bad split:
       A: 40 good / 10 bad   B: 10 good / 40 bad   NaN: 8 good / 2 bad."""
    pi = [0] * 40 + [1] * 10 + [0] * 10 + [1] * 40 + [0] * 8 + [1] * 2
    cat = ["A"] * 50 + ["B"] * 50 + ["NaN"] * 10
    return pd.DataFrame({"PI": pi, "f_cat": cat})


def _expected_woe(good, bad, tot_good, tot_bad, s=0.5):
    return np.log(((good + s) / tot_good) / ((bad + s) / tot_bad))


def test_woe_formula_matches_smoothed_definition():
    df = _make_df()
    out, woe_map = woe.woe_transform(df, "PI", smoothing=0.5)
    tg, tb = 58, 52
    woe_a = _expected_woe(40, 10, tg, tb)
    # Encoding of an 'A' row equals the smoothed WoE of bin A.
    a_rows = df["f_cat"] == "A"
    assert out.loc[a_rows, "f_cat"].to_numpy() == pytest.approx(woe_a)


def test_missing_is_neutral_in_encoding_but_fair_in_map():
    df = _make_df()
    out, woe_map = woe.woe_transform(df, "PI")
    # Model encoding: NaN rows contribute nothing (WoE = 0).
    nan_rows = df["f_cat"] == "NaN"
    assert np.allclose(out.loc[nan_rows, "f_cat"].to_numpy(), 0.0)
    # Display map: the *fair* (non-zero) WoE of the NaN bin is still reported.
    m = woe_map["f_cat"].set_index("Category")
    fair_nan = _expected_woe(8, 2, 58, 52)
    assert m.loc["NaN", "WoE"] == pytest.approx(fair_nan)
    assert abs(fair_nan) > 0.5                       # genuinely non-neutral


def test_woe_map_shape_and_shares():
    df = _make_df()
    out, woe_map = woe.woe_transform(df, "PI")
    m = woe_map["f_cat"]
    assert list(m.columns) == ["Category", "WoE", "Count", "Share (%)"]
    assert m["Count"].sum() == len(df)
    assert m["Share (%)"].sum() == pytest.approx(100.0, abs=0.2)


def test_output_keeps_target_and_index():
    df = _make_df()
    out, _ = woe.woe_transform(df, "PI")
    assert "PI" in out.columns
    assert out["PI"].tolist() == df["PI"].tolist()
    assert list(out.index) == list(df.index)


def test_higher_woe_means_lower_risk_direction():
    # Bin A (mostly good) must have a higher WoE than bin B (mostly bad).
    df = _make_df()
    _, woe_map = woe.woe_transform(df, "PI")
    m = woe_map["f_cat"].set_index("Category")
    assert m.loc["A", "WoE"] > m.loc["B", "WoE"]

"""Unit tests for feature_engineering.py — Box-Cox/power transforms with
monotonicity-preserving trend inheritance, and exclusion of predictors that
carry no declared trend.
"""
import numpy as np
import pandas as pd
import pytest

import feature_engineering as fe


def test_label_naming():
    assert fe._label("x", 0.0) == "x__log"
    assert fe._label("x", 0.5) == "x__pow0.5"
    assert fe._label("x", -2.0) == "x__pow-2"


def test_flip():
    assert fe._flip("asc") == "desc"
    assert fe._flip("desc") == "asc"


def test_boxcox_maps_are_monotonic_increasing():
    """Every λ in LAMBDAS (incl. log) must be an increasing map of x>0 — this is
    *why* the transform preserves the source trend."""
    x = np.linspace(1.0, 50.0, 200)
    for lam in fe.LAMBDAS:
        t = np.log(x) if lam == 0 else (np.power(x, lam) - 1.0) / lam
        assert np.all(np.diff(t) > 0), f"lambda={lam} is not increasing"


def test_transforms_inherit_source_trend(rng):
    n = 500
    x = rng.uniform(1, 100, n)
    df = pd.DataFrame({"feat_desc": x})
    eng, asc, desc = fe.engineer_numerical(df, asc_cols=[], desc_cols=["feat_desc"])
    names = list(eng.columns)
    assert names, "expected some transforms"
    # Increasing maps + declared 'desc' -> every transform routed descending.
    assert set(desc) == set(names)
    assert asc == []


def test_ascending_source_preserved(rng):
    df = pd.DataFrame({"feat_asc": rng.uniform(1, 30, 500)})
    eng, asc, desc = fe.engineer_numerical(df, asc_cols=["feat_asc"], desc_cols=[])
    assert set(asc) == set(eng.columns)
    assert desc == []


def test_undeclared_predictor_is_not_transformed(rng):
    """A predictor with no declared trend (e.g. a new field the user didn't add)
    must produce NO transforms, so nothing leaks into the scorecard."""
    df = pd.DataFrame({
        "declared": rng.uniform(1, 100, 500),
        "new_unadded": rng.uniform(1, 50, 500),
    })
    eng, asc, desc = fe.engineer_numerical(df, asc_cols=["declared"], desc_cols=[])
    assert all("new_unadded" not in c for c in eng.columns)
    assert any(c.startswith("declared") for c in eng.columns)


def test_no_declared_columns_returns_empty(rng):
    df = pd.DataFrame({"a": rng.uniform(1, 10, 300), "b": rng.uniform(1, 10, 300)})
    eng, asc, desc = fe.engineer_numerical(df, asc_cols=[], desc_cols=[])
    assert eng.empty and asc == [] and desc == []
    assert list(eng.index) == list(df.index)   # index preserved for safe concat


def test_negative_values_are_shifted_positive(rng):
    """Columns with non-positive values are shifted to strictly positive so the
    power/log maps stay finite; transforms must contain no inf/-inf."""
    df = pd.DataFrame({"with_negs": rng.normal(0, 5, 500)})
    eng, asc, desc = fe.engineer_numerical(df, asc_cols=["with_negs"], desc_cols=[])
    assert eng.shape[1] > 0
    finite_or_nan = np.isfinite(eng.to_numpy()) | np.isnan(eng.to_numpy())
    assert finite_or_nan.all()


def test_too_few_values_are_skipped():
    df = pd.DataFrame({"tiny": [1.0, 2.0, 3.0]})   # < 20 valid values
    eng, asc, desc = fe.engineer_numerical(df, asc_cols=["tiny"], desc_cols=[])
    assert eng.shape[1] == 0


def test_constant_column_does_not_misroute(rng):
    # A (near-)constant column yields an undefined slope; the code must default to
    # "increasing" and keep the source's declared trend (never crash / never flip).
    df = pd.DataFrame({"const": np.full(300, 7.0)})
    eng, asc, desc = fe.engineer_numerical(df, asc_cols=["const"], desc_cols=[])
    assert desc == []
    assert all(c in asc for c in eng.columns)

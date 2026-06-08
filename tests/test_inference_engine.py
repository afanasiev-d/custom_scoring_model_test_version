"""Unit tests for inference_engine.py — scorecard parsing/application and drift."""
import numpy as np
import pandas as pd
import pytest

import inference_engine as ie


# ── a hand-built scorecard exercising all three bin kinds ──────────────────
@pytest.fixture
def scorecard():
    rows = [
        ('Intercept', '', np.nan, np.nan, 500),
        ('age', '(-inf, 30.0]', 0.0, 0.0, 10),          # numerical, right-closed
        ('age', '(30.0, 50.0]', 0.0, 0.0, 20),
        ('age', '(50.0, inf]', 0.0, 0.0, 30),
        ('age', 'NaN', np.nan, 0.0, 0),
        ('region', "['North', 'South']", 0.0, 0.0, 15),  # categorical, grouped bin
        ('region', "['East']", 0.0, 0.0, -5),
        ('region', 'NaN', np.nan, 0.0, 0),
        ('verified', 'not NaN', 0.0, 0.0, 8),            # geo present/missing bin
        ('verified', 'NaN', np.nan, 0.0, 0),
    ]
    return pd.DataFrame(rows, columns=['Feature', 'Category', 'WoE', 'Share (%)', 'Score'])


def test_parse_scorecard_structure(scorecard):
    p = ie.parse_scorecard(scorecard)
    assert p['base'] == 500.0
    kinds = {f['name']: f['kind'] for f in p['features']}
    assert kinds == {'age': 'numerical', 'region': 'categorical', 'verified': 'present'}
    assert ie.scorecard_features(p) == ['age', 'region', 'verified']


def test_score_dataframe_known_values(scorecard):
    p = ie.parse_scorecard(scorecard)
    df = pd.DataFrame({
        'age':      [25,       40,     np.nan,  60],
        'region':   ['North',  'East', 'West',  'South'],   # 'West' is unseen
        'verified': ['yes',    np.nan, 'x',     'y'],
    })
    # row0: 500 +10(age≤30) +15(North) +8(present)              = 533
    # row1: 500 +20(30–50)  −5(East)   +0(verified missing)     = 515
    # row2: 500 +0(age NaN) +0(unseen) +8(present)              = 508
    # row3: 500 +30(>50)    +15(South) +8(present)              = 553
    assert ie.score_dataframe(df, p).tolist() == [533, 515, 508, 553]


def test_numeric_right_closed_boundary(scorecard):
    p = ie.parse_scorecard(scorecard)
    df = pd.DataFrame({'age': [30.0, 50.0], 'region': ['East', 'East'], 'verified': ['x', 'x']})
    # 30 → (-inf,30] = 10 ; 50 → (30,50] = 20  (right-closed)
    assert ie.score_dataframe(df, p).tolist() == [500+10-5+8, 500+20-5+8]


def test_string_numeric_coercion(scorecard):
    p = ie.parse_scorecard(scorecard)
    df = pd.DataFrame({'age': ['25', 'NULL'], 'region': ['North', 'North'], 'verified': ['x', 'x']})
    # '25' coerces to 25 → 10 ; 'NULL' → NaN → 0
    assert ie.score_dataframe(df, p).tolist() == [533, 523]


def test_missing_feature_scored_neutrally(scorecard):
    p = ie.parse_scorecard(scorecard)
    df = pd.DataFrame({'age': [25], 'verified': ['x']})       # 'region' column absent
    assert ie.missing_features(df, p) == ['region']
    assert ie.score_dataframe(df, p).tolist() == [500 + 10 + 8]   # region contributes 0


def test_parse_interval_and_category_helpers():
    assert ie._parse_interval('(-inf, 10.5]') == (-np.inf, 10.5)
    assert ie._parse_interval('(10.5, inf]') == (10.5, np.inf)
    assert ie._parse_interval("['A', 'B']") is None
    assert ie._parse_category_list("['A', 'B']") == ['A', 'B']
    assert ie._parse_category_list('not NaN') == ['not NaN']


# ── drift: PSI / CSI / bad-rate-by-band ────────────────────────────────────
def test_psi_zero_for_identical(rng):
    x = rng.normal(0, 1, 5000)
    val, table = ie.psi(x, x.copy(), bins=10)
    assert val == pytest.approx(0.0, abs=1e-9)
    assert set(['band', 'reference %', 'current %', 'PSI']).issubset(table.columns)


def test_psi_grows_with_shift(rng):
    ref = rng.normal(0, 1, 5000)
    small = ie.psi(ref, rng.normal(0.3, 1, 5000))[0]
    large = ie.psi(ref, rng.normal(1.5, 1, 5000))[0]
    assert 0 < small < large
    assert large > 0.25                                  # a big shift is flagged


def test_psi_from_labels(rng):
    a = ['x'] * 500 + ['y'] * 500
    assert ie.psi_from_labels(a, list(a)) == pytest.approx(0.0, abs=1e-9)
    b = ['x'] * 900 + ['y'] * 100
    assert ie.psi_from_labels(a, b) > 0.1


def test_characteristic_psi_table(scorecard, rng):
    p = ie.parse_scorecard(scorecard)
    old = pd.DataFrame({'age': rng.normal(40, 10, 800), 'region': rng.choice(['North', 'East'], 800),
                        'verified': rng.choice(['x', np.nan], 800)})
    new = pd.DataFrame({'age': rng.normal(55, 10, 800), 'region': rng.choice(['North', 'East'], 800),
                        'verified': rng.choice(['x', np.nan], 800)})
    csi = ie.characteristic_psi(old, new, p)
    assert set(csi['Characteristic']) == {'age', 'region', 'verified'}
    assert (csi['CSI'] >= 0).all()
    assert csi.iloc[0]['CSI'] >= csi.iloc[-1]['CSI']     # sorted most-drifted first
    assert csi.loc[csi['Characteristic'] == 'age', 'CSI'].iloc[0] > 0.1  # age shifted


def test_bad_rate_by_band(rng):
    s_old = rng.normal(500, 40, 1000); y_old = (rng.random(1000) < 0.3).astype(int)
    s_new = rng.normal(500, 40, 1000); y_new = (rng.random(1000) < 0.3).astype(int)
    br = ie.bad_rate_by_band(s_old, y_old, s_new, y_new, bins=10)
    assert {'score band', 'reference bad rate', 'current bad rate'}.issubset(br.columns)
    assert (br['reference bad rate'].dropna().between(0, 1)).all()

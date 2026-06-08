"""Integration tests.

`test_full_pipeline_synthetic` wires the real pipeline modules together on a
small, strongly-signalled synthetic dataset (fast) and asserts the end-to-end
contracts. `test_app_example_builds` drives the actual Streamlit app end to end
on the bundled example dataset (slow; deselect with ``-m "not slow"``).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import binning
import correlation
import preprocessing as pp
import scoring as sc
import viz
import woe as woe_mod

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.integration
def test_full_pipeline_synthetic(stub_streamlit, rng):
    """split → binning → merge → *Match guard → Cramér's V → WoE → Pearson →
    logistic fit → scoring, asserting the scorecard / CI contracts hold."""
    viz.reset_gallery()
    n = 900
    y = (rng.random(n) < 0.4).astype(int)
    # higher number → lower risk (descending event-rate trend); moderate signal so
    # IV lands inside the (min_iv, 1) selection window.
    risk_num = np.where(y == 1, rng.normal(31, 12, n), rng.normal(37, 12, n)).astype(float)
    risk_cat = np.where(y == 1, rng.choice(["poor", "fair", "fair"], n),
                        rng.choice(["good", "fair", "fair"], n))
    df = pd.DataFrame({"PI": y, "risk_num": risk_num, "risk_cat": risk_cat})

    df_num, df_cat = pp.num_cat_split(df)
    lnf, lcf, lnf_asc, lnf_desc, stats, plots = binning.feature_selection_palencia(
        df_num, df_cat, ["risk_num"], [], [], [], target="PI",
        new_predictors_asc=[], new_predictors_desc=[], min_iv=0.01)
    assert lnf or lcf, "the pipeline should select at least one predictor"

    dfb = binning.merging_for_model(df, lnf, lcf, "PI", lnf_asc, lnf_desc)
    dfb, _ = pp.drop_illogical_match_features(dfb, "PI")
    dfb = correlation.filtering_categorical(dfb, "PI", threshold=0.7)
    df_dum, woe_map = woe_mod.woe_transform(dfb, "PI")
    df_dum = correlation.filtering(df_dum, "PI", threshold=0.95)

    X = df_dum.loc[:, df_dum.columns != "PI"]
    yv = df_dum["PI"]
    assert X.shape[1] >= 1
    lr = LogisticRegression(max_iter=500).fit(X, yv)

    df_ppt, df_scorecard, dash_fig, df_ci = sc.scoring(
        df_dum, X, yv, "PI", lr, woe_map=woe_map,
        target_score=450, target_odds=1, pts_double_odds=80, n_boot=200)

    # ── scorecard contracts ──
    assert (df_scorecard["Feature"] == "Intercept").any()
    all_zero = [f for f, g in df_scorecard.groupby("Feature")
                if f != "Intercept" and (g["Score"] == 0).all()]
    assert all_zero == [], "features that score 0 everywhere must be dropped"

    # ── PPT + CI contracts ──
    assert not df_ppt.empty
    assert len(df_ci) == 3
    for _, r in df_ci.iterrows():
        assert r["CI lower"] <= r["Estimate"] <= r["CI upper"]


@pytest.mark.slow
@pytest.mark.integration
def test_app_example_builds(tmp_path):
    """Drive the full Streamlit app on the bundled example dataset and assert it
    builds a complete result (downloads + scorecard) with zero exceptions."""
    from streamlit.testing.v1 import AppTest

    cwd = os.getcwd()
    os.chdir(PROJ)
    try:
        at = AppTest.from_file(os.path.join(PROJ, "main.py"), default_timeout=900)
        at.run()
        buttons = [b for b in at.button if "Example Dataset" in b.label]
        assert buttons, "example-dataset button missing"
        buttons[0].click().run()

        assert not at.exception, f"app raised: {[getattr(e, 'value', e) for e in at.exception]}"
        assert "cs_results" in at.session_state
        res = at.session_state["cs_results"]
        assert res["xlsx"] and res["zip"]                      # downloads were built
        sc_df = res["scorecard"]
        assert (sc_df["Feature"] == "Intercept").any()
        # no all-zero characteristic survived into the published scorecard
        all_zero = [f for f, g in sc_df.groupby("Feature")
                    if f != "Intercept" and (g["Score"] == 0).all()]
        assert all_zero == []
    finally:
        os.chdir(cwd)

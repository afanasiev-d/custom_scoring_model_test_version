"""Tests for scoring.py.

Pure helpers (`_cutoff_metrics`, `_ci_panel`, `_approval_dashboard`) are tested
directly. The full `scoring()` is exercised as a controlled integration test on a
hand-built WoE frame + fitted model, so the scorecard's core contracts can be
asserted exactly:
  * **additivity** — a client's score reconstructed from the scorecard equals the
    model's `score_rounded`;
  * **zero-coefficient features are dropped**;
  * the displayed KS / CI use the tie-consistent (threshold) statistic.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import bootstrap as bs
import scoring
import viz


# ── _cutoff_metrics ───────────────────────────────────────────────────────
def test_cutoff_metrics_basic_properties(binary_scores):
    score, y = binary_scores
    cm, optimal = scoring._cutoff_metrics(score, y)
    # Approval can only fall as the cut-off rises (approve scores ≥ cut-off).
    assert np.all(np.diff(cm["approval"].to_numpy()) <= 1e-9)
    # default = 100 − good on the accepted book.
    ok = cm["good_rate"].notna()
    assert np.allclose(cm.loc[ok, "default_rate"], 100 - cm.loc[ok, "good_rate"])
    # The optimal cut-off maximises KS.
    assert optimal == cm.loc[cm["ks"].idxmax(), "cutoff"]


def test_cutoff_metrics_ks_is_threshold_based(binary_scores):
    """The cut-off KS must equal the threshold KS the bootstrap uses (not the
    tie-splitting per-row KS)."""
    score, y = binary_scores
    cm, _ = scoring._cutoff_metrics(score, y)
    _, ks_boot = bs._metrics(score[y == 0], score[y == 1])
    assert cm["ks"].max() == pytest.approx(ks_boot * 100, abs=1e-9)


# ── _ci_panel ─────────────────────────────────────────────────────────────
def test_ci_panel_table_is_two_dp_strings(binary_scores):
    import matplotlib
    score, y = binary_scores
    ci = bs.confidence_intervals(score, y, n_boot=200, ci_level=0.95, seed=1)
    table, fig = scoring._ci_panel(ci)
    assert list(table.columns) == ["Metric", "Estimate", "95% CI (BCa)", "Bootstrap SE"]
    # every metric estimate rendered at exactly 2 decimals
    m = ci["metrics"].set_index("Metric")
    for _, row in table.iterrows():
        est = m.loc[row["Metric"], "estimate"]
        assert row["Estimate"] == f"{est:.2f}"
        assert "–" in row["95% CI (BCa)"]
    assert isinstance(fig, matplotlib.figure.Figure)


# ── _approval_dashboard ───────────────────────────────────────────────────
def test_approval_dashboard_traces_and_two_dp(binary_scores):
    score, y = binary_scores
    cm, optimal = scoring._cutoff_metrics(score, y)
    fig = scoring._approval_dashboard(cm, optimal)
    names = {t.name for t in fig.data}
    assert names == {"Approval rate", "Good rate (accepted)",
                     "Default rate (accepted)", "KS separation"}
    for t in fig.data:
        assert "%{y:.2f}" in t.hovertemplate          # unified 2-dp readout


def test_approval_dashboard_diff_colour_logic(binary_scores):
    """Favourable moves are green, adverse moves red — inverted for default rate."""
    score, y = binary_scores
    cm, optimal = scoring._cutoff_metrics(score, y)
    fig = scoring._approval_dashboard(cm, optimal)
    traces = {t.name: t for t in fig.data}
    opt_row = cm.iloc[int((cm["cutoff"] - optimal).abs().values.argmin())]

    def colour_of(span):
        return viz.GOOD if viz.GOOD in span else (viz.BAD if viz.BAD in span else "neutral")

    ap = traces["Approval rate"]
    for y_val, span in zip(ap.y, ap.customdata[:, 0]):
        if y_val > opt_row["approval"] + 1e-9:
            assert colour_of(span) == viz.GOOD       # more approvals = good
        elif y_val < opt_row["approval"] - 1e-9:
            assert colour_of(span) == viz.BAD

    df_ = traces["Default rate (accepted)"]
    for y_val, span in zip(df_.y, df_.customdata[:, 0]):
        if y_val > opt_row["default_rate"] + 1e-9:
            assert colour_of(span) == viz.BAD        # more defaults = bad (inverted)
        elif y_val < opt_row["default_rate"] - 1e-9:
            assert colour_of(span) == viz.GOOD


# ── full scoring() — controlled integration ───────────────────────────────
@pytest.fixture
def scored(rng):
    """A small WoE frame + fitted logistic model, ready for scoring()."""
    n = 600
    f1_vals = np.array([-1.0, 0.3, 1.5])
    f2_vals = np.array([-0.5, 0.8])
    f1 = rng.choice(f1_vals, n)
    f2 = rng.choice(f2_vals, n)
    # Higher WoE -> lower risk: P(bad) falls with f1+f2.
    p_bad = 1.0 / (1.0 + np.exp(1.3 * (f1 + f2)))
    y = (rng.random(n) < p_bad).astype(int)
    X = pd.DataFrame({"f1": f1, "f2": f2})
    lr = LogisticRegression().fit(X, y)
    df_dum = X.copy()
    df_dum["PI"] = y
    woe_map = {
        "f1": pd.DataFrame({"Category": ["A", "B", "C"], "WoE": f1_vals,
                            "Count": [1, 1, 1], "Share (%)": [33.3, 33.3, 33.4]}),
        "f2": pd.DataFrame({"Category": ["lo", "hi"], "WoE": f2_vals,
                            "Count": [1, 1], "Share (%)": [50.0, 50.0]}),
    }
    return df_dum, X, y, lr, woe_map


def _run_scoring(df_dum, X, y, lr, woe_map, **kw):
    return scoring.scoring(df_dum, X, y, "PI", lr, woe_map=woe_map,
                           target_score=450, target_odds=1, pts_double_odds=80,
                           n_boot=kw.get("n_boot", 200), ci_level=0.95)


def test_scoring_returns_four_tuple(stub_streamlit, scored):
    viz.reset_gallery()
    out = _run_scoring(*scored)
    assert len(out) == 4
    df_ppt, df_scorecard, dash_fig, df_ci = out
    assert {"Feature", "Category", "WoE", "Share (%)", "Score"} <= set(df_scorecard.columns)
    assert df_scorecard.iloc[0]["Feature"] == "Intercept"


def test_scorecard_is_additive(stub_streamlit, scored):
    """A client's score, rebuilt by summing scorecard points for the bins they
    fall in (+ base points), must equal the model's score_rounded — the defining
    property of a points-based scorecard."""
    viz.reset_gallery()
    df_dum, X, y, lr, woe_map = scored
    _, df_scorecard, _, _ = _run_scoring(df_dum, X, y, lr, woe_map)

    base = float(df_scorecard.loc[df_scorecard["Feature"] == "Intercept", "Score"].iloc[0])
    recon = np.full(len(df_dum), base, dtype=float)
    for feat in ("f1", "f2"):
        sub = df_scorecard[df_scorecard["Feature"] == feat]
        lookup = dict(zip(sub["WoE"].round(4), sub["Score"]))
        recon += df_dum[feat].round(4).map(lookup).to_numpy()

    assert np.allclose(recon, df_dum["score_rounded"].to_numpy())


def test_zero_coefficient_feature_is_dropped(stub_streamlit, scored):
    df_dum, X, y, lr, woe_map = scored
    lr.coef_ = lr.coef_.copy()
    lr.coef_[0][list(lr.feature_names_in_).index("f2")] = 0.0   # zero out f2
    viz.reset_gallery()
    _, df_scorecard, _, _ = _run_scoring(df_dum, X, y, lr, woe_map)
    feats = set(df_scorecard["Feature"])
    assert "f2" not in feats           # no effect on any score -> dropped
    assert "f1" in feats and "Intercept" in feats


def test_ppt_contracts(stub_streamlit, scored):
    """The Performance Projection Table must be internally consistent and use the
    same accept/reject convention and KS as the rest of the app."""
    df_dum, X, y, lr, woe_map = scored
    viz.reset_gallery()
    df_ppt, _, _, _ = _run_scoring(df_dum, X, y, lr, woe_map)

    expected = {"cutoff_score", "approval rate", "good rate (accepted)",
                "default rate (accepted)", "KS", "marginal odds ratio",
                "marginal good rate", "odds (accepted)", "good rate (rejected)",
                "odds (rejected)"}
    assert expected <= set(df_ppt.columns)

    # default = 1 − good on the accepted book
    assert np.allclose(df_ppt["default rate (accepted)"],
                       1 - df_ppt["good rate (accepted)"], equal_nan=True)
    # rates are valid probabilities
    for c in ("approval rate", "good rate (accepted)", "default rate (accepted)"):
        assert df_ppt[c].between(0, 1).all()
    # approval is monotone: higher cut-off ⇒ fewer approvals
    asc = df_ppt.sort_values("cutoff_score")
    assert np.all(np.diff(asc["approval rate"].to_numpy()) <= 1e-9)
    # the PPT's KS matches the headline / bootstrap threshold KS
    _, ks_boot = bs._metrics(df_dum["score_rounded"][y == 0].to_numpy(),
                             df_dum["score_rounded"][y == 1].to_numpy())
    assert df_ppt["KS"].max() == pytest.approx(ks_boot * 100, abs=1e-9)
    # the max-KS row is the canonical optimal cut-off
    _, cutoff = scoring._cutoff_metrics(df_dum["score_rounded"], df_dum["PI"])
    assert int(df_ppt.loc[df_ppt["KS"].idxmax(), "cutoff_score"]) == int(cutoff)


def test_ci_estimate_uses_threshold_ks(stub_streamlit, scored):
    df_dum, X, y, lr, woe_map = scored
    viz.reset_gallery()
    _, _, _, df_ci = _run_scoring(df_dum, X, y, lr, woe_map)
    ks_est = df_ci.loc[df_ci["Metric"].str.startswith("KS"), "Estimate"].iloc[0]
    _, ks_boot = bs._metrics(df_dum["score_rounded"][y == 0].to_numpy(),
                             df_dum["score_rounded"][y == 1].to_numpy())
    assert ks_est == pytest.approx(round(ks_boot * 100, 2), abs=0.01)

"""Unit tests for eva.py — the KS-lift table and separation plot. Focus on the
tie-consistent (threshold) KS introduced for the per-observation case."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import bootstrap as bs
from eva import eva_dfkslift, eva_pks


def test_kslift_ks_is_at_least_threshold(binary_scores):
    """eva_dfkslift uses the per-row (tie-splitting) KS, which is >= the
    tie-consistent threshold KS the bootstrap uses. (The threshold collapse lives
    in scoring.py's df_kslift; applying it to eva for Step-6 is deferred.)"""
    score, y = binary_scores
    dk = eva_dfkslift(pd.DataFrame({"label": y, "pred": score}))
    ks_eva = (100 * dk["ks"]).max()
    _, ks_boot = bs._metrics(score[y == 0], score[y == 1])
    assert ks_eva >= ks_boot * 100 - 1e-9
    assert 0 <= ks_eva <= 100


def test_kslift_cumulatives_are_monotone_and_bounded(binary_scores):
    score, y = binary_scores
    dk = eva_dfkslift(pd.DataFrame({"label": y, "pred": score}))
    for col in ("cumgood", "cumbad"):
        v = dk[col].to_numpy()
        assert np.all(np.diff(v) >= -1e-9)        # non-decreasing
        assert v.min() >= -1e-9 and v.max() <= 1 + 1e-9
    assert (dk["ks"] >= -1e-9).all()


def test_eva_pks_renders_without_error(binary_scores):
    score, y = binary_scores
    dk = eva_dfkslift(pd.DataFrame({"label": y, "pred": score}))
    fig = plt.figure()
    eva_pks(dk, "")                                # draws onto the current axes
    assert fig.axes                                # something was drawn
    plt.close(fig)

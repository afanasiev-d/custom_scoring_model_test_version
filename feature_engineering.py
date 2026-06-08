import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Box-Cox style power exponents, limited to [-2, 2] (lambda = 0 means log).
LAMBDAS = (0.0, -2.0, -1.0, -0.5, 0.5, 2.0)
_N_WORKERS = min(8, max(1, (os.cpu_count() or 2) - 2))


def _label(col, lam):
    return f'{col}__log' if lam == 0 else f'{col}__pow{lam:g}'


def _flip(trend):
    return 'desc' if trend == 'asc' else 'asc'


def _engineer_one(task):
    """Worker: build the Box-Cox/power transforms of one numerical feature and
    assign each transform's event-rate trend by **preserving monotonicity** from
    the source predictor's declared trend (no Streamlit calls).

    A transform is a monotone map of x, so the transformed feature carries the
    SAME event-rate trend as its source when the map is increasing, and the
    OPPOSITE when it is decreasing. (The Box-Cox/power maps ``(xˡ−1)/λ`` and
    ``log x`` are increasing for every λ used here, so the trend is preserved;
    the empirical slope-sign check keeps this correct if ``LAMBDAS`` or the
    transform family ever change.) Trend is therefore deterministic, not
    re-inferred from a noisy correlation with the target.
    """
    col, x, src_trend = task
    res = {}
    mask = ~np.isnan(x)
    if mask.sum() < 20:                       # too few values to engineer reliably
        return res
    mn = float(np.nanmin(x))
    shift = (1.0 - mn) if mn <= 0 else 0.0     # shift to strictly positive for power/log
    x_pos = x + shift
    for lam in LAMBDAS:
        with np.errstate(all='ignore'):
            t = np.log(x_pos) if lam == 0 else (np.power(x_pos, lam) - 1.0) / lam
        t = np.where(np.isfinite(t), t, np.nan)
        tm = ~np.isnan(t)
        if tm.sum() < 20 or np.nanstd(t[tm]) == 0:
            continue
        # Sign of the (monotone) map x → t: increasing preserves the trend,
        # decreasing flips it. Default to "increasing" (preserve) if undefined.
        xv = x_pos[tm]
        with np.errstate(all='ignore'):
            slope = np.corrcoef(xv, t[tm])[0, 1] if np.std(xv) > 0 else 1.0
        trend = src_trend if (not np.isfinite(slope) or slope >= 0) else _flip(src_trend)
        res[_label(col, lam)] = (t, trend)
    return res


def engineer_numerical(df_num, asc_cols, desc_cols, n_workers=_N_WORKERS):
    """Generate Box-Cox / power transforms (lambda in [-2, 2] + log) of the
    numerical predictors **that carry a declared event-rate trend**, in parallel.

    ``asc_cols`` / ``desc_cols`` are the predictors with a known ascending /
    descending trend (dictionary predictors plus user-added external ones). A new
    predictor that was NOT added carries no trend and is skipped entirely, so its
    transforms never reach binning or the scorecard.

    Each transform **inherits its source predictor's trend** (monotonicity
    preserved — see ``_engineer_one``), so the transform bins in a direction
    consistent with the original characteristic.

    Returns ``(eng_df, asc_features, desc_features)`` — the new transform columns
    and their names split by trend.
    """
    asc_set, desc_set = set(asc_cols), set(desc_cols)
    trend_map = {}
    for c in df_num.columns:                   # only present, declared-trend columns
        if c in asc_set:
            trend_map[c] = 'asc'
        elif c in desc_set:
            trend_map[c] = 'desc'

    tasks = [(c, df_num[c].to_numpy(dtype=float), trend_map[c]) for c in trend_map]
    if not tasks:
        return pd.DataFrame(index=df_num.index), [], []

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_engineer_one, tasks))

    eng, asc, desc = {}, [], []
    for res in results:
        for name, (vals, trend) in res.items():
            eng[name] = vals
            (asc if trend == 'asc' else desc).append(name)

    eng_df = pd.DataFrame(eng, index=df_num.index)
    return eng_df, asc, desc

import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Box-Cox style power exponents, limited to [-2, 2] (lambda = 0 means log).
LAMBDAS = (0.0, -2.0, -1.0, -0.5, 0.5, 2.0)
_N_WORKERS = min(8, max(1, (os.cpu_count() or 2) - 2))


def _label(col, lam):
    return f'{col}__log' if lam == 0 else f'{col}__pow{lam:g}'


def _engineer_one(task):
    """Worker: build the Box-Cox/power transforms of one numerical feature and
    infer each transform's event-rate trend (no Streamlit calls)."""
    col, x, y = task
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
        yc, tc = y[tm], t[tm]
        if len(np.unique(yc)) < 2 or np.std(tc) == 0:
            continue
        corr = float(np.corrcoef(tc, yc)[0, 1])
        if not np.isfinite(corr):
            continue
        # corr > 0  ->  transform rises with the bad flag  ->  ascending event rate
        res[_label(col, lam)] = (t, 'asc' if corr > 0 else 'desc')
    return res


def engineer_numerical(df_num, y, n_workers=_N_WORKERS):
    """Generate Box-Cox / power transforms (lambda in [-2, 2] + log) of every
    numerical feature, computed in parallel across a thread pool.

    Returns ``(eng_df, asc_features, desc_features)``: the new transform columns and
    the names split by inferred monotonic trend (so they bin with the right
    direction downstream). Trend inference is self-correcting — a wrongly-routed
    transform simply yields a low IV and is dropped during binning selection.
    """
    yv = np.asarray(y, dtype=float)
    tasks = [(c, df_num[c].to_numpy(dtype=float), yv) for c in df_num.columns]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_engineer_one, tasks))

    eng, asc, desc = {}, [], []
    for res in results:
        for name, (vals, trend) in res.items():
            eng[name] = vals
            (asc if trend == 'asc' else desc).append(name)

    eng_df = pd.DataFrame(eng, index=df_num.index)
    return eng_df, asc, desc

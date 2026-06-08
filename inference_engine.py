"""Inference / monitoring engine — deploy a published scorecard on fresh data.

Two responsibilities, both pure (no Streamlit), so they are unit-testable:

* **Apply a scorecard.** Parse a "Scorecard" sheet (the exact format the build app
  produces — ``Feature · Category · WoE · Share (%) · Score``, Intercept first) into
  a deployable model and score an arbitrary dataframe: a client's score is the base
  (intercept) points plus, for every characteristic, the points of the bin they fall
  in. Missing / unseen values are neutral (the ``NaN`` bin, 0 points by construction).

* **Drift.** Population Stability Index (PSI) on the score, Characteristic Stability
  Index (CSI = PSI on each characteristic's scorecard-bin distribution) for data
  drift, and bad-rate-by-score-band for concept drift.
"""
import ast
import re

import numpy as np
import pandas as pd

# ── scorecard parsing ──────────────────────────────────────────────────────
# Numerical bins are pandas-Interval strings, right-closed: "(-inf, 10.5]".
_INTERVAL_RE = re.compile(
    r'^\s*[\(\[]\s*(-?inf|[-+]?\d*\.?\d+)\s*,\s*(-?inf|[-+]?\d*\.?\d+)\s*[\]\)]\s*$')


def _to_float(tok):
    t = tok.lower()
    if t in ('inf', '+inf'):
        return np.inf
    if t == '-inf':
        return -np.inf
    return float(t)


def _parse_interval(label):
    """Return ``(left, right)`` for a numerical bin string, else ``None``."""
    m = _INTERVAL_RE.match(str(label))
    return (_to_float(m.group(1)), _to_float(m.group(2))) if m else None


def _parse_category_list(label):
    """A categorical bin label is a list repr like ``"['A', 'B']"`` → ``['A','B']``;
    anything else is treated as a single literal category."""
    try:
        v = ast.literal_eval(str(label))
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v]
    except (ValueError, SyntaxError):
        pass
    return [str(label)]


def parse_scorecard(df_scorecard):
    """Parse a scorecard dataframe into a deployable model:
    ``{'base': float, 'features': [ {name, kind, ...} ]}`` where ``kind`` is
    ``'numerical'`` / ``'categorical'`` / ``'present'`` (the not-NaN/NaN geo bins)."""
    sc = df_scorecard.copy()
    sc.columns = [str(c).strip() for c in sc.columns]
    sc['Feature'] = sc['Feature'].astype(str)
    sc['Category'] = sc['Category'].astype(str)
    sc['Score'] = pd.to_numeric(sc['Score'], errors='coerce').fillna(0.0)

    inter = sc[sc['Feature'] == 'Intercept']
    base = float(inter['Score'].iloc[0]) if len(inter) else 0.0

    features = []
    for name, g in sc[sc['Feature'] != 'Intercept'].groupby('Feature', sort=False):
        rows = list(zip(g['Category'].tolist(), g['Score'].tolist()))
        nan_pts = float(next((s for c, s in rows if c == 'NaN'), 0.0))
        non_nan = [(c, float(s)) for c, s in rows if c != 'NaN']
        labels = [c for c, _ in non_nan]
        intervals = [(_parse_interval(c), s) for c, s in non_nan]

        if non_nan and all(iv is not None for iv, _ in intervals):                # numerical
            parsed = sorted(((iv[0], iv[1], s) for iv, s in intervals), key=lambda t: t[0])
            breaks = np.array([parsed[0][0]] + [r for (_l, r, _s) in parsed], dtype=float)
            points = np.array([s for (_l, _r, s) in parsed], dtype=float)
            features.append({'name': name, 'kind': 'numerical', 'breaks': breaks,
                             'points': points, 'nan_points': nan_pts})
        elif set(labels) <= {'not NaN'}:                                          # geo present/missing
            present_pts = float(next((s for c, s in non_nan if c == 'not NaN'), 0.0))
            features.append({'name': name, 'kind': 'present', 'present_points': present_pts,
                             'nan_points': nan_pts})
        else:                                                                     # categorical
            cat_points, cat_bin = {}, {}
            for c, s in non_nan:
                for member in _parse_category_list(c):
                    cat_points[member] = s
                    cat_bin[member] = c
            features.append({'name': name, 'kind': 'categorical', 'points': cat_points,
                             'bin': cat_bin, 'nan_points': nan_pts})
    return {'base': base, 'features': features}


def scorecard_features(parsed):
    return [f['name'] for f in parsed['features']]


# ── applying the scorecard ─────────────────────────────────────────────────
def _resolve_series(df, name):
    """The numeric series for a characteristic. Handles feature-engineered names
    (``col__log`` / ``col__pow0.5``) by recomputing the Box-Cox/power map from the
    base column. Returns ``None`` if the column is absent / unresolvable.

    Caveat: the transform's positivity shift is recomputed from *this* dataset's
    minimum (the training shift is not stored), so it is exact for strictly
    positive characteristics and approximate for ones with non-positive values.
    """
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce')
    if '__' in name:
        base, suf = name.split('__', 1)
        if base in df.columns:
            x = pd.to_numeric(df[base], errors='coerce').to_numpy(dtype=float)
            finite = x[np.isfinite(x)]
            mn = float(finite.min()) if finite.size else 0.0
            xp = x + ((1.0 - mn) if mn <= 0 else 0.0)
            with np.errstate(all='ignore'):
                if suf == 'log':
                    t = np.log(xp)
                elif suf.startswith('pow'):
                    t = (np.power(xp, float(suf[3:])) - 1.0) / float(suf[3:])
                else:
                    return None
            return pd.Series(np.where(np.isfinite(t), t, np.nan), index=df.index)
    return None


def _feature_points(df, f):
    n = len(df)
    series = None if f['kind'] == 'present' else _resolve_series(df, f['name'])
    if f['kind'] == 'present':
        col = df[f['name']] if f['name'] in df.columns else pd.Series([np.nan] * n)
        return np.where(col.isna().to_numpy(), f['nan_points'], f['present_points']).astype(float)
    if series is None:                                   # unresolved column → neutral
        return np.full(n, f['nan_points'], dtype=float)
    if f['kind'] == 'numerical':
        x = series.to_numpy(dtype=float)
        out = np.full(n, f['nan_points'], dtype=float)
        mask = ~np.isnan(x)
        if mask.any():
            idx = np.clip(np.searchsorted(f['breaks'], x[mask], side='left') - 1,
                          0, len(f['points']) - 1)
            out[mask] = f['points'][idx]
        return out
    # categorical: map category → points; unseen / missing → NaN bin (neutral)
    out = np.full(n, f['nan_points'], dtype=float)
    col = df[f['name']]
    present = ~col.isna().to_numpy()
    if present.any():
        out[present] = col[present].astype(str).map(f['points']).fillna(f['nan_points']).to_numpy(dtype=float)
    return out


def score_dataframe(df, parsed):
    """Return the integer scorecard score for every row of ``df``."""
    total = np.full(len(df), float(parsed['base']), dtype=float)
    for f in parsed['features']:
        total += _feature_points(df, f)
    return np.rint(total).astype(int)


def missing_features(df, parsed):
    """Scorecard characteristics that cannot be resolved from ``df`` (neither a
    direct column nor a recomputable transform) — scored neutrally, worth warning."""
    miss = []
    for f in parsed['features']:
        if f['kind'] == 'present':
            if f['name'] not in df.columns:
                miss.append(f['name'])
        elif _resolve_series(df, f['name']) is None:
            miss.append(f['name'])
    return miss


# ── drift: bin labels, PSI / CSI, bad-rate-by-band ─────────────────────────
def feature_bin_labels(df, f):
    """The scorecard bin each row falls in (for CSI). Numerical → interval index,
    categorical → bin label, present → 'not NaN' / 'NaN'."""
    n = len(df)
    if f['kind'] == 'present':
        col = df[f['name']] if f['name'] in df.columns else pd.Series([np.nan] * n)
        return np.where(col.isna().to_numpy(), 'NaN', 'not NaN').astype(object)
    series = _resolve_series(df, f['name'])
    if series is None:
        return np.array(['NaN'] * n, dtype=object)
    if f['kind'] == 'numerical':
        x = series.to_numpy(dtype=float)
        lab = np.array(['NaN'] * n, dtype=object)
        mask = ~np.isnan(x)
        if mask.any():
            idx = np.clip(np.searchsorted(f['breaks'], x[mask], side='left') - 1,
                          0, len(f['points']) - 1)
            lab[mask] = [f'bin{j}' for j in idx]
        return lab
    col = df[f['name']]
    lab = np.array(['NaN'] * n, dtype=object)
    present = ~col.isna().to_numpy()
    if present.any():
        lab[present] = col[present].astype(str).map(f['bin']).fillna('«unseen»').to_numpy()
    return lab


def psi(expected, actual, bins=10, eps=1e-6):
    """Population Stability Index between a reference (``expected``) and a current
    (``actual``) numeric sample, using quantile bins of the reference. Returns
    ``(psi_value, breakdown_dataframe)``. <0.1 stable · 0.1–0.25 moderate · >0.25 large."""
    expected = np.asarray(expected, dtype=float); actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]; actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return 0.0, pd.DataFrame()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:                                   # (near-)constant reference
        return 0.0, pd.DataFrame()
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / expected.size
    a = np.histogram(actual, bins=edges)[0] / actual.size
    ec, acl = np.clip(e, eps, None), np.clip(a, eps, None)
    contrib = (acl - ec) * np.log(acl / ec)
    labels = [f'[{edges[i]:.4g}, {edges[i+1]:.4g})' for i in range(len(edges) - 1)]
    table = pd.DataFrame({'band': labels, 'reference %': e * 100, 'current %': a * 100,
                          'PSI': contrib})
    return float(contrib.sum()), table


def psi_from_labels(reference_labels, current_labels, eps=1e-6):
    """PSI between two categorical (bin-label) samples — the CSI of a characteristic."""
    ref = pd.Series(reference_labels).value_counts()
    cur = pd.Series(current_labels).value_counts()
    cats = sorted(set(ref.index) | set(cur.index))
    r = np.array([ref.get(c, 0) for c in cats], dtype=float); r = r / max(r.sum(), 1)
    a = np.array([cur.get(c, 0) for c in cats], dtype=float); a = a / max(a.sum(), 1)
    rc, ac = np.clip(r, eps, None), np.clip(a, eps, None)
    return float(((ac - rc) * np.log(ac / rc)).sum())


def characteristic_psi(df_old, df_new, parsed):
    """CSI for every scorecard characteristic, sorted most-drifted first."""
    rows = [{'Characteristic': f['name'],
             'CSI': psi_from_labels(feature_bin_labels(df_old, f), feature_bin_labels(df_new, f))}
            for f in parsed['features']]
    out = pd.DataFrame(rows).sort_values('CSI', ascending=False).reset_index(drop=True)
    out['stability'] = pd.cut(out['CSI'], [-np.inf, 0.1, 0.25, np.inf],
                              labels=['stable', 'moderate shift', 'large shift'])
    return out


def bad_rate_by_band(scores_ref, y_ref, scores_cur, y_cur, bins=10):
    """Bad rate per score band (concept drift): bands are quantiles of the reference
    score; a band-wise change in bad rate while the population is stable signals that
    the score↔risk relationship has shifted."""
    sr = np.asarray(scores_ref, dtype=float); yr = np.asarray(y_ref)
    sc = np.asarray(scores_cur, dtype=float); yc = np.asarray(y_cur)
    edges = np.unique(np.quantile(sr, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return pd.DataFrame()
    edges[0], edges[-1] = -np.inf, np.inf
    ref_bin = np.searchsorted(edges, sr, side='right') - 1
    cur_bin = np.searchsorted(edges, sc, side='right') - 1
    rows = []
    for k in range(len(edges) - 1):
        rm, cm = ref_bin == k, cur_bin == k
        rows.append({
            'score band': f'[{edges[k]:.4g}, {edges[k+1]:.4g})',
            'reference bad rate': float(yr[rm].mean()) if rm.any() else np.nan,
            'current bad rate': float(yc[cm].mean()) if cm.any() else np.nan,
            'reference n': int(rm.sum()), 'current n': int(cm.sum()),
        })
    return pd.DataFrame(rows)

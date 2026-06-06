
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2_contingency
import streamlit as st
import viz

# Sequential colormap for Cramér's V (which lives in [0, 1]).
_CRAMERS_CMAP = LinearSegmentedColormap.from_list('cramers', ['#FFFFFF', '#CFF1F5', viz.TEAL, viz.NAVY], N=256)


def _heatmap(matrix, title, subtitle, cbar_label, cmap, vmin, vmax, center, capture_name, ticks):
    """Shared lower-triangle heatmap renderer (themed, captured to the gallery)."""
    n = len(matrix.columns)
    annotate = n <= 28
    side = float(np.clip(1.4 + 0.6 * n, 7, 18))
    fig, ax = plt.subplots(figsize=(side, side * 0.88))
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    sn.heatmap(matrix, mask=mask, ax=ax, cmap=cmap, center=center, vmin=vmin, vmax=vmax,
               annot=annotate, fmt='.2f', annot_kws={'size': 8.5, 'color': viz.INK},
               linewidths=0.8, linecolor=viz.BG, square=True,
               cbar_kws={'shrink': 0.4, 'label': cbar_label, 'ticks': ticks})
    viz.title(ax, title, subtitle)
    ax.tick_params(left=False, bottom=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    fig.tight_layout()
    viz.capture(capture_name, fig)
    st.pyplot(fig, width='content')


def filtering(df_dum, target, threshold=0.67):
    corrMatrix=df_dum.corr().round(2)
    correlated_features=set()
    for i in range(len(corrMatrix.columns)):
        for j in range(i):
            if abs(corrMatrix.iloc[j,i]) > threshold:
                if (abs(corrMatrix.iloc[j,-1])>abs(corrMatrix.iloc[i,-1])):
                    colname = corrMatrix.columns[i]
                else:
                    colname = corrMatrix.columns[j]
                correlated_features.add(colname)
    new_columns=list(set(df_dum.columns.tolist())-correlated_features)
    new_columns.remove(target)
    new_columns.sort()
    new_columns.append(target)
    corrMatrix=df_dum[new_columns].corr().round(2)
    _heatmap(corrMatrix, 'Feature Correlation Matrix (WoE)',
             f'Pairs above |{threshold:.2f}| are dropped to avoid multicollinearity',
             'Pearson correlation', viz.CORR_CMAP, -1, 1, 0, '2_correlation_matrix_woe', [-1, -0.5, 0, 0.5, 1])
    df_dum=df_dum[new_columns]

    return df_dum


def cramers_v(x, y):
    """Bias-corrected Cramér's V association between two categorical series (0..1)."""
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = ct.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def filtering_categorical(df, target, threshold=0.7):
    """Drop binned categorical features that are highly associated with each other
    (Cramér's V > threshold), keeping the one more associated with the target."""
    feats = [c for c in df.columns if c != target]
    if len(feats) < 2:
        return df
    cols = feats + [target]
    V = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = round(cramers_v(df[cols[i]], df[cols[j]]), 2)
            V.iloc[i, j] = v
            V.iloc[j, i] = v

    drop = set()
    for i in range(len(feats)):
        for j in range(i):
            fi, fj = feats[i], feats[j]
            if V.loc[fi, fj] > threshold:
                drop.add(fi if V.loc[fi, target] <= V.loc[fj, target] else fj)

    keep = sorted(c for c in feats if c not in drop) + [target]
    _heatmap(V.loc[keep, keep], "Categorical Association (Cramér's V)",
             f"Categorical pairs above {threshold:.2f} are dropped to avoid redundancy",
             "Cramér's V", _CRAMERS_CMAP, 0, 1, None, '1_categorical_association', [0, 0.25, 0.5, 0.75, 1])

    return df[keep]

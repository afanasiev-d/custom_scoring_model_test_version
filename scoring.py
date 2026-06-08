
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from io import BytesIO
from sklearn.metrics import roc_auc_score, roc_curve
import plotly.graph_objects as go
from stqdm import stqdm
from eva import eva_dfkslift, eva_pks
import viz
import bootstrap

plot_type = ['ks']
title=''


def _cutoff_metrics(scores, y):
    """Per-cut-off decision metrics (vectorised): approval rate, good rate and
    default rate of the accepted book, and the KS separation at each threshold
    (approve scores ≥ cut-off). Returns (dataframe, optimal_cutoff) where the
    optimal cut-off is the one that maximises KS."""
    sc = np.asarray(scores, dtype=float); yv = np.asarray(y)
    good = np.sort(sc[yv == 0]); bad = np.sort(sc[yv == 1]); alls = np.sort(sc)
    G, B, N = max(len(good), 1), max(len(bad), 1), max(len(sc), 1)
    thr = np.unique(sc)
    approval = (N - np.searchsorted(alls, thr, 'left')) / N * 100.0
    n_good_acc = len(good) - np.searchsorted(good, thr, 'left')
    n_bad_acc = len(bad) - np.searchsorted(bad, thr, 'left')
    n_acc = n_good_acc + n_bad_acc
    good_rate = np.where(n_acc > 0, n_good_acc / np.maximum(n_acc, 1) * 100.0, np.nan)
    ks = np.abs(np.searchsorted(bad, thr, 'left') / B
                - np.searchsorted(good, thr, 'left') / G) * 100.0
    cm = pd.DataFrame({'cutoff': thr, 'approval': approval, 'good_rate': good_rate,
                       'default_rate': 100.0 - good_rate, 'ks': ks})
    optimal = float(thr[int(np.nanargmax(ks))])
    return cm, optimal


def _approval_dashboard(cm, optimal):
    """Interactive Plotly decision dashboard: approval / good / default / KS by
    cut-off, with the optimal (max-KS) cut-off pinned and a unified hover that
    reports each metric (2 dp) and its absolute difference (percentage points)
    vs. the optimal cut-off — coloured green when the move is favourable and red
    when adverse (higher approval / good / KS is better; higher default worse)."""
    opt = cm.iloc[int((cm['cutoff'] - optimal).abs().values.argmin())]
    fig = go.Figure()
    for col, name, color, unit, higher_is_better in (
            ('approval', 'Approval rate', viz.TEAL, '%', True),
            ('good_rate', 'Good rate (accepted)', viz.GOOD, '%', True),
            ('default_rate', 'Default rate (accepted)', viz.BAD, '%', False),
            ('ks', 'KS separation', viz.NAVY, '%', True)):
        base = float(opt[col])
        # Absolute difference in percentage points (NOT relative to the optimal):
        # e.g. optimal approval 40.31% and this cut-off 60.87% -> +20.56%.
        diff = cm[col].to_numpy() - base
        favourable = (diff > 0) == higher_is_better
        diff_str = []
        for d, fav in zip(diff, favourable):
            if np.isnan(d):
                diff_str.append('')
            else:
                c = viz.SLATE if d == 0 else (viz.GOOD if fav else viz.BAD)
                diff_str.append(f'<span style="color:{c}">{d:+.2f}{unit}</span>')
        fig.add_trace(go.Scatter(
            x=cm['cutoff'], y=cm[col], name=name, mode='lines',
            line=dict(color=color, width=2),
            customdata=np.array(diff_str, dtype=object).reshape(-1, 1),
            hovertemplate=f'{name}: %{{y:.2f}}{unit}  (%{{customdata[0]}} vs optimal)<extra></extra>'))
    fig.add_vline(x=optimal, line=dict(color=viz.GOLD, width=1.4, dash='dash'))
    fig.add_annotation(x=optimal, y=1.03, yref='paper', showarrow=False, xanchor='left',
                       text=f'★ Optimal cut-off {optimal:.0f} (max KS)',
                       font=dict(color=viz.NAVY, size=11))
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Inter, sans-serif', color=viz.SLATE, size=12),
        title=dict(text='Approval Strategy — interactive cut-off dashboard',
                   x=0.0, xanchor='left', font=dict(color=viz.INK, size=15)),
        hovermode='x unified', height=460,
        margin=dict(l=60, r=30, t=64, b=70),
        legend=dict(orientation='h', y=-0.2, x=0, font=dict(size=10)),
        xaxis=dict(title='Cut-off score  (approve applicants scoring above the cut-off)',
                   showgrid=True, gridcolor=viz.GRID, zeroline=False),
        yaxis=dict(title='Percent (%) / KS', showgrid=True, gridcolor=viz.GRID,
                   zeroline=False, rangemode='tozero'),
        paper_bgcolor='white', plot_bgcolor='white',
    )
    return fig


def _ci_panel(ci):
    """Render the bootstrap confidence intervals as (display table, forest plot).

    The table is built entirely from strings (Arrow-safe under pandas 3 — a mix
    of numeric scales in one column otherwise trips the Streamlit/xlsx writer).
    KS and Gini are shown on 0–100 and AUC on 0–1; the forest plot places all
    three on a shared 0–100 axis with AUC scaled ×100 for visual comparability.
    All metrics use 2 decimals everywhere (headline, table, plot, Excel) so the
    same number reads identically across the app.
    """
    DP = 2                                          # decimals — unified for all metrics
    lvl = int(round(ci['ci_level'] * 100))
    rows = ci['metrics']

    disp = []
    for _, r in rows.iterrows():
        disp.append({
            'Metric': r['Metric'],
            'Estimate': f"{r['estimate']:.{DP}f}",
            f'{lvl}% CI (BCa)': f"{r['ci_low']:.{DP}f} – {r['ci_high']:.{DP}f}",
            'Bootstrap SE': f"{r['se']:.{DP}f}",
        })
    table = pd.DataFrame(disp)

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    colors = {'KS': viz.GOLD, 'AUC ROC': viz.TEAL, 'Gini': viz.NAVY}
    labels, ys = [], []
    for i, (_, r) in enumerate(rows.iterrows()):
        scale = 100.0 if r['Metric'] == 'AUC ROC' else 1.0
        est, lo, hi = r['estimate'] * scale, r['ci_low'] * scale, r['ci_high'] * scale
        yy = len(rows) - 1 - i                     # first row at the top
        col = colors.get(r['Metric'], viz.NAVY)
        ax.errorbar(est, yy, xerr=[[est - lo], [hi - est]], fmt='o', color=col, ecolor=col,
                    elinewidth=viz.LW, capsize=4, markersize=7,
                    markeredgecolor=viz.BG, markeredgewidth=1.0, zorder=5)
        # Uniform 2 dp everywhere (matches the headline metrics and the table).
        ax.annotate(f'{est:.2f}  [{lo:.2f}, {hi:.2f}]', xy=(est, yy), xytext=(0, 9),
                    textcoords='offset points', ha='center', fontsize=8, color=viz.INK)
        labels.append(r['Metric'] + (' ×100' if scale == 100 else ''))
        ys.append(yy)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel('Metric value  (KS & Gini on 0–100; AUC scaled ×100)')
    ax.grid(axis='x', color=viz.GRID, lw=0.8); ax.set_axisbelow(True)
    viz.title(ax, 'Discrimination — Bootstrap Confidence Intervals',
              f"{lvl}% BCa · {ci['n_boot']:,} stratified resamples · "
              f"n={ci['n_good'] + ci['n_bad']:,} ({ci['n_good']:,} good / {ci['n_bad']:,} bad)")
    sns.despine(ax=ax, left=True)
    fig.tight_layout()
    return table, fig


def scoring(df_dum, X_dum, y_dum, target, lr, woe_map=None, target_score = 450, target_odds = 1, pts_double_odds = 80, n_boot = 2000, ci_level = 0.95):

    df_dum['logit']=np.log(lr.predict_proba(X_dum)[:,0]/lr.predict_proba(X_dum)[:,1])
    df_dum['odds'] = np.exp(df_dum['logit'])
    df_dum['probs'] = df_dum['odds'] / (df_dum['odds'] + 1)
    factor = pts_double_odds / np.log(2)
    offset = target_score - factor * np.log(target_odds)
    df_dum['score'] = offset + factor * df_dum['logit']

    # WoE scorecard: each feature has one coefficient and the points awarded for a
    # category are round(-factor * coef * WoE). The row score is the sum of those
    # per-category points plus the intercept points.
    feat_cols = list(lr.feature_names_in_)
    coefs = lr.coef_[0]
    intercept_points = float(np.round(offset - factor * lr.intercept_[0]))
    points = pd.DataFrame(index=df_dum.index)
    for i, col in enumerate(feat_cols):
        points[col] = np.round(-factor * coefs[i] * df_dum[col].to_numpy())
    df_dum['score_rounded'] = points.sum(axis=1) + intercept_points

    groupnum=len(df_dum.index)
    def n0(x): return sum(x==0)
    def n1(x): return sum(x==1)
    df_kslift = df_dum.sort_values('score_rounded', ascending=True).reset_index(drop=True)\
          .assign(group=lambda x: np.ceil((x.index+1)/(len(x.index)/groupnum)))\
          .groupby('group')[target].agg([n0,n1])\
          .reset_index().rename(columns={'n0':'good','n1':'bad'})\
          .assign(
            group=lambda x: (x.index+1)/len(x.index),
            good_distri=lambda x: x.good/sum(x.good), 
            bad_distri=lambda x: x.bad/sum(x.bad), 
            badrate=lambda x: x.bad/(x.good+x.bad),
            cumbadrate=lambda x: np.cumsum(x.bad)/np.cumsum(x.good+x.bad),
            lift=lambda x: (np.cumsum(x.bad)/np.cumsum(x.good+x.bad))/(sum(x.bad)/sum(x.good+x.bad)),
            cumgood=lambda x: np.cumsum(x.good)/sum(x.good), 
            cumbad=lambda x: np.cumsum(x.bad)/sum(x.bad)
          ).assign(ks=lambda x:abs(x.cumbad-x.cumgood))
    df_kslift=pd.concat([
        pd.DataFrame({'group':0, 'good':0, 'bad':0, 'good_distri':0, 'bad_distri':0, 'badrate':0, 'cumbadrate':np.nan, 'cumgood':0, 'cumbad':0, 'ks':0, 'lift':np.nan}, index=np.arange(1)),
        df_kslift
    ], ignore_index=True)
    
    score_list=df_dum['score_rounded'].sort_values(ascending=True).tolist()
    df_kslift['score']=[np.nan]+score_list

    # Collapse ties to the threshold (block-end) cumulative: within each equal-score
    # block use the cumulative reached at the END of the block, so KS is measured only
    # at attainable cut-off values (a cut-off can sit at a score, never mid-tie). This
    # makes the displayed KS — headline metric and the KS-separation plot — the
    # decision-relevant, order-independent statistic, identical to the bootstrap-CI KS
    # estimate and the optimal-cut-off KS. Only cumgood/cumbad/ks change; the per-row
    # good/bad/score/group columns the performance-projection table uses are untouched.
    _row = df_kslift['score'].notna()
    df_kslift.loc[_row, 'cumgood'] = df_kslift.loc[_row].groupby('score')['cumgood'].transform('max')
    df_kslift.loc[_row, 'cumbad']  = df_kslift.loc[_row].groupby('score')['cumbad'].transform('max')
    df_kslift.loc[_row, 'ks'] = (df_kslift.loc[_row, 'cumbad'] - df_kslift.loc[_row, 'cumgood']).abs()

    good_scores=df_dum[df_dum[target]==0]['score_rounded']
    bad_scores=df_dum[df_dum[target]==1]['score_rounded']
    # Single canonical optimal cut-off (maximises KS), shared by the histogram,
    # the static approval chart and the interactive dashboard.
    cutoff_metrics, cutoff_score = _cutoff_metrics(df_dum['score_rounded'], df_dum[target])

    # ── Headline metrics ─────────────────────────────────────────────────────
    logit_roc_auc = roc_auc_score(y_dum, -1*df_dum['score_rounded'])
    fpr, tpr, thresholds = roc_curve(y_dum, -1*df_dum['score_rounded'])
    max_ks=(100*df_kslift['ks']).max()
    m1, m2, m3 = st.columns(3)
    m1.metric(label="KS-score", value=round(max_ks, 2))
    m2.metric(label="AUC ROC", value=round(logit_roc_auc, 2))
    m3.metric(label="Gini", value=round(100*(2*logit_roc_auc-1.0), 2))

    # ── Bootstrap confidence intervals (stratified BCa) ──────────────────────
    # Sampling uncertainty around the headline metrics. Stratified resampling +
    # BCa intervals (bootstrap.py), evaluated on the same sample as the point
    # estimates above so each interval brackets its headline value.
    st.markdown('**Discrimination with uncertainty — bootstrap confidence intervals**')
    st.caption(f'{int(round(ci_level*100))}% BCa (bias-corrected & accelerated) intervals from '
               f'{int(n_boot):,} stratified bootstrap resamples — the sampling uncertainty of '
               'KS, AUC and Gini. Gini = 2·AUC − 1, so its interval is the exact image of the '
               'AUC interval.')
    ci = bootstrap.confidence_intervals(
        df_dum['score_rounded'].to_numpy(), df_dum[target].to_numpy(),
        n_boot=int(n_boot), ci_level=float(ci_level),
        progress=lambda it: stqdm(it, desc='Bootstrapping CIs'))
    ci_table, ci_fig = _ci_panel(ci)
    viz.capture('4_bootstrap_confidence_intervals', ci_fig)
    c_tab, c_plot = st.columns([1.0, 1.15])
    c_tab.table(viz.style_table(ci_table))
    _ci_buf = BytesIO(); ci_fig.savefig(_ci_buf, format='png'); c_plot.image(_ci_buf, width='stretch')

    # Numeric CI table for the Excel report (one sheet). Scale is encoded in the
    # metric name so the workbook is self-documenting; separate lower/upper
    # columns keep the values numeric (the in-app table is a string render).
    _lvl = int(round(ci_level * 100))
    _scale = {'KS': 'KS (0–100)', 'AUC ROC': 'AUC ROC (0–1)', 'Gini': 'Gini (0–100)'}
    cm_ = ci['metrics']
    df_ci = pd.DataFrame({
        'Metric': [_scale.get(m, m) for m in cm_['Metric']],
        'Estimate': cm_['estimate'].round(2).to_numpy(),
        'CI lower': cm_['ci_low'].round(2).to_numpy(),
        'CI upper': cm_['ci_high'].round(2).to_numpy(),
        'Bootstrap SE': cm_['se'].round(2).to_numpy(),
        'Method': f"{_lvl}% BCa · stratified · {int(n_boot):,} resamples · "
                  f"n={ci['n_good'] + ci['n_bad']:,}",
    })

    # ── Score distribution (full width, compact) ─────────────────────────────
    fig, ax=plt.subplots(figsize=(11,3.4))
    dist_df=pd.DataFrame({
        'Credit score': df_dum['score_rounded'].to_numpy(),
        'Outcome': np.where(df_dum[target].to_numpy()==0, 'Good (repaid)', 'Bad (default)'),
    })
    bins=np.histogram_bin_edges(df_dum['score_rounded'], bins=40)
    sns.histplot(data=dist_df, x='Credit score', hue='Outcome', bins=bins, ax=ax,
                 multiple='dodge', shrink=0.85, alpha=1.0, edgecolor=viz.BG, linewidth=0.25,
                 hue_order=['Good (repaid)', 'Bad (default)'],
                 palette={'Good (repaid)': viz.GOOD, 'Bad (default)': viz.BAD})
    ax.axvline(cutoff_score, color=viz.NAVY, linestyle='--', linewidth=viz.LW, zorder=5)
    ax.annotate(f'Cut-off {cutoff_score:.0f}',
                xy=(cutoff_score, ax.get_ylim()[1]*0.95), xytext=(6, 0), textcoords='offset points',
                color=viz.NAVY, fontsize=8, fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.25', fc=viz.PANEL, ec=viz.NAVY, lw=0.9))
    viz.title(ax, 'Score Distribution by Outcome',
              'Where good and bad applicants fall along the score')
    ax.set_xlabel('Credit score'); ax.set_ylabel('Applicants')
    sns.move_legend(ax, 'upper right', title='Outcome', title_fontsize=8.5)
    sns.despine(ax=ax)
    fig.tight_layout()
    viz.capture('4_score_distribution', fig)
    buf=BytesIO(); fig.savefig(buf, format='png'); st.image(buf, width='stretch')

    # ── KS and ROC, side by side ─────────────────────────────────────────────
    df_ks = df_kslift
    plist = ["eva_p"+i+'(df_'+i+',title)' for i in plot_type]
    subplot_nrows = int(np.ceil(len(plist)/2))
    subplot_ncols = int(np.ceil(len(plist)/subplot_nrows))

    fig_ks = plt.figure(figsize=(5.6,4.6))
    for i in np.arange(len(plist)):
        plt.subplot(subplot_nrows,subplot_ncols,i+1)
        eval(plist[i])
    fig_ks.tight_layout()
    viz.capture('5_scorecard_ks', fig_ks)
    ks_buf=BytesIO(); fig_ks.savefig(ks_buf, format='png')

    fig_roc, ax=plt.subplots(figsize=(5.6,4.6))
    ax.plot(fpr, tpr, color=viz.TEAL, lw=viz.LW, label=f'Model · AUC = {logit_roc_auc:.2f}')
    ax.fill_between(fpr, tpr, color=viz.TEAL, alpha=0.12, lw=0)
    ax.plot([0,1], [0,1], color=viz.SLATE, ls='--', lw=viz.LW_REF, label='Random (0.50)')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.01]); ax.set_aspect('equal')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    viz.title(ax, 'ROC Curve', f'Gini = {100*(2*logit_roc_auc-1):.2f}')
    ax.legend(loc='lower right')
    sns.despine(ax=ax)
    fig_roc.tight_layout(); fig_roc.savefig('Log_ROC')
    viz.capture('6_scorecard_roc', fig_roc)
    roc_buf=BytesIO(); fig_roc.savefig(roc_buf, format='png')

    c_ks, c_roc = st.columns(2)
    c_ks.image(ks_buf, width='stretch')
    c_roc.image(roc_buf, width='stretch')
    
    df_ks['score'] = df_ks['score'].bfill()
    df_ks['score_prev']=df_ks['score'].astype(int)
    df_ks['score_next']=df_ks['score'].astype(int)+1
    
    df_ppt=pd.DataFrame(data={'cutoff_score': df_ks['score_prev'].sort_values(ascending=False).unique().tolist()})

    df_ppt['approval rate']=0.0
    for score in df_ks['score_prev'].sort_values(ascending=False).unique().tolist():
        df_ppt.loc[df_ppt['cutoff_score']==score, 'approval rate']=df_ks[df_ks['score']>score]['group'].count()/df_ks['group'].count()

    df_ppt['marginal odds ratio']=np.exp((df_ppt['cutoff_score']-offset)/factor)
    df_ppt['marginal good rate']=df_ppt['marginal odds ratio']/(1+df_ppt['marginal odds ratio'])
    df_ppt['good rate for total accepted']=0.0
    for score in df_ks['score_prev'].sort_values(ascending=False).unique().tolist():
        df_ppt.loc[df_ppt['cutoff_score']==score, 'good rate for total accepted']=df_ks[(df_ks['score']>=score)&(df_ks['good']==1)]['group'].count()/df_ks[df_ks['score']>=score]['group'].count()

    df_ppt['odds for total accepted']=df_ppt['good rate for total accepted']/(1-df_ppt['good rate for total accepted'])
    df_ppt['good rate for total rejected']=0.0
    for score in df_ks['score_prev'].sort_values(ascending=False).unique().tolist():
        df_ppt.loc[df_ppt['cutoff_score']==score, 'good rate for total rejected']=df_ks[(df_ks['score']<=score)&(df_ks['good']==1)]['group'].count()/df_ks[df_ks['score']<=score]['group'].count()

    df_ppt.loc[df_ppt['good rate for total rejected'].isna()==True, 'good rate for total rejected']=0
    df_ppt['odds for total rejected']=df_ppt['good rate for total rejected']/(1-df_ppt['good rate for total rejected'])

    # ── Approval-strategy chart (static; kept for the Excel report and visuals ZIP)
    ppt_sorted=df_ppt.sort_values('cutoff_score')
    xs=ppt_sorted['cutoff_score'].to_numpy()
    appr=ppt_sorted['approval rate'].to_numpy()*100
    goodacc=ppt_sorted['good rate for total accepted'].to_numpy()*100
    appr_at=float(np.interp(cutoff_score, xs, appr))
    good_at=float(np.interp(cutoff_score, xs, goodacc))

    fig, ax=plt.subplots(figsize=(11,3.6))
    ax.fill_between(xs, appr, color=viz.TEAL, alpha=0.13, lw=0)
    ax.plot(xs, appr, color=viz.TEAL, lw=viz.LW, label='Approval rate (% approved)')
    ax.plot(xs, goodacc, color=viz.GOOD, lw=viz.LW, label='Good rate of accepted (quality)')
    ax.axvline(cutoff_score, color=viz.NAVY, ls='--', lw=viz.LW, zorder=4,
               label=f'Recommended cut-off ({cutoff_score:.0f})')
    ax.scatter([cutoff_score, cutoff_score], [appr_at, good_at], s=viz.MS, color=viz.NAVY,
               edgecolor=viz.BG, linewidth=1.1, zorder=5)
    ax.set_xlabel('Cut-off score  (approve applicants scoring above the cut-off)')
    ax.set_ylabel('Share of applicants')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
    viz.title(ax, 'Approval Strategy by Score Cut-off',
              'A higher cut-off means fewer approvals but better quality of the accepted book')
    ax.legend(loc='lower left')
    sns.despine(ax=ax)
    fig.tight_layout()
    viz.capture('7_approval_strategy_ppt', fig)
    plt.close(fig)

    # ── In-app: interactive Plotly decision dashboard (replaces the static plot) ─
    st.markdown('**Approval strategy — interactive cut-off dashboard**')
    st.caption('Hover any cut-off to read approval / good / default / KS (2 dp) and the '
               'absolute difference (percentage points) from the optimal (max-KS) cut-off — '
               'green when favourable, red when adverse — for flexible policy selection.')
    dash_fig = _approval_dashboard(cutoff_metrics, cutoff_score)
    st.plotly_chart(dash_fig, width='stretch', key='approval_dash')

    # ── Interpretable WoE scorecard ──────────────────────────────────────────
    # One coefficient per feature; every category of every selected feature is
    # listed with its WoE, dataset share and points = round(-factor * coef * WoE).
    # The NaN category is naturally included (it is just another category).
    sc_rows = [{'Feature': 'Intercept', 'Category': '', 'WoE': np.nan,
                'Share (%)': np.nan, 'Score': intercept_points}]
    for i, col in enumerate(feat_cols):
        feat = col[:-4] if col.endswith('_cat') else col
        m = woe_map.get(col) if woe_map is not None else None
        if m is None:
            continue
        has_nan = False
        for _, r in m.iterrows():
            cat = str(r['Category'])
            w = float(r['WoE'])
            is_nan = (cat == 'NaN')
            has_nan = has_nan or is_nan
            # Missing values are neutral: the NaN category always scores 0 points
            # (its fair WoE and share are still shown).
            pts = 0.0 if is_nan else float(np.round(-factor * coefs[i] * w))
            sc_rows.append({'Feature': feat, 'Category': cat,
                            'WoE': round(w, 4), 'Share (%)': r['Share (%)'], 'Score': pts})
        if not has_nan:   # always surface a NaN row (neutral) for every field
            sc_rows.append({'Feature': feat, 'Category': 'NaN', 'WoE': np.nan,
                            'Share (%)': 0.0, 'Score': 0.0})

    df_scorecard = pd.DataFrame(sc_rows, columns=['Feature', 'Category', 'WoE', 'Share (%)', 'Score'])
    # Intercept always first, then grouped by field, highest score first.
    df_scorecard['_o'] = np.where(df_scorecard['Feature'] == 'Intercept', 0, 1)
    df_scorecard = (df_scorecard.sort_values(['_o', 'Feature', 'Score'], ascending=[True, True, False])
                    .drop(columns='_o').reset_index(drop=True))

    with pd.option_context('display.max_rows', None,):
        st.write('Scorecard:')
        st.table(viz.style_table(df_scorecard))

    return df_ppt, df_scorecard, dash_fig, df_ci

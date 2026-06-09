"""Scorecard Inference & Monitoring — a second Streamlit app (run with
``streamlit run inference.py``).

Upload a published scorecard (an Excel file with a **"Scorecard"** sheet in the
exact format the build app produces) and a **new** dataset from the same client.
The app:
  1. scores every applicant (adds a ``Score`` column),
  2. on the new file (if it carries the target) computes KS / AUC / Gini with
     bootstrap (BCa) confidence intervals, the score distribution and the
     Performance Projection Table — identical machinery to the build app,
  3. if the **old** training/reference dataset is also uploaded, investigates
     **data drift** (score PSI + per-characteristic CSI) and **concept drift**
     (bad-rate-by-score-band and KS/AUC/Gini old vs new).
"""
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import roc_auc_score

import bootstrap
import inference_engine as ie
import scorecard_ppt
import viz
from scoring import (_approval_dashboard, _cutoff_metrics, _ci_panel, _style_ppt,
                     build_ppt, score_distribution_fig)

viz.setup()


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def _read_table(upload):
    name = upload.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(upload)
    return pd.read_excel(upload)


def _png(fig):
    buf = BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    return buf.getvalue()


def _metric_card(ci):
    m = ci['metrics'].set_index('Metric')
    c1, c2, c3 = st.columns(3)
    c1.metric('KS-score', round(float(m.loc['KS', 'estimate']), 2))
    c2.metric('AUC ROC', round(float(m.loc['AUC ROC', 'estimate']), 2))
    c3.metric('Gini', round(float(m.loc['Gini', 'estimate']), 2))


def _psi_bar(psi_table, title, subtitle):
    """Reference vs current population share per band (PSI visual)."""
    fig, ax = plt.subplots(figsize=(11, 3.4))
    x = np.arange(len(psi_table)); w = 0.4
    ax.bar(x - w/2, psi_table['reference %'], w, color=viz.SLATE, label='Reference (old)')
    ax.bar(x + w/2, psi_table['current %'], w, color=viz.TEAL, label='Current (new)')
    ax.set_xticks(x); ax.set_xticklabels(psi_table['band'], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('% of population'); viz.title(ax, title, subtitle)
    ax.legend(loc='upper right'); sns.despine(ax=ax); fig.tight_layout()
    return fig


def _bad_rate_fig(br):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    x = np.arange(len(br))
    ax.plot(x, br['reference bad rate']*100, 'o-', color=viz.SLATE, lw=viz.LW, label='Reference (old)')
    ax.plot(x, br['current bad rate']*100, 'o-', color=viz.BAD, lw=viz.LW, label='Current (new)')
    ax.set_xticks(x); ax.set_xticklabels(br['score band'], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Bad rate (%)')
    viz.title(ax, 'Concept drift — bad rate by score band',
              'A band-wise shift in bad rate (population stable) means the score↔risk link moved')
    ax.legend(loc='upper right'); sns.despine(ax=ax); fig.tight_layout()
    return fig


def _inference_xlsx(scored_df, df_ppt, df_ci):
    """Excel deliverable: scored data + (optional) PPT and CI sheets, styled like
    the build app's report, with the optimal (max-KS) PPT row highlighted."""
    out = BytesIO()
    writer = pd.ExcelWriter(out, engine='xlsxwriter')
    wb = writer.book
    used = set()
    sheets = {'Scored data': scored_df}
    if df_ppt is not None:
        sheets['PPT'] = df_ppt.round(4)
    if df_ci is not None:
        sheets['Confidence intervals'] = df_ci
    for sheetname, df in sheets.items():
        name = scorecard_ppt._safe_sheet_name(sheetname, used)
        df.to_excel(writer, index=False, header=False, startrow=1, sheet_name=name)
        ws = writer.sheets[name]
        ws.set_tab_color(viz.TEAL)
        scorecard_ppt._style_sheet(wb, ws, df, highlight_max_col='KS' if sheetname == 'PPT' else None)
    writer.close()
    return out.getvalue()


def _metrics_row(scores, y):
    auc = roc_auc_score(y, -np.asarray(scores, dtype=float))
    _, ks = bootstrap._metrics(np.asarray(scores, float)[y == 0], np.asarray(scores, float)[y == 1])
    return {'KS': ks*100, 'AUC ROC': auc, 'Gini': (2*auc-1)*100}


# ──────────────────────────────────────────────────────────────────────────
# Page chrome (same theme as the build app)
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Scorecard Inference & Monitoring', layout='wide',
                   initial_sidebar_state='expanded')
st.markdown(viz.app_css(), unsafe_allow_html=True)

with st.sidebar.header('1. Type your project name'):
    project_name = st.sidebar.text_input('Project name')
with st.sidebar.header('2. Type the exact target name'):
    target = st.sidebar.text_input('Target name').strip()

with st.sidebar.header('3. Upload the scorecard'):
    sc_file = st.sidebar.file_uploader('Excel with a "Scorecard" sheet', type=['xlsx', 'xls'])
with st.sidebar.header('4. Upload the new dataset'):
    new_file = st.sidebar.file_uploader('New data to score (CSV / Excel)', type=['csv', 'xls', 'xlsx'])
with st.sidebar.header('5. Reference dataset for drift (optional)'):
    old_file = st.sidebar.file_uploader('Old / training data (CSV / Excel)', type=['csv', 'xls', 'xlsx'])

with st.sidebar.subheader('6. Scoring parameters (match the build)'):
    target_score = st.sidebar.slider('Target score', 300, 600, 450, 50)
    target_odds = st.sidebar.slider('Target odds', 0.5, 2.0, 1.0, 0.5)
    pts_double_odds = st.sidebar.slider('Points to double odds', 10, 100, 80, 10)
with st.sidebar.subheader('7.  Evaluation (bootstrap confidence intervals)'):
    ci_level = st.sidebar.select_slider('Confidence level (%)', options=[90, 95, 99], value=95) / 100.0
    n_boot = st.sidebar.select_slider('Bootstrap resamples', options=[500, 1000, 2000, 3000, 5000], value=2000)

factor = pts_double_odds / np.log(2)
offset = target_score - factor * np.log(target_odds)

_label = f' · <b>{project_name}</b>' if project_name else ''
st.markdown(f"""
<div class="hero">
  <h1>Scorecard Inference &amp; Monitoring</h1>
  <p>Deploy a published scorecard on fresh data — score every applicant, re-measure KS / AUC / Gini with
  bootstrap intervals, and investigate data &amp; concept drift against the training sample{_label}.</p>
  <span class="pill">Automatic scoring</span>&nbsp;
  <span class="pill">KS · AUC · Gini + CIs</span>&nbsp;
  <span class="pill">PSI / CSI drift</span>&nbsp;
  <span class="pill">Concept drift</span>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Render (also used to re-draw cached results on a download-click rerun)
# ──────────────────────────────────────────────────────────────────────────
def _render(res):
    st.markdown('---')
    st.success('Inference complete — **downloading will not recompute the app**.')
    st.download_button('Download scored data (Excel)', data=res['xlsx'],
                       file_name=res['xlsx_name'], key='dl_inf')

    st.subheader('1. Scorecard')
    st.caption(f"{res['n_features']} characteristic(s). "
               + (f"⚠️ {len(res['missing'])} not found in the new data (scored neutrally): "
                  f"{', '.join(res['missing'])}." if res['missing'] else 'All characteristics resolved.'))
    st.table(viz.style_table(res['scorecard']))

    st.subheader('2. Scored new dataset (Score column appended)')
    st.dataframe(res['scored_head'], width='stretch')

    if res['ci'] is not None:
        st.subheader('3. Discrimination on the new data')
        _metric_card(res['ci'])
        st.caption('Bootstrap (BCa) confidence intervals quantify the sampling uncertainty.')
        c1, c2 = st.columns([1.0, 1.15])
        c1.table(viz.style_table(res['ci_table']))
        c2.image(res['ci_png'], width='stretch')

        st.markdown('**Score distribution by outcome (histogram + KDE)**')
        st.image(res['dist_png'], width='stretch')

        st.markdown('**Approval strategy — interactive cut-off dashboard**')
        st.plotly_chart(res['dash_fig'], width='stretch', key='inf_dash')

        st.markdown('**Performance Projection Table** — ★ row = optimal (max-KS) cut-off')
        st.table(_style_ppt(res['ppt'], res['cutoff']))
    else:
        st.subheader('3. Score distribution')
        st.caption('No target column found in the new data — showing the score distribution only.')
        st.image(res['dist_png'], width='stretch')

    if res['drift'] is not None:
        d = res['drift']
        st.subheader('4. Drift analysis vs the reference dataset')
        st.markdown(f"**Data drift — model score PSI = {d['score_psi']:.3f}**  "
                    f"({_psi_label(d['score_psi'])})")
        st.image(d['score_psi_png'], width='stretch')
        st.markdown('**Characteristic Stability Index (CSI) per scorecard characteristic**')
        st.table(viz.style_table(d['csi']))
        if d.get('bad_rate_png') is not None:
            st.markdown('**Concept drift**')
            st.image(d['bad_rate_png'], width='stretch')
            st.table(viz.style_table(d['perf']))

    if st.button('Restart (clear results)', key='inf_restart'):
        st.session_state.pop('inf_results', None)
        st.rerun()


def _psi_label(v):
    return 'stable' if v < 0.1 else ('moderate shift' if v < 0.25 else 'LARGE shift')


# ──────────────────────────────────────────────────────────────────────────
# Main flow
# ──────────────────────────────────────────────────────────────────────────
ready = sc_file is not None and new_file is not None
run = st.button('Run inference', type='primary', disabled=not ready)
if not ready:
    st.info('Upload a **scorecard** (Excel with a "Scorecard" sheet) and a **new dataset** to begin.')

# Re-render cached results on a rerun that is not a fresh run (e.g. a download click).
if not run and 'inf_results' in st.session_state:
    _render(st.session_state['inf_results'])
    st.stop()

if run:
    st.session_state.pop('inf_results', None)

    # ── load scorecard + new data ──
    try:
        scorecard = pd.read_excel(sc_file, sheet_name='Scorecard')
    except Exception as e:
        st.error(f'Could not read a "Scorecard" sheet from the uploaded Excel: {e}')
        st.stop()
    parsed = ie.parse_scorecard(scorecard)
    new_df = _read_table(new_file)

    with st.status('Scoring the new dataset…', expanded=True) as s:
        missing = ie.missing_features(new_df, parsed)
        scores = ie.score_dataframe(new_df, parsed)
        scored_df = new_df.copy(); scored_df['Score'] = scores
        s.update(label='Scoring complete ✓', state='complete')

    has_target = bool(target) and target in new_df.columns
    y = None
    if has_target:
        y = pd.to_numeric(new_df[target], errors='coerce').fillna(0).astype(int).to_numpy()
        both = len(np.unique(y)) == 2
        has_target = both

    # ── metrics / CIs / PPT / distribution (needs the target) ──
    ci = ci_table = ci_png = dash_fig = ppt = None
    cutoff = float(np.median(scores))
    if has_target:
        cm, cutoff = _cutoff_metrics(scores, y)
        with st.status('Bootstrapping confidence intervals…', expanded=True) as s:
            ci = bootstrap.confidence_intervals(scores, y, n_boot=int(n_boot), ci_level=float(ci_level))
            s.update(label='Bootstrap complete ✓', state='complete')
        ci_table, ci_fig = _ci_panel(ci)
        ci_png = _png(ci_fig)
        dash_fig = _approval_dashboard(cm, cutoff)
        ppt = build_ppt(scores, y, offset, factor)
        # CI table for the Excel deliverable
        _scale = {'KS': 'KS (0–100)', 'AUC ROC': 'AUC ROC (0–1)', 'Gini': 'Gini (0–100)'}
        cm_ = ci['metrics']
        df_ci_x = pd.DataFrame({'Metric': [_scale.get(m, m) for m in cm_['Metric']],
                                'Estimate': cm_['estimate'].round(2), 'CI lower': cm_['ci_low'].round(2),
                                'CI upper': cm_['ci_high'].round(2), 'Bootstrap SE': cm_['se'].round(2)})
        dist_png = _png(score_distribution_fig(scores, y, cutoff))
    else:
        # unlabelled new data — plain score histogram
        fig, ax = plt.subplots(figsize=(11, 3.4))
        sns.histplot(scores, bins=40, color=viz.TEAL, edgecolor=viz.BG, ax=ax)
        ax.axvline(cutoff, color=viz.NAVY, ls='--', lw=viz.LW)
        viz.title(ax, 'Score Distribution', 'Distribution of the scored new population')
        ax.set_xlabel('Credit score'); ax.set_ylabel('Applicants'); sns.despine(ax=ax); fig.tight_layout()
        dist_png = _png(fig)
        df_ci_x = None

    # ── drift (needs the old/reference dataset) ──
    drift = None
    if old_file is not None:
        old_df = _read_table(old_file)
        old_scores = ie.score_dataframe(old_df, parsed)
        score_psi, psi_tbl = ie.psi(old_scores, scores, bins=10)
        csi = ie.characteristic_psi(old_df, new_df, parsed)
        drift = {'score_psi': score_psi,
                 'score_psi_png': _png(_psi_bar(psi_tbl, 'Data drift — model score PSI',
                                                f'Population share by score band (PSI = {score_psi:.3f})')),
                 'csi': csi, 'bad_rate_png': None, 'perf': None}
        # concept drift needs the target in BOTH files
        if has_target and target in old_df.columns:
            y_old = pd.to_numeric(old_df[target], errors='coerce').fillna(0).astype(int).to_numpy()
            if len(np.unique(y_old)) == 2:
                br = ie.bad_rate_by_band(old_scores, y_old, scores, y, bins=10)
                if not br.empty:
                    drift['bad_rate_png'] = _png(_bad_rate_fig(br))
                m_old, m_new = _metrics_row(old_scores, y_old), _metrics_row(scores, y)
                drift['perf'] = pd.DataFrame({
                    'Metric': ['KS', 'AUC ROC', 'Gini'],
                    'Reference (old)': [round(m_old['KS'], 2), round(m_old['AUC ROC'], 2), round(m_old['Gini'], 2)],
                    'Current (new)': [round(m_new['KS'], 2), round(m_new['AUC ROC'], 2), round(m_new['Gini'], 2)],
                    'Δ (new − old)': [round(m_new['KS']-m_old['KS'], 2), round(m_new['AUC ROC']-m_old['AUC ROC'], 2),
                                      round(m_new['Gini']-m_old['Gini'], 2)],
                })

    _ts = pd.Timestamp.now().strftime('%d-%m-%Y_%H-%M-%S')
    res = {
        'xlsx': _inference_xlsx(scored_df, ppt, df_ci_x),
        'xlsx_name': f'{project_name or "scorecard"}_SCORED_{_ts}.xlsx',
        'scorecard': scorecard, 'n_features': len(parsed['features']), 'missing': missing,
        'scored_head': scored_df.head(20), 'ci': ci, 'ci_table': ci_table, 'ci_png': ci_png,
        'dist_png': dist_png, 'dash_fig': dash_fig, 'ppt': ppt, 'cutoff': cutoff, 'drift': drift,
    }
    st.session_state['inf_results'] = res
    _render(res)

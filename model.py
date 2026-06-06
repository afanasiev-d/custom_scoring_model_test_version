import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import viz
import optuna
import optuna.visualization as ov
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.exceptions import ConvergenceWarning
from eva import eva_dfkslift, eva_pks

plot_type = ['ks']
title=''

# Hyperparameter search settings.
N_TRIALS  = 60
# Trials run concurrently on a thread pool (Optuna's n_jobs uses ThreadPoolExecutor);
# scikit-learn's saga solver releases the GIL, so fitting genuinely runs in parallel.
N_THREADS = min(8, max(1, (os.cpu_count() or 2) - 2))


def _ks(clf, X, y_true):
    """Kolmogorov–Smirnov statistic of a fitted classifier on (X, y_true)."""
    pred = clf.predict_proba(X)[:, 1]
    d = pd.DataFrame({'label': y_true, 'pred': pred}).sample(frac=1, random_state=0)
    dk = eva_dfkslift(d)
    return round(dk.loc[lambda x: x.ks == max(x.ks), 'ks'].iloc[0], 4)


def _theme_plotly(fig, title_text=None, bar_color=None):
    """Apply the fintech look to an Optuna Plotly figure."""
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Inter, sans-serif', color=viz.SLATE, size=12),
        title=dict(text=title_text if title_text is not None else (fig.layout.title.text or ''),
                   font=dict(color=viz.INK, size=15), x=0.0, xanchor='left'),
        colorway=[viz.TEAL, viz.NAVY, viz.GOOD, viz.GOLD, viz.BAD],
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=60, r=30, t=52, b=48), height=340,
        legend=dict(font=dict(size=10)),
    )
    fig.update_xaxes(showgrid=True, gridcolor=viz.GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=viz.GRID, zeroline=False)
    if bar_color:
        fig.update_traces(marker_color=bar_color)
    return fig


def _show_optuna(study):
    """Render the hyperparameter search as a set of beautiful Plotly charts."""
    st.markdown('**Hyperparameter search — Optuna (TPE sampler)**')
    c1, c2 = st.columns(2)
    try:
        c1.plotly_chart(_theme_plotly(ov.plot_optimization_history(study),
                                      'Optimization history'), width='stretch')
    except Exception:
        c1.info('Optimization history unavailable.')
    try:
        c2.plotly_chart(_theme_plotly(ov.plot_param_importances(study),
                                      'Parameter importance', bar_color=viz.TEAL), width='stretch')
    except Exception:
        c2.info('Parameter importance needs a few more varied trials.')
    try:
        st.plotly_chart(_theme_plotly(ov.plot_parallel_coordinate(study),
                                      'Parallel coordinates of trials'), width='stretch')
    except Exception:
        pass
    try:
        st.plotly_chart(_theme_plotly(ov.plot_slice(study), 'Slice plot per parameter'),
                        width='stretch')
    except Exception:
        pass
    # Contour is only meaningful for the two continuous params (the elastic-net
    # region), and only when enough elastic-net trials exist to interpolate.
    n_l1 = sum('l1_ratio' in t.params for t in study.trials)
    if n_l1 >= 5:
        try:
            fig_c = ov.plot_contour(study, params=['C', 'l1_ratio'])
            fig_c.update_traces(colorscale=[[0.0, '#F1FAFB'], [0.5, viz.TEAL], [1.0, viz.NAVY]],
                                selector=dict(type='contour'))
            st.plotly_chart(_theme_plotly(fig_c, 'Objective landscape · C × l1_ratio (elastic-net)'),
                            width='stretch')
        except Exception:
            pass


def build(df_dum1, target, metric='KS'):
    use_auc = str(metric).upper().startswith('AUC')
    metric_name = 'AUC ROC' if use_auc else 'KS'

    X_dum=df_dum1.loc[:, df_dum1.columns!= target]
    y_dum=df_dum1[target]
    X_train, X_test, y_train, y_test=train_test_split(X_dum, y_dum,  test_size=0.3, random_state=42)
    st.markdown('**Train subdataset**')
    st.write(X_train.head(5))
    st.markdown('**Test subdataset**')
    st.write(X_test.head(5))

    def _metric(clf, X, y_true):
        if use_auc:
            return round(roc_auc_score(y_true, clf.predict_proba(X)[:, 1]), 4)
        return _ks(clf, X, y_true)

    # ── Optuna hyperparameter optimisation (L1 / L2 / elastic-net) ────────────
    def objective(trial):
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        C = trial.suggest_float('C', 1e-4, 10.0, log=True)
        kw = dict(penalty=penalty, C=C, solver='saga', max_iter=500)
        if penalty == 'elasticnet':
            kw['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
        clf = LogisticRegression(**kw)
        clf.fit(X_train, y_train)
        m_tr = _metric(clf, X_train, y_train)
        m_te = _metric(clf, X_test, y_test)
        trial.set_user_attr('metric_train', m_tr)
        trial.set_user_attr('metric_test', m_te)
        # maximise the validation metric while penalising the train/validation gap (overfit)
        return m_te - abs(m_tr - m_te)

    st.caption(f'Optimising hyperparameters with Optuna to maximise **{metric_name}** — '
               f'{N_TRIALS} trials across {N_THREADS} threads…')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_THREADS, show_progress_bar=False)

    best = study.best_params
    def _val(v):
        return f'{v:.4f}' if isinstance(v, float) else str(v)
    mt = study.best_trial.user_attrs.get('metric_test')
    rows = [{'Parameter': k, 'Value': _val(v)} for k, v in best.items()]
    rows.append({'Parameter': f'{metric_name} validation', 'Value': _val(mt) if mt is not None else '—'})
    rows.append({'Parameter': f'objective ({metric_name}-stability)', 'Value': f'{study.best_value:.4f}'})
    best_df = pd.DataFrame(rows)
    st.markdown('**Best hyperparameters**')
    st.table(viz.style_table(best_df))

    _show_optuna(study)

    # ── Refit the best model ─────────────────────────────────────────────────
    kw = dict(penalty=best['penalty'], C=best['C'], solver='saga', max_iter=1000)
    if best.get('l1_ratio') is not None:
        kw['l1_ratio'] = best['l1_ratio']
    lr = LogisticRegression(**kw)
    st.write(lr)
    lr.fit(X_train, y_train)

    label=y_dum
    pred=lr.predict_proba(X_dum)[:,1]
    df = pd.DataFrame({'label':label, 'pred':pred}).sample(frac=1, random_state=0)

    df_ks = eva_dfkslift(df)
    ks_score = round(df_ks.loc[lambda x: x.ks==max(x.ks),'ks'].iloc[0],4)

    plist = ["eva_p"+i+'(df_'+i+',title)' for i in plot_type]
    subplot_nrows = int(np.ceil(len(plist)/2))
    subplot_ncols = int(np.ceil(len(plist)/subplot_nrows))

    logit_roc_auc = roc_auc_score(y_dum, lr.predict_proba(X_dum)[:,1])
    fpr, tpr, thresholds = roc_curve(y_dum, lr.predict_proba(X_dum)[:,1])

    st.markdown('**Model performance on the full sample**')
    perf_left, perf_right = st.columns(2)

    fig_ks = plt.figure(figsize=(5.6,4.6))
    for i in np.arange(len(plist)):
        plt.subplot(subplot_nrows,subplot_ncols,i+1)
        eval(plist[i])
    fig_ks.tight_layout()
    viz.capture('2_model_ks_full_sample', fig_ks)
    perf_left.pyplot(fig_ks, width='stretch')

    fig_roc, ax = plt.subplots(figsize=(5.6,4.6))
    ax.plot(fpr, tpr, color=viz.TEAL, lw=viz.LW,
            label=f'Logistic regression · AUC = {logit_roc_auc:.3f}')
    ax.fill_between(fpr, tpr, color=viz.TEAL, alpha=0.12, lw=0)
    ax.plot([0, 1], [0, 1], color=viz.SLATE, ls='--', lw=viz.LW_REF, label='Random (0.50)')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.01]); ax.set_aspect('equal')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    viz.title(ax, 'ROC Curve', f'Gini = {100*(2*logit_roc_auc-1):.1f}')
    ax.legend(loc='lower right')
    sns.despine(ax=ax)
    fig_roc.tight_layout()
    fig_roc.savefig('Log_ROC')
    viz.capture('3_model_roc_full_sample', fig_roc)
    perf_right.pyplot(fig_roc, width='stretch')

    return lr, X_dum, y_dum

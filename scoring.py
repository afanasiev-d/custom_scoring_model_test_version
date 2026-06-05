
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from io import BytesIO
from sklearn.metrics import roc_auc_score, roc_curve
from eva import eva_dfkslift, eva_pks
import viz

plot_type = ['ks']
title=''

def scoring(df_dum, X_dum, y_dum, target, lr, target_score = 450, target_odds = 1, pts_double_odds = 80):
    
    df_dum['logit']=np.log(lr.predict_proba(X_dum)[:,0]/lr.predict_proba(X_dum)[:,1])
    df_dum['odds'] = np.exp(df_dum['logit'])
    df_dum['probs'] = df_dum['odds'] / (df_dum['odds'] + 1)
    factor = pts_double_odds / np.log(2)
    offset = target_score - factor * np.log(target_odds)
    df_dum['score'] = offset + factor * df_dum['logit']
    
    intercept=offset-factor*lr.intercept_
    intercept_rounded=intercept.round(0)
    coefs=-factor*lr.coef_
    coefs_rounded=coefs.round(0)
    
    df_dum['score_rounded']=df_dum.loc[:, ~df_dum.columns.isin([target,'logit','odds','probs','score'])].dot(coefs_rounded[0])+intercept_rounded
    
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
    optimal_cutoff=df_kslift[df_kslift['ks']==df_kslift['ks'].max()]['group'].tolist()[0]
    
    good_scores=df_dum[df_dum[target]==0]['score_rounded']
    bad_scores=df_dum[df_dum[target]==1]['score_rounded']
    cutoff_score=np.percentile(df_dum['score_rounded'],optimal_cutoff*100)

    # ── Headline metrics ─────────────────────────────────────────────────────
    logit_roc_auc = roc_auc_score(y_dum, -1*df_dum['score_rounded'])
    fpr, tpr, thresholds = roc_curve(y_dum, -1*df_dum['score_rounded'])
    max_ks=(100*df_kslift['ks']).max()
    m1, m2, m3 = st.columns(3)
    m1.metric(label="KS-score", value=round(max_ks, 2))
    m2.metric(label="AUC ROC", value=round(logit_roc_auc, 2))
    m3.metric(label="Gini", value=round(100*(2*logit_roc_auc-1.0), 2))

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
    ax.plot(fpr, tpr, color=viz.TEAL, lw=viz.LW, label=f'Model · AUC = {logit_roc_auc:.3f}')
    ax.fill_between(fpr, tpr, color=viz.TEAL, alpha=0.12, lw=0)
    ax.plot([0,1], [0,1], color=viz.SLATE, ls='--', lw=viz.LW_REF, label='Random (0.50)')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.01]); ax.set_aspect('equal')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    viz.title(ax, 'ROC Curve', f'Gini = {100*(2*logit_roc_auc-1):.1f}')
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

    # ── Approval-strategy chart for the Performance Projection Table ──────────
    st.markdown('**Approval strategy (Performance Projection Table)**')
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
    buf=BytesIO()
    fig.savefig(buf, format='png')
    st.image(buf, width='stretch')

    df_scorecard=pd.DataFrame()

    df_scorecard['Feature']=np.concatenate((['Intercept'], lr.feature_names_in_))
    df_scorecard['Score']=np.concatenate((intercept_rounded, coefs_rounded[0]))

    with pd.option_context('display.max_rows', None,):
        st.write('Scorecard:')
        st.dataframe(df_scorecard.sort_values(by=['Feature']).reset_index(drop=True))
        
    #df_scored=pd.concat([df,df_dum['score_rounded']], axis=1)
        
    return df_ppt, df_scorecard

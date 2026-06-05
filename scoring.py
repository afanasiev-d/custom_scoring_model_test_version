
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

    fig, ax=plt.subplots(figsize=(15,7))
    bins=np.histogram_bin_edges(df_dum['score_rounded'], bins=60)
    sns.histplot(good_scores, bins=bins, ax=ax, color=viz.GOOD, edgecolor=viz.BG,
                 linewidth=0.6, alpha=0.78, label='Good (repaid)')
    sns.histplot(bad_scores, bins=bins, ax=ax, color=viz.BAD, edgecolor=viz.BG,
                 linewidth=0.6, alpha=0.72, label='Bad (default)')
    ax.axvline(cutoff_score, color=viz.NAVY, linestyle='--', linewidth=2, zorder=5)
    ax.annotate(f'Optimal cut-off\n{cutoff_score:.0f}',
                xy=(cutoff_score, ax.get_ylim()[1]*0.94),
                xytext=(8, 0), textcoords='offset points',
                color=viz.NAVY, fontsize=11, fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc=viz.PANEL, ec=viz.NAVY, lw=1.1))
    viz.title(ax, 'Score Distribution by Outcome',
              'Where good and bad applicants fall along the score — clean separation is better')
    ax.set_xlabel('Credit score')
    ax.set_ylabel('Number of applicants')
    ax.legend(title='Outcome', loc='upper right')
    sns.despine(ax=ax)
    fig.tight_layout()
    buf=BytesIO()
    fig.savefig(buf, format='png')
    st.image(buf)
    st.write('Optimal cutoff = ', cutoff_score.round(0))
    
    df_ks = df_kslift
    
    ks_score = round(df_ks.loc[lambda x: x.ks==max(x.ks),'ks'].iloc[0],4)
    plist = ["eva_p"+i+'(df_'+i+',title)' for i in plot_type]
    subplot_nrows = int(np.ceil(len(plist)/2))
    subplot_ncols = int(np.ceil(len(plist)/subplot_nrows))
   
    fig = plt.figure(figsize=(9,7.5))
    for i in np.arange(len(plist)):
        plt.subplot(subplot_nrows,subplot_ncols,i+1)
        eval(plist[i])
    fig.tight_layout()
    buf=BytesIO()
    fig.savefig(buf, format='png')
    st.image(buf)

    logit_roc_auc = roc_auc_score(y_dum, -1*df_dum['score_rounded'])
    fpr, tpr, thresholds = roc_curve(y_dum, -1*df_dum['score_rounded'])
    fig, ax=plt.subplots(figsize=(8,8))
    ax.plot(fpr, tpr, color=viz.TEAL, lw=2.8, solid_capstyle='round',
            label=f'Scorecard model · AUC = {logit_roc_auc:.3f}')
    ax.fill_between(fpr, tpr, color=viz.TEAL, alpha=0.12, lw=0)
    ax.plot([0,1], [0,1], color=viz.SLATE, ls='--', lw=1.6, label='Random (AUC = 0.50)')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.01])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_aspect('equal')
    viz.title(ax, 'ROC Curve', f'Gini = {100*(2*logit_roc_auc-1):.1f}  ·  area under the curve')
    ax.legend(loc='lower right')
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig('Log_ROC')
    buf=BytesIO()
    fig.savefig(buf, format='png')
    st.image(buf)
    
    max_ks=(100*df_kslift['ks']).max()
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="KS-score",value=round(max_ks, 2))
    col2.metric(label="AUC ROC",value=round(logit_roc_auc, 2))
    col3.metric(label="Gini", value=round(100* (2*logit_roc_auc-1.0), 2))
    
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
    
    df_scorecard=pd.DataFrame()

    df_scorecard['Feature']=np.concatenate((['Intercept'], lr.feature_names_in_))
    df_scorecard['Score']=np.concatenate((intercept_rounded, coefs_rounded[0]))

    with pd.option_context('display.max_rows', None,):
        st.write('Scorecard:')
        st.dataframe(df_scorecard.sort_values(by=['Feature']).reset_index(drop=True))
        
    #df_scored=pd.concat([df,df_dum['score_rounded']], axis=1)
        
    return df_ppt, df_scorecard

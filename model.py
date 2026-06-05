import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import viz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import itertools
from itertools import product
from stqdm import stqdm
from eva import eva_dfkslift, eva_pks

plot_type = ['ks']
title=''

def build(df_dum1, target):
    X_dum=df_dum1.loc[:, df_dum1.columns!= target]
    y_dum=df_dum1[target]
    X_train, X_test, y_train, y_test=train_test_split(X_dum, y_dum,  test_size=0.3, random_state=42)
    st.markdown('**Train subdataset**')
    st.write(X_train.head(5))
    st.markdown('**Test subdataset**')
    st.write(X_test.head(5))
    data_grid_search=[]
    grid={'penalty':['l1','l2'], 'C':[0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    params_dict={}
    st.write('Grid search progress:')
    

    for params in stqdm(list(itertools.product(grid['penalty'], grid['C'])), desc='Grid search (logistic regression)'):

        lr_clr = LogisticRegression(penalty=params[0], C=params[1], solver='saga')
        lr_clr.fit(X_train, y_train)

        label_train=y_train
        pred_train=lr_clr.predict_proba(X_train)[:,1]
        df = pd.DataFrame({'label':label_train, 'pred':pred_train}).sample(frac=1, random_state=0)
        df_ks = eva_dfkslift(df)
        ks_score_train = round(df_ks.loc[lambda x: x.ks==max(x.ks),'ks'].iloc[0],4)

        label_test=y_test
        pred_test=lr_clr.predict_proba(X_test)[:,1]
        df = pd.DataFrame({'label':label_test, 'pred':pred_test}).sample(frac=1, random_state=0)
        df_ks = eva_dfkslift(df)
        ks_score_test = round(df_ks.loc[lambda x: x.ks==max(x.ks),'ks'].iloc[0],4)

        data_grid_search.append([params, ks_score_train,ks_score_test, ks_score_test-np.abs(ks_score_train-ks_score_test)])
        params_dict[params]=ks_score_test-np.abs(ks_score_train-ks_score_test)
        

    df_grid_search=pd.DataFrame(data_grid_search, columns=['Parametrs', 'KS_train', 'KS_validation', 'Quality Measure'])
    df_grid_search['Parametrs']=df_grid_search['Parametrs'].astype(str)
    st.table(viz.style_table(df_grid_search))
        
    lr = LogisticRegression(penalty=max(params_dict, key=params_dict.get)[0], C=max(params_dict, key=params_dict.get)[1], solver='saga')
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

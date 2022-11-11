
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import itertools
from itertools import product
from eva import eva_dfkslift, eva_pks

plot_type = ['ks']
title=''

def build(df_dum1, target):
    X_dum=df_dum1.loc[:, df_dum1.columns!= target]
    y_dum=df_dum1[target]
    X_train, X_test, y_train, y_test=train_test_split(X_dum, y_dum,  test_size=0.3, random_state=0)
    data_grid_search=[]
    grid={'penalty':['l1','l2'], 'C':[0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    params_dict={}
    st.write('Grid search progress:')
    

    for params in list(itertools.product(grid['penalty'], grid['C'])):

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
    st.dataframe(df_grid_search)
        
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
  
    fig = plt.figure(figsize=(8,8))
    for i in np.arange(len(plist)):
        plt.subplot(subplot_nrows,subplot_ncols,i+1)
        eval(plist[i])
    
    logit_roc_auc = roc_auc_score(y_dum, lr.predict_proba(X_dum)[:,1])
    fpr, tpr, thresholds = roc_curve(y_dum, lr.predict_proba(X_dum)[:,1])
    fig = plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    
    return lr, X_dum, y_dum

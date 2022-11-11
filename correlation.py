
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import streamlit as st

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
    fig=plt.figure(figsize=(10,10))
    #sn.set(rc={'figure.figsize':(13,13)})
    sn.heatmap(corrMatrix, annot=True, cmap='YlGnBu')

    st.pyplot(fig)
    df_dum=df_dum[new_columns]
    
    return df_dum

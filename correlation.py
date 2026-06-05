
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import streamlit as st
import viz

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

    n=len(corrMatrix.columns)
    annotate=n<=30                       # keep the matrix readable when it grows
    side=float(np.clip(1.1+0.55*n, 6, 20))
    fig, ax=plt.subplots(figsize=(side, side*0.92))
    mask=np.triu(np.ones_like(corrMatrix, dtype=bool), k=1)   # show lower triangle only
    sn.heatmap(corrMatrix, mask=mask, ax=ax, cmap=viz.CORR_CMAP, center=0, vmin=-1, vmax=1,
               annot=annotate, fmt='.2f', annot_kws={'size':8, 'color':viz.INK},
               linewidths=0.6, linecolor=viz.BG, square=True,
               cbar_kws={'shrink':0.55, 'label':'Pearson correlation', 'ticks':[-1,-0.5,0,0.5,1]})
    viz.title(ax, 'Feature Correlation Matrix',
              f'Pairs above |{threshold:.2f}| are dropped to avoid multicollinearity')
    ax.tick_params(left=False, bottom=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()

    st.pyplot(fig)
    df_dum=df_dum[new_columns]

    return df_dum

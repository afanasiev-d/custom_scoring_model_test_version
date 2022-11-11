
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plot_type = ['ks']
title=''

def eva_dfkslift(df, groupnum=None):
    if groupnum is None: groupnum=len(df.index)
    # good bad func
    def n0(x): return sum(x==0)
    def n1(x): return sum(x==1)
    df_kslift = df.sort_values('pred', ascending=False).reset_index(drop=True)\
      .assign(group=lambda x: np.ceil((x.index+1)/(len(x.index)/groupnum)))\
      .groupby('group')['label'].agg([n0,n1])\
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
    # bind 0
    df_kslift=pd.concat([
      pd.DataFrame({'group':0, 'good':0, 'bad':0, 'good_distri':0, 'bad_distri':0, 'badrate':0, 'cumbadrate':np.nan, 'cumgood':0, 'cumbad':0, 'ks':0, 'lift':np.nan}, index=np.arange(1)),
      df_kslift
    ], ignore_index=True)
    # return
    return df_kslift
    
#---------------------------------#

def eva_pks(dfkslift, title):
    dfks = dfkslift.loc[lambda x: x.ks==max(x.ks)].sort_values('group').iloc[0]
    ###### plot ###### 
    # fig, ax = plt.subplots()
    # ks, cumbad, cumgood
    plt.plot(dfkslift.group, dfkslift.ks, 'b-', 
      dfkslift.group, dfkslift.cumgood, 'g', 
      dfkslift.group, dfkslift.cumbad, 'r')
    # ks vline
    plt.plot([dfks['group'], dfks['group']], [0, dfks['ks']], 'r--')
    # set xylabel
    plt.gca().set(title=title+'Kolmlgorov-Smirnov test', 
      xlabel='% of population', ylabel='% of total Good/Bad', 
      xlim=[0,1], ylim=[0,1], aspect='equal')
    # text
    # plt.text(0.5,0.96,'K-S', fontsize=15,horizontalalignment='center')
    plt.text(0.4,0.6,'Bad',horizontalalignment='center')
    plt.text(0.8,0.55,'Good',horizontalalignment='center')
    plt.text(dfks['group'], dfks['ks'], 'KS:'+ str(round(dfks['ks'],4)), horizontalalignment='center',color='b')
    # plt.grid()
    # plt.show()
    # return fig

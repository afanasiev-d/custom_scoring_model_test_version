
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
    import viz
    dfks = dfkslift.loc[lambda x: x.ks==max(x.ks)].sort_values('group').iloc[0]
    ax = plt.gca()
    # cumulative good / bad curves
    ax.plot(dfkslift.group, dfkslift.cumgood, color=viz.GOOD, lw=2.6, label='Cumulative Good', solid_capstyle='round')
    ax.plot(dfkslift.group, dfkslift.cumbad, color=viz.BAD, lw=2.6, label='Cumulative Bad', solid_capstyle='round')
    # K-S separation curve + shaded gap
    ax.plot(dfkslift.group, dfkslift.ks, color=viz.NAVY, lw=2.6, label='K-S separation')
    ax.fill_between(dfkslift.group, dfkslift.cumbad, dfkslift.cumgood, color=viz.NAVY, alpha=0.06, lw=0)
    # max-KS marker
    ax.plot([dfks['group'], dfks['group']], [dfks['cumgood'], dfks['cumbad']],
            color=viz.GOLD, ls='--', lw=2, zorder=4)
    ax.scatter([dfks['group']], [dfks['ks']], s=55, color=viz.GOLD, edgecolor=viz.BG,
               linewidth=1.4, zorder=5)
    ax.annotate(f"KS = {dfks['ks']*100:.1f}",
                xy=(dfks['group'], dfks['ks']),
                xytext=(dfks['group']+0.04, dfks['ks']-0.08),
                color=viz.INK, fontweight='bold', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', fc=viz.PANEL, ec=viz.GOLD, lw=1.2))
    ax.set(xlabel='% of population', ylabel='% of total Good / Bad', xlim=[0,1], ylim=[0,1])
    viz.title(ax, (title or '') + 'Kolmogorov–Smirnov Separation')
    ax.legend(loc='upper left')

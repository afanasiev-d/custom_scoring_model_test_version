
import os
import re
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import pandas as pd
import numpy as np
from optbinning import OptimalBinning
from stqdm import stqdm
import viz

BIN_DECIMALS = 1  # round numerical bin bounds to this many decimals for interpretability

# OptimalBinning fits are run concurrently on a thread pool. The heavy ortools /
# scikit-learn work releases the GIL, so threads give real parallelism without the
# serialization / process-spawn overhead (and the mip solver is thread-safe here).
_N_BIN_WORKERS = min(8, max(1, (os.cpu_count() or 2) - 2))


def _fit_binning(task):
    """Pure worker (no Streamlit): fit one OptimalBinning and return its table + IV.

    ``task`` is ``(feature, x, y, dtype, trend)`` where x/y are plain numpy arrays
    extracted on the main thread (so workers never touch shared DataFrames)."""
    feature, x, y, dtype, trend = task
    try:
        if dtype == 'categorical':
            optb = OptimalBinning(name=feature, dtype='categorical', solver='mip')
        else:
            optb = OptimalBinning(name=feature, dtype='numerical', solver='mip', monotonic_trend=trend)
        optb.fit(x, y)
        bt = optb.binning_table.build()
        return feature, dtype, trend, bt, float(bt['IV'].max())
    except Exception:
        return feature, dtype, trend, None, None


def _round_bin_label(label, ndigits=BIN_DECIMALS):
    """Round every number that appears in an interval label to ``ndigits`` decimals
    (e.g. '(-inf, 10152.47)' -> '(-inf, 10152.5)'). Non-numeric bins are untouched."""
    return re.sub(r'-?\d+\.\d+',
                  lambda m: f'{round(float(m.group()), ndigits):g}',
                  str(label))

def _clean_bin_label(b):
    """Render an optbinning bin as a clean label.

    Categorical bins are arrays of category values; under pandas 3 these are
    ArrowStringArrays whose ``str()`` is an ugly multi-line repr
    (``<ArrowStringArray> [...] Length: n, dtype: str``). Convert those to a
    readable ``"['A', 'B']"`` string and leave scalar/special bins (e.g. the
    numerical interval strings, ``'Special'``, ``'Missing'``) unchanged.
    """
    if isinstance(b, str):
        return b
    try:
        return str([str(v) for v in b])
    except TypeError:
        return str(b)

def feature_selection_palencia(df_num, df_cat, list_numerical_desc_features, list_numerical_asc_features, list_categ_y_better, list_categ_n_better, target, new_predictors_asc=[], new_predictors_desc=[], min_iv=0.01):
    dictionary_feature_stat={}

    # ── Build the task list (categorical first, then numerical asc / desc), with
    #    x/y extracted here so the worker threads never touch shared DataFrames. ──
    Xc = df_cat.loc[:, df_cat.columns != target]
    yc = df_cat[target].values
    tasks = [(f, Xc[f].values, yc, 'categorical', None) for f in Xc.columns.tolist()]

    df_num = pd.concat([df_num, df_cat[target]], axis=1)
    Xn = df_num.loc[:, df_num.columns != target]
    yn = df_num[target].values
    asc_set = set(list_numerical_asc_features) | set(new_predictors_asc)
    desc_set = set(list_numerical_desc_features) | set(new_predictors_desc)
    for f in Xn.columns.tolist():
        if f in asc_set:
            tasks.append((f, Xn[f].values, yn, 'numerical', 'ascending'))
        if f in desc_set:
            tasks.append((f, Xn[f].values, yn, 'numerical', 'descending'))

    # ── Fit every feature in parallel (compute only — no Streamlit calls). ──
    with ThreadPoolExecutor(max_workers=_N_BIN_WORKERS) as ex:
        results = list(stqdm(ex.map(_fit_binning, tasks), total=len(tasks),
                             desc=f'3.1 Binning {len(tasks)} features ({_N_BIN_WORKERS} threads)'))

    # ── Filter on IV and display sequentially in the main thread. ──
    list_categorical_features = []
    list_numerical_features_asc = []
    list_numerical_features_desc = []
    for feature, dtype, trend, df_binning_table, iv in results:
        if df_binning_table is None or iv is None or not (iv > min_iv and iv < 1):
            continue
        df_binning_table['WoE'] = pd.to_numeric(df_binning_table['WoE'])
        df_binning_table.index = df_binning_table.index.map(str)
        if dtype == 'categorical':
            df_binning_table.Bin = df_binning_table.Bin.map(_clean_bin_label)
            list_categorical_features.append(feature)
        else:
            df_binning_table['Bin'] = df_binning_table['Bin'].map(_round_bin_label)
            (list_numerical_features_asc if trend == 'ascending'
             else list_numerical_features_desc).append(feature)
        st.write(feature)
        st.table(viz.style_table(df_binning_table))
        dictionary_feature_stat[feature] = df_binning_table

    list_numerical_features = list_numerical_features_asc + list_numerical_features_desc

    return list_numerical_features, list_categorical_features, list_numerical_features_asc, list_numerical_features_desc, dictionary_feature_stat
        
    
#---------------------------------#

def merging_for_model(df_all, list_numerical_features, list_categorical_features, target, list_numerical_features_asc, list_numerical_features_desc):
    list_categorical_features_spec_nan=[]
    for spec_cat_feat in list(set(list_categorical_features).intersection(['ELJCOUNTY1', 'ELJCOUNTY2', 'MBELJFILINGNAME1', 'MBELJFILINGNAME2', 'City_App', 'State_App'])):
        list_categorical_features.remove(spec_cat_feat)
        list_categorical_features_spec_nan.append(spec_cat_feat)
    df=pd.DataFrame()
    df[target]=df_all[target]
    for feat in stqdm(list_categorical_features, desc='Building model bins (categorical)'):
        df[feat]=df_all[feat]
        X = df.loc[:, df.columns!= target]
        y = df[target]
        x=X[feat].values
        optb = OptimalBinning(name=feat,dtype="categorical",solver="mip")
        optb.fit(x, y)
        binning_table = optb.binning_table
        df[feat+'_cat']=pd.Series(np.nan, index=df.index, dtype=object)
        for index in range(len(binning_table.build())-3):
            df.loc[df[feat].isin(binning_table.build()['Bin'][index]), feat+'_cat']= _clean_bin_label(binning_table.build()['Bin'][index])
        df.drop(feat, inplace=True, axis=1)
        df.loc[df[feat+'_cat'].isna(), feat+'_cat']= 'NaN'

    for feat in stqdm(list_numerical_features_asc, desc='Building model bins (numerical ↑)'):
        df[feat]=df_all[feat]
        X = df.loc[:, df.columns!= target]
        y = df[target]
        x=X[feat].values
        optb = OptimalBinning(name=feat,dtype="numerical",solver="mip", monotonic_trend="ascending")
        optb.fit(x, y)
        binning_table = optb.binning_table
        splits=sorted(set(np.round(optb.splits, BIN_DECIMALS).tolist()))  # interpretable, rounded bounds
        bins=pd.IntervalIndex.from_breaks([-np.inf]+splits+[np.inf])
        df[feat+'_cat']=pd.cut(df[feat], bins)
        df.drop(feat, inplace=True, axis=1)
        df[feat+'_cat']=df[feat+'_cat'].astype('string')
        df.loc[df[feat+'_cat'].isna(), feat+'_cat']= 'NaN'

    for feat in stqdm(list_numerical_features_desc, desc='Building model bins (numerical ↓)'):
        df[feat]=df_all[feat]
        X = df.loc[:, df.columns!= target]
        y = df[target]
        x=X[feat].values
        optb = OptimalBinning(name=feat,dtype="numerical",solver="mip", monotonic_trend="descending")
        optb.fit(x, y)
        binning_table = optb.binning_table
        splits=sorted(set(np.round(optb.splits, BIN_DECIMALS).tolist()))  # interpretable, rounded bounds
        bins=pd.IntervalIndex.from_breaks([-np.inf]+splits+[np.inf])
        df[feat+'_cat']=pd.cut(df[feat], bins)
        df.drop(feat, inplace=True, axis=1)
        df[feat+'_cat']=df[feat+'_cat'].astype('string')
        df.loc[df[feat+'_cat'].isna(), feat+'_cat']= 'NaN'

    for feat in list_categorical_features_spec_nan:
        df[feat]=df_all[feat]
        df[feat+'_cat']=pd.Series(np.nan, index=df.index, dtype=object)
        df.loc[~df[feat].isna(), feat+'_cat']= 'not NaN'    
        df.loc[df[feat].isna(), feat+'_cat']= 'NaN' 
        df.drop(feat, inplace=True, axis=1)
        
    return df

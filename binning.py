
import re
import streamlit as st
import pandas as pd
import numpy as np
from optbinning import OptimalBinning
from stqdm import stqdm
import viz

BIN_DECIMALS = 1  # round numerical bin bounds to this many decimals for interpretability


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
    X = df_cat.loc[:, df_cat.columns!= target]
    y = df_cat[target]
    list_categorical_features=[]
    for feature in stqdm(X.columns.tolist(), desc='3.1 Binning categorical features'):
        try:
            x=X[feature].values
            optb = OptimalBinning(name=feature,dtype="categorical",solver="mip")
            optb.fit(x, y)
            binning_table = optb.binning_table
            df_binning_table = binning_table.build()
            df_binning_table['WoE']=pd.to_numeric(df_binning_table['WoE'])
            df_binning_table.index=df_binning_table.index.map(str)
            df_binning_table.Bin=df_binning_table.Bin.map(_clean_bin_label)

            if (df_binning_table['IV'].max()>min_iv) & (df_binning_table['IV'].max()<1):
                st.write(feature)
                st.dataframe(viz.style_table(df_binning_table), hide_index=True)
                list_categorical_features.append(feature)
                dictionary_feature_stat[feature]=df_binning_table
        except:
            pass

    df_num=pd.concat([df_num,df_cat[target]], axis=1)
    X = df_num.loc[:, df_num.columns!= target]
    y = df_num[target]
    list_numerical_features_asc=[]
    list_numerical_features_desc=[]
    for feature in stqdm(X.columns.tolist(), desc='3.1 Binning numerical features'):
        try:
            if feature in list_numerical_asc_features+new_predictors_asc:
                x=X[feature].values
                optb = OptimalBinning(name=feature,dtype="numerical",solver="mip", monotonic_trend="ascending")
                optb.fit(x, y)
                binning_table = optb.binning_table
                df_binning_table = binning_table.build()
                df_binning_table['WoE']=pd.to_numeric(df_binning_table['WoE'])
                df_binning_table.index=df_binning_table.index.map(str)
                df_binning_table['Bin']=df_binning_table['Bin'].map(_round_bin_label)

                if (df_binning_table['IV'].max()>min_iv) & (df_binning_table['IV'].max()<1):
                    st.write(feature)
                    st.dataframe(viz.style_table(df_binning_table), hide_index=True)
                    list_numerical_features_asc.append(feature)
                    dictionary_feature_stat[feature]=df_binning_table

            if feature in list_numerical_desc_features+new_predictors_desc:
                x=X[feature].values
                optb = OptimalBinning(name=feature,dtype="numerical",solver="mip", monotonic_trend="descending")
                optb.fit(x, y)
                binning_table = optb.binning_table
                df_binning_table = binning_table.build()
                df_binning_table['WoE']=pd.to_numeric(df_binning_table['WoE'])
                df_binning_table.index=df_binning_table.index.map(str)
                df_binning_table['Bin']=df_binning_table['Bin'].map(_round_bin_label)

                if (df_binning_table['IV'].max()>min_iv) & (df_binning_table['IV'].max()<1):
                    st.write(feature)
                    st.dataframe(viz.style_table(df_binning_table), hide_index=True)
                    list_numerical_features_desc.append(feature)
                    dictionary_feature_stat[feature]=df_binning_table
        except:
            pass
    list_numerical_features=list_numerical_features_asc+list_numerical_features_desc
        
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

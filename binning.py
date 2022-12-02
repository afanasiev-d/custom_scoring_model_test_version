
import streamlit as st
import pandas as pd
import numpy as np
from optbinning import OptimalBinning

def feature_selection_palencia(df_num, df_cat, list_numerical_desc_features, list_numerical_asc_features, list_categ_y_better, list_categ_n_better, target, new_predictors_asc=[], new_predictors_desc=[], min_iv=0.01):
    dictionary_feature_stat={}
    X = df_cat.loc[:, df_cat.columns!= target]
    y = df_cat[target]
    list_categorical_features=[]
    for feature in X.columns.tolist():
        try:
            x=X[feature].values
            optb = OptimalBinning(name=feature,dtype="categorical",solver="cp")
            optb.fit(x, y)
            binning_table = optb.binning_table
            df_binning_table = binning_table.build()
            df_binning_table['WoE']=pd.to_numeric(df_binning_table['WoE'])
            df_binning_table.index=df_binning_table.index.map(str)
            df_binning_table.Bin=df_binning_table.Bin.map(str)

            if (df_binning_table['IV'].max()>min_iv) & (df_binning_table['IV'].max()<1):
                st.write(feature)
                st.dataframe(df_binning_table)
                list_categorical_features.append(feature)
                dictionary_feature_stat[feature]=binning_table.build()
        except:
            pass

    df_num=pd.concat([df_num,df_cat[target]], axis=1)
    X = df_num.loc[:, df_num.columns!= target]
    y = df_num[target]
    list_numerical_features_asc=[]
    list_numerical_features_desc=[]
    for feature in X.columns.tolist():
        try:
            if feature in list_numerical_asc_features+new_predictors_asc:
                x=X[feature].values
                optb = OptimalBinning(name=feature,dtype="numerical",solver="cp", monotonic_trend="ascending")
                optb.fit(x, y)
                binning_table = optb.binning_table
                df_binning_table = binning_table.build()
                df_binning_table['WoE']=pd.to_numeric(df_binning_table['WoE'])
                df_binning_table.index=df_binning_table.index.map(str)

                if (df_binning_table['IV'].max()>min_iv) & (df_binning_table['IV'].max()<1):
                    st.write(feature)
                    st.dataframe(df_binning_table)
                    list_numerical_features_asc.append(feature)
                    dictionary_feature_stat[feature]=binning_table.build()
                    
            if feature in list_numerical_desc_features+new_predictors_desc:
                x=X[feature].values
                optb = OptimalBinning(name=feature,dtype="numerical",solver="cp", monotonic_trend="descending")
                optb.fit(x, y)
                binning_table = optb.binning_table
                df_binning_table = binning_table.build()
                df_binning_table['WoE']=pd.to_numeric(df_binning_table['WoE'])
                df_binning_table.index=df_binning_table.index.map(str)

                if (df_binning_table['IV'].max()>min_iv) & (df_binning_table['IV'].max()<1):
                    st.write(feature)
                    st.dataframe(df_binning_table)
                    list_numerical_features_desc.append(feature)
                    dictionary_feature_stat[feature]=binning_table.build()
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
    for feat in list_categorical_features:
        df[feat]=df_all[feat]
        X = df.loc[:, df.columns!= target]
        y = df[target]
        x=X[feat].values
        optb = OptimalBinning(name=feat,dtype="categorical",solver="mip")
        optb.fit(x, y)
        binning_table = optb.binning_table
        df[feat+'_cat']=np.nan
        for index in range(len(binning_table.build())-3): 
            df.loc[df[feat].isin(binning_table.build()['Bin'][index]), feat+'_cat']= str(binning_table.build()['Bin'][index])
        df.drop(feat, inplace=True, axis=1)
        df.loc[df[feat+'_cat'].isna(), feat+'_cat']= 'NaN'

    for feat in list_numerical_features_asc:
        df[feat]=df_all[feat]
        X = df.loc[:, df.columns!= target]
        y = df[target]
        x=X[feat].values
        optb = OptimalBinning(name=feat,dtype="numerical",solver="cp", monotonic_trend="ascending")
        optb.fit(x, y)
        binning_table = optb.binning_table
        bins=pd.IntervalIndex.from_breaks([-np.inf]+optb.splits.tolist()+[np.inf])
        df[feat+'_cat']=pd.cut(df[feat], bins)
        df.drop(feat, inplace=True, axis=1)
        df[feat+'_cat']=df[feat+'_cat'].astype('string')
        df.loc[df[feat+'_cat'].isna(), feat+'_cat']= 'NaN'
        
    for feat in list_numerical_features_desc:
        df[feat]=df_all[feat]
        X = df.loc[:, df.columns!= target]
        y = df[target]
        x=X[feat].values
        optb = OptimalBinning(name=feat,dtype="numerical",solver="cp", monotonic_trend="descending")
        optb.fit(x, y)
        binning_table = optb.binning_table
        bins=pd.IntervalIndex.from_breaks([-np.inf]+optb.splits.tolist()+[np.inf])
        df[feat+'_cat']=pd.cut(df[feat], bins)
        df.drop(feat, inplace=True, axis=1)
        df[feat+'_cat']=df[feat+'_cat'].astype('string')
        df.loc[df[feat+'_cat'].isna(), feat+'_cat']= 'NaN'

    for feat in list_categorical_features_spec_nan:
        df[feat]=df_all[feat]
        df[feat+'_cat']=np.nan
        df.loc[~df[feat].isna(), feat+'_cat']= 'not NaN'    
        df.loc[df[feat].isna(), feat+'_cat']= 'NaN' 
        df.drop(feat, inplace=True, axis=1)
        
    return df


import numpy as np
import pandas as pd

def encoder(df, target):
    df_dum=df.loc[:, df.columns!= target]#df.select_dtypes(include='object')
    cat_vars=df.loc[:, df.columns!= target].columns#df.select_dtypes(include='object').columns
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(df_dum[var], prefix=var)
        data1=df_dum.join(cat_list)
        df_dum=data1
    cat_vars=df.loc[:, df.columns!= target].columns#df.select_dtypes(include='object').columns
    data_vars=df.loc[:, df.columns!= target].columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_vars]

    #df_dum.drop(df.select_dtypes(include='object').columns, inplace=True, axis=1)
    df_dum.drop(df.loc[:, df.columns!= target].columns, inplace=True, axis=1)
    df_dum[target]=df[target]
    
    temp_cols=df_dum.columns.tolist()
    index=df_dum.columns.get_loc(target)
    new_cols=temp_cols[0:index] + temp_cols[index+1:]+temp_cols[index:index+1]
    df_dum=df_dum[new_cols]
    
    return df_dum

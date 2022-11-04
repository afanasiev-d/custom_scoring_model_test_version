
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.datasets import load_diabetes, load_boston
from optbinning import OptimalBinning
from io import BytesIO
import seaborn as sn
import itertools
from itertools import product
from stqdm import stqdm
from datetime import datetime

dictionary='Full Dictionary.xlsx'
plot_type = ['ks']
title=''
new_predictors=[]
direction='C:/Users/Daniil Afanasiev/Projects/Preprocessing Pipeline'


# Page layout
## Page expands to full width
st.set_page_config(page_title='The Custom Model App',
    layout='wide')

#---------------------------------#

def df_preprocessing(df, sparse_threshold=0.95, target='First_payment_default_Flag'):
    
    info_list=['MBID',
               'Sequence_Number',
               'Application_Number',
               'Application_Type',
               'Renewal_Application_Number',
               'Original_Application_Number',
               'Merchant_Number','MicroBilt_Inquiry_ID',
               'Marketing_Description',
               'Application_Status',
               'Funded_Status',
               'Applicant_Last_Name',
               'Applicant_First_Name',
               'Applicant_Middle_Name',
               'Applicant_Suffix',
               'Applicant_Address_1',
               'Applicant_City',
               'Applicant_State',
               'Applicant_Postal_Code',
               'Applicant_Postal_Code___4',
               'Applicant_ID_Country',
               'Applicant_ID_Type',
               'Applicant_SSN',
               'Applicant_Home_Phone',
               'Applicant_Work_Phone',
               'Applicant_Mobile_Phone',
               'Applicant_Email_Address',
               'Applicant_ABA_Number',
               'Applicant_Account_Number',
               'Applicant_Date_of_Birth',
               'Loan_Number',
               'Loan_Rollover_Refinance_Number',
               'Loan_Status_Date',
               'SSN',
               'LOAN_CODE',
               'year',
               'lyear',
               'Address1_App',
               'Charged_Off_Flag',
               'LastName_App',
               'LeadID',
               'PostalCode_App',
               'FirstName_App',
               'CustSSN_APP',
               'HomePhone_App',
               'City_App',
               'WorkPhone_App',
               'State_App'

    ]

    state_list=[ 'HH_TOTALALLCREDIT_SEVEREDEROG',
     'HIGHCRD_NONMTGCREDIT',
     'INQUIRY_CONSUMERFINANCE',
     'NUM_CONSUMERFINANCE_60DPD',
     'NUM_OTHERNONMTG_NEW',
     'PRCNT_INQUIRIES',
     'LastReportPeriod',
     'DATE',
     'BUSAPPPCTCHGY_U',
     'HIGHPROPRBUSAPPPCTCHGY_S',
     'BUSAPPPLANNEDWAGESPCTCHGY_S',
     'HIRES_RATE_S',
     'JOB_OPENINGS_RATE_S',
     'QUITS_RATE_S',
     'JOB_OPENINGS_PCTCHGY_S',
     'LFPARTRATE_S',
     'lmonth',
     'lyear'
     'REALGDPPERCAPITAPCTCHGY',
     'GDPPERCAPITAPCTCHGY',
     'PERSINCPERCAPITAPCTCHGY',
     'XBUSAPPPCTCHGY_U',
     'XHIGHPROPRBUSAPPPCTCHGY_S',
     'XBUSAPPPLANNEDWAGESPCTCHGY_S',
     'XHIRES_RATE_S',
     'XJOB_OPENINGS_RATE_S',
     'XQUITS_RATE_S',
     'XJOB_OPENINGS_PCTCHGY_S',
     'XLFPARTRATE_S',
     'XREALGDPPERCAPITAPCTCHGY',
     'XGDPPERCAPITAPCTCHGY',
     'XPERSINCPERCAPITAPCTCHGY']



    column_list_without_info=list(set(df.columns.values.tolist())-set(info_list))
    column_list_without_info_and_score=list(set(column_list_without_info)-set(df.filter(regex='score').columns.values.tolist())-set(df.filter(regex='SCORE').columns.values.tolist())-set(df.filter(regex='Score').columns.values.tolist())-set(['p_0','p_1']))
    column_list_without_info_score_and_integrators=list(set(column_list_without_info_and_score)-set(['NAP','NAS','CVI']))
    column_list_without_info_score_integrators_and_states=list(set(column_list_without_info_score_and_integrators)-set(df.filter(regex='_S').columns.values.tolist())-set(df.filter(regex='GDP').columns.values.tolist())-set(state_list))
    column_list_without_info_score_integrators_states_and_cl=list(set(column_list_without_info_score_integrators_and_states)-set(df.filter(regex='CL').columns.values.tolist()))
    column_list_without_info_score_integrators_states_cl_and_dates=list(set(column_list_without_info_score_integrators_states_and_cl)-set(df.select_dtypes(include='datetime64[ns]').columns)-set(df.filter(regex='ELJFILINGDT').columns.values.tolist())-set(df.filter(regex='ELJRLSDT').columns.values.tolist())-set(df.filter(regex='MBBKRLSDT').columns.values.tolist())-set(df.filter(regex='DATE').columns.values.tolist())-set(df.filter(regex='date').columns.values.tolist())-set(df.filter(regex='Date').columns.values.tolist())-set(df.filter(regex='DT').columns.values.tolist()))
    
    df.replace('.', np.NaN, inplace=True)

    df_filling=pd.DataFrame((df[column_list_without_info_score_integrators_states_cl_and_dates].isna().sum()/df[column_list_without_info_score_integrators_states_cl_and_dates].shape[0])>sparse_threshold)
    column_list_of_sparse=df_filling[df_filling[0]==True].index.tolist()
    column_list_without_info_score_integrators_states_cl_dates_and_sparse=list(set(column_list_without_info_score_integrators_states_cl_and_dates)-set(column_list_of_sparse))
    
    df=df[column_list_without_info_score_integrators_states_cl_dates_and_sparse]
    
    df=df.reindex(sorted(df.columns), axis=1)
    
    temp_cols=df.columns.tolist()
    index=df.columns.get_loc(target)
    new_cols=temp_cols[0:index] + temp_cols[index+1:]+temp_cols[index:index+1]
    df=df[new_cols]
    
    return df

#---------------------------------#

def generator_of_predictors_logic(dictionary, list_new_to_desc=[], list_new_to_asc=[]):
    
    df_logic_dict=pd.read_excel(dictionary, index_col=None)

    list_numerical_desc_features=df_logic_dict[((df_logic_dict['Type'].isin(df_logic_dict[df_logic_dict['Type'].str.contains('Num', na=False)].Type.unique().tolist())) | (df_logic_dict['Type']=='Character (#,###.##)'))&(df_logic_dict['Sign']=='>')]['Variable Name (ReNamed)'].tolist()
    list_numerical_asc_features=df_logic_dict[((df_logic_dict['Type'].isin(df_logic_dict[df_logic_dict['Type'].str.contains('Num', na=False)].Type.unique().tolist())) | (df_logic_dict['Type']=='Character (#,###.##)')) &(df_logic_dict['Sign']=='<')]['Variable Name (ReNamed)'].tolist()

    list_categ_y_better=df_logic_dict[(df_logic_dict['Type'].isin(df_logic_dict[df_logic_dict['Type'].str.contains('Char', na=False) & (df_logic_dict['Type']!='Character (#,###.##)')].Type.unique().tolist())) &(df_logic_dict['Sign']=='Y')]['Variable Name (ReNamed)'].tolist()
    list_categ_n_better=df_logic_dict[(df_logic_dict['Type'].isin(df_logic_dict[df_logic_dict['Type'].str.contains('Char', na=False) & (df_logic_dict['Type']!='Character (#,###.##)')].Type.unique().tolist())) &(df_logic_dict['Sign']=='N')]['Variable Name (ReNamed)'].tolist()
    
    list_numerical_desc_features+=list_new_to_desc
    list_numerical_asc_features+=list_new_to_asc
    
    return list_numerical_desc_features, list_numerical_asc_features, list_categ_y_better, list_categ_n_better, df_logic_dict
    
#---------------------------------#

def num_cat_split(df):
    df_num=df.select_dtypes(include=['int64','float64'])
    num_to_cat_feat_list=[]  #create list of features that should be categorical, but have numerical gradations

    for column in df_num.columns:
        array=df_num[column].values
        if np.isin(array[~np.isnan(array.tolist())], [0., 1.]).all()==True:
            num_to_cat_feat_list.append(column)
            
    df_cat=df.select_dtypes(include='object')
    df_cat=pd.concat([df_cat,df[num_to_cat_feat_list]], axis=1)
    df_num.drop(labels=num_to_cat_feat_list, axis=1, inplace=True)
    
    return df_num, df_cat
    
#---------------------------------#

def feature_selection_palencia(df_num, df_cat, list_numerical_desc_features, list_numerical_asc_features, list_categ_y_better, list_categ_n_better, target, new_predictors_asc=[], new_predictors_desc=[], min_iv=0.01):
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
        except:
            pass
    list_numerical_features=list_numerical_features_asc+list_numerical_features_desc
        
    return list_numerical_features, list_categorical_features, list_numerical_features_asc, list_numerical_features_desc
        
    
#---------------------------------#

def merging_for_model(df_all, list_numerical_features, list_categorical_features, target, list_numerical_features_asc, list_numerical_features_desc):
    list_categorical_features_spec_nan=[]
    for spec_cat_feat in list(set(list_categorical_features).intersection(['ELJCOUNTY1', 'ELJCOUNTY2', 'MBELJFILINGNAME1', 'MBELJFILINGNAME2', 'ELJSTATE1', 'ELJSTATE2', 'City_App', 'State_App'])):
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
    
#---------------------------------#

def encoder(df):
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


#---------------------------------#

def correlation_analysis(df_dum, target, threshold=0.67):
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
    
#---------------------------------#
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
    
#---------------------------------#

def build_model1(df_dum1, target):
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

            #print(params, ks_score_train,ks_score_test, ks_score_test-np.abs(ks_score_train-ks_score_test))
        data_grid_search.append([params, ks_score_train,ks_score_test, ks_score_test-np.abs(ks_score_train-ks_score_test)])
        params_dict[params]=ks_score_test-np.abs(ks_score_train-ks_score_test)
        


        
    #print(data_grid_search)
    df_grid_search=pd.DataFrame(data_grid_search, columns=['Parametrs', 'KS_train', 'KS_validation', 'Quality Measure'])
    st.dataframe(df_grid_search)
        
    lr = LogisticRegression(penalty=max(params_dict, key=params_dict.get)[0], C=max(params_dict, key=params_dict.get)[1], solver='saga')
    st.info(lr)
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
    #plt.show()
    #st.pyplot(fig)
    #buf=BytesIO()
    #fig.savefig(buf, format='png')
    #st.image(buf)
    
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
    #plt.show()
    #st.pyplot(fig)
    #buf=BytesIO()
    #fig.savefig(buf, format='png')
    #st.image(buf)

    #st.write('Gini =', 100* (2*logit_roc_auc-1.0).round(4))
    
    return lr, X_dum, y_dum
    
#---------------------------------#

def scoring(df_dum, X_dum, y_dum, target, lr, target_score = 450, target_odds = 1, pts_double_odds = 80):
    
    
    
    df_dum['logit']=np.log(lr.predict_proba(X_dum)[:,0]/lr.predict_proba(X_dum)[:,1])
    df_dum['odds'] = np.exp(df_dum['logit'])
    df_dum['probs'] = df_dum['odds'] / (df_dum['odds'] + 1)
    factor = pts_double_odds / np.log(2)
    offset = target_score - factor * np.log(target_odds)
    df_dum['score'] = offset + factor * df_dum['logit']
    
    intercept=offset-factor*lr.intercept_
    intercept_rounded=intercept.round(0)
    coefs=-factor*lr.coef_
    coefs_rounded=coefs.round(0)
    
    df_dum['score_rounded']=df_dum.loc[:, ~df_dum.columns.isin([target,'logit','odds','probs','score'])].dot(coefs_rounded[0])+intercept_rounded
    
    groupnum=len(df_dum.index)
    def n0(x): return sum(x==0)
    def n1(x): return sum(x==1)
    df_kslift = df_dum.sort_values('score_rounded', ascending=True).reset_index(drop=True)\
          .assign(group=lambda x: np.ceil((x.index+1)/(len(x.index)/groupnum)))\
          .groupby('group')[target].agg([n0,n1])\
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
    df_kslift=pd.concat([
        pd.DataFrame({'group':0, 'good':0, 'bad':0, 'good_distri':0, 'bad_distri':0, 'badrate':0, 'cumbadrate':np.nan, 'cumgood':0, 'cumbad':0, 'ks':0, 'lift':np.nan}, index=np.arange(1)),
        df_kslift
    ], ignore_index=True)
    
    score_list=df_dum['score_rounded'].sort_values(ascending=True).tolist()
    df_kslift['score']=[np.nan]+score_list
    optimal_cutoff=df_kslift[df_kslift['ks']==df_kslift['ks'].max()]['group'].tolist()[0]
    
    fig=plt.figure(figsize=(16,8))
    plt.hist([df_dum[df_dum[target]==0]['score_rounded'],df_dum[df_dum[target]==1]['score_rounded']],
             bins=80,
             edgecolor='white',
             color = ['g','r'],
             linewidth=1.2)

    plt.title('Scorecard Distribution', fontweight="bold", fontsize=14)
    plt.axvline(np.percentile(df_dum['score_rounded'],optimal_cutoff*100), color='k', linestyle='dashed', linewidth=1.5, alpha=0.5)
    plt.xlabel('Score')
    plt.ylabel('Count');
    buf=BytesIO()
    fig.savefig(buf, format='png')
    st.image(buf)
    st.write('Optimal cutoff = ', np.percentile(df_dum['score_rounded'],optimal_cutoff*100).round(0))
    #col1 = st.columns(1)
    #col1.metric(label="Optimal cutoff",value=round(np.percentile(df_dum['score_rounded'],optimal_cutoff*100), 0))
    
    df_ks = df_kslift
    
    ks_score = round(df_ks.loc[lambda x: x.ks==max(x.ks),'ks'].iloc[0],4)
    plist = ["eva_p"+i+'(df_'+i+',title)' for i in plot_type]
    subplot_nrows = int(np.ceil(len(plist)/2))
    subplot_ncols = int(np.ceil(len(plist)/subplot_nrows))
   
    fig = plt.figure(figsize=(8,8))
    for i in np.arange(len(plist)):
        plt.subplot(subplot_nrows,subplot_ncols,i+1)
        eval(plist[i])
    plt.show()
    buf=BytesIO()
    fig.savefig(buf, format='png')
    st.image(buf)
    
    logit_roc_auc = roc_auc_score(y_dum, -1*df_dum['score_rounded'])
    fpr, tpr, thresholds = roc_curve(y_dum, -1*df_dum['score_rounded'])
    fig=plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    buf=BytesIO()
    fig.savefig(buf, format='png')
    st.image(buf)

    #st.write('Gini =', 100* (2*logit_roc_auc-1.0).round(4))
    
    max_ks=(100*df_kslift['ks']).max()
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="KS-score",value=round(max_ks, 2))
    col2.metric(label="AUC ROC",value=logit_roc_auc.round(2))
    col3.metric(label="Gini", value=(100* (2*logit_roc_auc-1.0)).round(2))
    
    df_ks['score'].fillna(method='bfill', inplace=True)
    df_ks['score_prev']=df_ks['score'].astype(int)
    df_ks['score_next']=df_ks['score'].astype(int)+1
    
    df_ppt=pd.DataFrame(data={'cutoff_score': df_ks['score_prev'].sort_values(ascending=False).unique().tolist()})

    df_ppt['approval rate']=0
    for score in df_ks['score_prev'].sort_values(ascending=False).unique().tolist():
        df_ppt.loc[df_ppt['cutoff_score']==score, 'approval rate']=df_ks[df_ks['score']>score]['group'].count()/df_ks['group'].count()

    df_ppt['marginal odds ratio']=np.exp((df_ppt['cutoff_score']-offset)/factor)
    df_ppt['marginal good rate']=df_ppt['marginal odds ratio']/(1+df_ppt['marginal odds ratio'])
    df_ppt['good rate for total accepted']=0
    for score in df_ks['score_prev'].sort_values(ascending=False).unique().tolist():
        df_ppt.loc[df_ppt['cutoff_score']==score, 'good rate for total accepted']=df_ks[(df_ks['score']>=score)&(df_ks['good']==1)]['group'].count()/df_ks[df_ks['score']>=score]['group'].count()

    df_ppt['odds for total accepted']=df_ppt['good rate for total accepted']/(1-df_ppt['good rate for total accepted'])
    df_ppt['good rate for total rejected']=0
    for score in df_ks['score_prev'].sort_values(ascending=False).unique().tolist():
        df_ppt.loc[df_ppt['cutoff_score']==score, 'good rate for total rejected']=df_ks[(df_ks['score']<=score)&(df_ks['good']==1)]['group'].count()/df_ks[df_ks['score']<=score]['group'].count()

    df_ppt.loc[df_ppt['good rate for total rejected'].isna()==True, 'good rate for total rejected']=0
    df_ppt['odds for total rejected']=df_ppt['good rate for total rejected']/(1-df_ppt['good rate for total rejected'])
    
    df_scorecard=pd.DataFrame()

    df_scorecard['Feature']=np.concatenate((['Intercept'], lr.feature_names_in_))
    df_scorecard['Score']=np.concatenate((intercept_rounded, coefs_rounded[0]))

    with pd.option_context('display.max_rows', None,):
        st.write('Scorecard:')
        st.dataframe(df_scorecard.sort_values(by=['Feature']).reset_index(drop=True))
        
    df_scored=pd.concat([df,df_dum['score_rounded']], axis=1)
        
    return df_ppt, df_scorecard, df_scored
    
#---------------------------------#

def create_scorecard_ppt(df_scorecard, df_ppt):#, direction):
    
    output = BytesIO()
    #now=datetime.now()
    #dt_string= now.strftime("%d-%m-%Y_%H-%M-%S")
    #f_name=direction+'/'+project_name+'_SCORECARD_with_PPT_'+dt_string+'.xlsx'
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df_scorecard.sort_values(by=['Feature']).reset_index(drop=True).to_excel(writer, index=False,sheet_name='Scorecard')
    df_ppt.to_excel(writer, index=False,sheet_name='PPT')
    writer.save()
    processed_data = output.getvalue()
    
    return processed_data

#---------------------------------#

def download_scorecard_ppt(df_scorecard, df_ppt, project_name):
    data_xlsx = create_scorecard_ppt(df_scorecard, df_ppt)
    now=datetime.now()
    dt_string= now.strftime("%d-%m-%Y_%H-%M-%S")
    f_name=project_name+'_SCORECARD_with_PPT_'+dt_string+'.xlsx'
    st.download_button(label='📥 Download Current Results',
                                data=data_xlsx ,
                                file_name=f_name)
    
    
    
    #with open(f_name) as f:
    #    st.download_button('Download Excel file with Scorecard and PPT', f)
    
    #st.download_button(label='Download Excel file with Scorecard and PPT', data=df_ppt, file_name=project_name+'_SCORECARD_with_PPT_'+dt_string+'.xlsx')
    
#---------------------------------#

# Model building
def build_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    y = df.iloc[:,-1] # Selecting the last column as Y

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-split_size)/100)
    
    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(y.name)

    lr = LogisticRegression(penalty=parameter_penalty, C=parameter_C, solver='saga')
    st.write(lr)
    lr.fit(X_train, y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    y_pred_train = lr.predict_proba(X_train)[:,1]
    logit_roc_auc = roc_auc_score(y_train, y_pred_train)
    st.write('Quality measure (AUC ROC):')
    st.info(logit_roc_auc)
    fpr, tpr, thresholds = roc_curve(y_train, y_pred_train)
    fig= plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    #st.pyplot(fig)
    buf=BytesIO()
    fig.savefig(buf, forrmat='png')
    st.image(buf)

    st.markdown('**2.2. Test set**')
    y_pred_test = lr.predict_proba(X_test)[:,1]
    logit_roc_auc = roc_auc_score(y_test, y_pred_test)
    st.write('Quality measure (AUC ROC):')
    st.info(logit_roc_auc)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
    fig= plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    #st.pyplot(fig)
    buf=BytesIO()
    fig.savefig(buf, forrmat='png')
    st.image(buf)

    st.subheader('3. Model Parameters')
    st.write(lr.get_params())


with st.sidebar.header('1. Type your project name'):
    project_name = st.sidebar.text_input("Project name")
    
#---------------------------------#

with st.sidebar.header('2. Type a target name like PI, First_payment_default_Flag etc.'):
    target = st.sidebar.text_input("Target name")
    
#---------------------------------#
st.write("""
# The Custom Model App
In this implementation, the ML pipeline is used in order to build a regression model for""",project_name,"""using the Logistic Regression algorithm with regularization techinque and Palencia-based binning.
""")

#---------------------------------#
    

# Sidebar - Collects user input features into dataframe
with st.sidebar.header('3. Upload your data either in CSV or Excel type'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=["csv", "xls", "xlsx"], accept_multiple_files=False)
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")
    
#with st.sidebar.header('4. Add logic to external predictors'):
#    new_predictors_asc=st.multiselect('Add external features with ascending event rate', new_predictors)
#    new_predictors_desc=st.multiselect('Add external features with descending event rate', new_predictors1)

# Sidebar - Specify parameter settings
with st.sidebar.header('4. Set Parameters'):
    sparse_threshold = st.sidebar.slider('Sparse threshold', 50, 100, 95, 5)
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 70, 5)
    min_iv = st.sidebar.slider('Minimum Information Value of predictor', 0.01, 0.05, 0.01, 0.005)
    corr_threshold=st.sidebar.slider('Maximum value of paired correlation', 0.3, 0.8, 0.65, 0.05)

#with st.sidebar.subheader('4.1. Learning Parameters'):
#    parameter_penalty = st.sidebar.select_slider('Penalty', options=['l1','l2', 'none'])
#    parameter_C = st.sidebar.slider('C', 0.0, 1.0, 1.0, 0.1)
#    #parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    #parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

#with st.sidebar.subheader('4.2. General Parameters'):
#    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
#    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
#    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
#    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
#    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    if(uploaded_file.name.lower().endswith('.xlsx')):
        df = pd.read_excel(uploaded_file)
    elif(uploaded_file.name.lower().endswith('.xls')):
        df = pd.read_excel(uploaded_file)
    elif(uploaded_file.name.lower().endswith('.csv')):
        df = pd.read_csv(uploaded_file)
    else:
        st.markdown('**Incorrect file type. Please, upload a file either in csv or excel format.**')
    df=df_preprocessing(df, sparse_threshold=sparse_threshold, target=target)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df.head(5))
    st.write('Dataset shape:')
    st.info(df.shape)
    st.markdown('**1.2. Add logic for external predictors (optional)**')
    list_numerical_desc_features, list_numerical_asc_features, list_categ_y_better, list_categ_n_better, df_logic_dict = generator_of_predictors_logic(dictionary)
    new_predictors=list(set(df.select_dtypes(include=['int64','float64']).columns.tolist())-set(df_logic_dict['Variable Name (ReNamed)'].tolist())-set([target])) # features considering to be new compared to Full Dictionary
    #st.write('New predictors')
    #st.write(new_predictors)
    new_predictors_asc=st.multiselect('Add external features with ascending event rate', new_predictors)
    new_predictors_desc=st.multiselect('Add external features with descending event rate', new_predictors)
    
    st.markdown('**1.3. Exclude inappropriate predictors (optional)**')
    
    predictors_to_exclude=st.multiselect('Add inappropriate features to exclude', df.columns.tolist())
    df=df.loc[:, ~df.columns.isin(predictors_to_exclude)]
    


    st.subheader('2. Split dataset on numerical and categorical sub datasets')
    
    st.markdown('**2.1. Numerical sub dataset**')
    df_num=num_cat_split(df)[0]
    st.write(df_num.head(5))
    st.info(df_num.shape)
    st.markdown('**2.2. Categorical sub dataset**')
    df_cat=num_cat_split(df)[1]
    st.write(df_cat.head(5))
    st.info(df_cat.shape)

    st.subheader('3. Palencia-based binning')
    
    st.markdown('**3.1. Extended binning chracteristics**')
    list_numerical_features, list_categorical_features, list_numerical_features_asc, list_numerical_features_desc=feature_selection_palencia(df_num, df_cat, list_numerical_desc_features, list_numerical_asc_features, list_categ_y_better, list_categ_n_better, target=target,new_predictors_asc=new_predictors_asc, new_predictors_desc= new_predictors_desc,  min_iv=min_iv)
    st.markdown('**3.2. Selected features**')
    st.write('Categorical features:')
    st.write(list_categorical_features)
    st.write('Numerical features:')
    st.write(list_numerical_features) 
    #st.info(list_numerical_features_asc) 
    #st.info(list_numerical_features_desc) 
    df=merging_for_model(df, list_numerical_features, list_categorical_features, target, list_numerical_features_asc, list_numerical_features_desc)
    
    st.subheader('4. Encoding of selected dataset')
    
    df_dum=encoder(df)
    st.markdown('**4.1. Correlation matrix**')
    df_dum=correlation_analysis(df_dum, target, threshold=corr_threshold)
    st.markdown('**4.2. Dummies dataset**')
    st.write(df_dum.head(5))
    #st.info(df_dum.columns)
    st.info(df_dum.shape)
    #build_model(df_dum)
    st.subheader('5. Grid search and optimal model construction')
    lr, X_dum, y_dum = build_model1(df_dum, target)
    #st.info(lr)
    df_ppt, df_scorecard, df_scored=scoring(df_dum, X_dum, y_dum, target, lr, target_score = 450, target_odds = 1, pts_double_odds = 80)
    download_scorecard_ppt(df_scorecard, df_ppt, project_name)
    #create_scorecard_ppt(df_scorecard, df_ppt, direction=direction)
else:
    st.info('Awaiting for the file with Dataframe to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Diabetes dataset
        #diabetes = load_diabetes()
        #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        #Y = pd.Series(diabetes.target, name='response')
        #df = pd.concat( [X,Y], axis=1 )

        #st.markdown('The Diabetes dataset is used as the example.')
        #st.write(df.head(5))

        # Boston housing dataset
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        Y = pd.Series(boston.target, name='response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Boston housing dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)

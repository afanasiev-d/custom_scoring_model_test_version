
import numpy as np
import pandas as pd

def initial_filtering(df, sparse_threshold=0.95, target='First_payment_default_Flag'):
    
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

def missing_rate(df):
    
    df_missing_rate=pd.DataFrame()

    df_missing_rate['Feature']=df.isna().sum().index
    df_missing_rate['Missing rate']=((df.isna().sum()/len(df)).round(4) * 100).values
    
    return df_missing_rate

# Calculate information value
def calc_iv(df, feature, target, pr=False):
    """
    Set pr=True to enable printing of output.
    
    Output: 
      * iv: float,
      * data: pandas.DataFrame
    """

    lst = []

    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())


    iv = data['IV'].sum()
    # print(iv)

    return iv, data

def get_init_iv(df, target):
    dictionary_feature_iv={}
    for feat in df.columns:
        if feat!=target:
            dictionary_feature_iv[feat]=calc_iv(df, feat, target, pr=False)[0]
    df_iv=pd.DataFrame.from_dict(dictionary_feature_iv, orient='index', columns=['IV'])
    
    return df_iv

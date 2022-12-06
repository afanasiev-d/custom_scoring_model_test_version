import streamlit as st
import pandas as pd
import numpy as np

import preprocessing
import binning
from encoder import encoder
import correlation
import model
from scoring import scoring
import scorecard_ppt

dictionary='Full Dictionary.xlsx'
plot_type = ['ks']
title=''
new_predictors=[]
direction='C:/Users/Daniil Afanasiev/Projects/Preprocessing Pipeline'

#@st.cache
# Page layout
## Page expands to full width
st.set_page_config(page_title='Credit Scoring Custom Model App',
    layout='wide')

#---------------------------------#

with st.sidebar.header('1. Type your project name'):
    project_name = st.sidebar.text_input("Project name")
    
#---------------------------------#

with st.sidebar.header('2. Type a target name like PI, First_payment_default_Flag etc.'):
    target = st.sidebar.text_input("Target name")
    
#---------------------------------#
st.write("""
# Credit Scoring Custom Model App
In this implementation, the ML pipeline is used in order to build a credit scoring model for""",project_name,"""using the Logistic Regression algorithm with regularization techinque and Palencia-based binning.
""")

#---------------------------------#

# Sidebar - Collects user input features into dataframe
with st.sidebar.header('3. Upload your data either in CSV or Excel type'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=["csv", "xls", "xlsx"], accept_multiple_files=False)
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Sidebar - Specify parameter settings
with st.sidebar.header('4. Set Parameters'):
    sparse_threshold = st.sidebar.slider('Sparse threshold', 50, 100, 95, 5)
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 70, 5)
    min_iv = st.sidebar.slider('Minimum Information Value of predictor', 0.01, 0.05, 0.01, 0.005)
    corr_threshold=st.sidebar.slider('Maximum value of paired correlation', 0.3, 0.8, 0.65, 0.05)
    
with st.sidebar.subheader('4.1. Scoring Parameters'):
    target_score = st.sidebar.slider('Target score', 300, 600, 450, 50)
    target_odds = st.sidebar.slider('Target odds', 0.5, 2.0, 1.0, 0.5)
    pts_double_odds = st.sidebar.slider('Points to double odds', 10, 100, 80, 10)



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
    df_copy=df.copy()
    df_missing_rate=preprocessing.missing_rate(df_copy)
    df=preprocessing.initial_filtering(df, sparse_threshold=sparse_threshold, target=target)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df.head(5))
    st.write('Dataset shape:')
    st.info(df.shape)
    df_copy=df.copy()
    df_iv=preprocessing.get_init_iv(df_copy, target)
    st.markdown('**1.2. Add logic for external predictors (optional)**')
    list_numerical_desc_features, list_numerical_asc_features, list_categ_y_better, list_categ_n_better, df_logic_dict = preprocessing.generator_of_predictors_logic(dictionary)
    new_predictors=sorted(list(set(df.select_dtypes(include=['int64','float64']).columns.tolist())-set(df_logic_dict['Variable Name (ReNamed)'].tolist())-set([target]))) # features considering to be new compared to Full Dictionary
    new_predictors_asc=st.multiselect('Add external features with ascending event rate', new_predictors)
    new_predictors_desc=st.multiselect('Add external features with descending event rate', new_predictors)
    
    st.markdown('**1.3. Exclude inappropriate predictors (optional)**')
    
    predictors_to_exclude=st.multiselect('Add inappropriate features to exclude', df.columns.tolist())
    df=df.loc[:, ~df.columns.isin(predictors_to_exclude)]

    st.subheader('2. Split dataset on numerical and categorical sub datasets')
    
    st.markdown('**2.1. Numerical sub dataset**')
    df_num=preprocessing.num_cat_split(df)[0]
    st.write(df_num.head(5))
    st.info(df_num.shape)
    st.markdown('**2.2. Categorical sub dataset**')
    df_cat=preprocessing.num_cat_split(df)[1]
    st.write(df_cat.head(5))
    st.info(df_cat.shape)

    st.subheader('3. Palencia-based binning')
    
    st.markdown('**3.1. Extended binning chracteristics**')
    list_numerical_features, list_categorical_features, list_numerical_features_asc, list_numerical_features_desc, dictionary_feature_stat=binning.feature_selection_palencia(df_num, df_cat, list_numerical_desc_features, list_numerical_asc_features, list_categ_y_better, list_categ_n_better, target=target,new_predictors_asc=new_predictors_asc, new_predictors_desc= new_predictors_desc,  min_iv=min_iv)
    st.markdown('**3.2. Selected features**')
    st.write('Categorical features:')
    st.write(list_categorical_features)
    st.write('Numerical features:')
    st.write(list_numerical_features)  
    df=binning.merging_for_model(df, list_numerical_features, list_categorical_features, target, list_numerical_features_asc, list_numerical_features_desc)
    
    st.subheader('4. Encoding of selected dataset')
    
    df_dum=encoder(df,target)
    st.markdown('**4.1. Correlation matrix**')
    df_dum=correlation.filtering(df_dum, target, threshold=corr_threshold)
    st.markdown('**4.2. Dummies dataset**')
    st.write(df_dum.head(5))
    
    st.info(df_dum.shape)
    
    st.subheader('5. Grid search and optimal model construction')
    lr, X_dum, y_dum = model.build(df_dum, target)
    
    df_ppt, df_scorecard=scoring(df_dum, X_dum, y_dum, target, lr, target_score = target_score, target_odds = target_odds, pts_double_odds = pts_double_odds)
    scorecard_ppt.download(df_scorecard, df_ppt, df_missing_rate, df_iv, project_name, dictionary_feature_stat)
    
else:
    
    st.info('Awaiting for the file with Dataframe to be uploaded.')
    if st.button('Press to use Example Dataset'):
        st.subheader('1. Dataset')
        project_name='Genesis'
        uploaded_file='Example.xlsx'
        target='PI'
        df = pd.read_excel(uploaded_file)
        df_copy=df.copy()
        df_missing_rate=preprocessing.missing_rate(df_copy)
        df=preprocessing.initial_filtering(df, sparse_threshold=sparse_threshold, target=target)
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df.head(5))
        st.write('Dataset shape:')
        st.info(df.shape)
        st.markdown('**1.2. Add logic for external predictors (optional)**')
        list_numerical_desc_features, list_numerical_asc_features, list_categ_y_better, list_categ_n_better, df_logic_dict = preprocessing.generator_of_predictors_logic(dictionary)
        new_predictors=sorted(list(set(df.select_dtypes(include=['int64','float64']).columns.tolist())-set(df_logic_dict['Variable Name (ReNamed)'].tolist())-set([target]))) # features considering to be new compared to Full Dictionary

        new_predictors_asc=st.multiselect('Add external features with ascending event rate', new_predictors)
        new_predictors_desc=st.multiselect('Add external features with descending event rate', new_predictors)

        st.markdown('**1.3. Exclude inappropriate predictors (optional)**')

        predictors_to_exclude=st.multiselect('Add inappropriate features to exclude', df.columns.tolist())
        df=df.loc[:, ~df.columns.isin(predictors_to_exclude)]

        st.subheader('2. Split dataset on numerical and categorical sub datasets')

        st.markdown('**2.1. Numerical sub dataset**')
        df_num=preprocessing.num_cat_split(df)[0]
        st.write(df_num.head(5))
        st.info(df_num.shape)
        st.markdown('**2.2. Categorical sub dataset**')
        df_cat=preprocessing.num_cat_split(df)[1]
        st.write(df_cat.head(5))
        st.info(df_cat.shape)

        st.subheader('3. Palencia-based binning')

        st.markdown('**3.1. Extended binning chracteristics**')
        list_numerical_features, list_categorical_features, list_numerical_features_asc, list_numerical_features_desc=binning.feature_selection_palencia(df_num, df_cat, list_numerical_desc_features, list_numerical_asc_features, list_categ_y_better, list_categ_n_better, target=target,new_predictors_asc=new_predictors_asc, new_predictors_desc= new_predictors_desc,  min_iv=min_iv)
        st.markdown('**3.2. Selected features**')
        st.write('Categorical features:')
        st.write(list_categorical_features)
        st.write('Numerical features:')
        st.write(list_numerical_features) 
        df=binning.merging_for_model(df, list_numerical_features, list_categorical_features, target, list_numerical_features_asc, list_numerical_features_desc)

        st.subheader('4. Encoding of selected dataset')

        df_dum=encoder(df,target)
        st.markdown('**4.1. Correlation matrix**')
        df_dum=correlation.filtering(df_dum, target, threshold=corr_threshold)
        st.markdown('**4.2. Dummies dataset**')
        st.write(df_dum.head(5))
        st.info(df_dum.shape)
        st.subheader('5. Grid search and optimal model construction')
        lr, X_dum, y_dum = model.build(df_dum, target)
        df_ppt, df_scorecard=scoring(df_dum, X_dum, y_dum, target, lr, target_score = target_score, target_odds = target_odds, pts_double_odds = pts_double_odds)
        scorecard_ppt.download(df_scorecard, df_ppt, df_missing_rate, project_name)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import preprocessing
import binning
import woe
import correlation
import feature_engineering
import model
from scoring import scoring
import scorecard_ppt
import viz

viz.setup()  # apply the global fintech plot theme


def _show_results(res, full=True):
    """Persistent results + download section. Rendered right after a build (full=False:
    just the downloads, since the step boxes above already show everything) and on
    every later rerun such as a download click (full=True: re-render the scorecard and
    plots from cached session_state). Downloading never recomputes or resets the app —
    only the Restart button does."""
    st.markdown('---')
    st.success('✅ Model is ready. Use the buttons below — **downloading will not restart the app**.')
    cols = st.columns([1.3, 1.6, 1.2])
    cols[0].download_button('📥 Download Current Results', data=res['xlsx'],
                            file_name=res['xlsx_name'], key='dl_xlsx')
    if res.get('zip'):
        cols[1].download_button('📊 Download All Visualizations (ZIP)', data=res['zip'],
                                file_name=res['zip_name'], mime='application/zip', key='dl_zip')
    if cols[2].button('🔄 Restart (build a new model)', key='restart'):
        st.session_state.pop('cs_results', None)
        st.rerun()
    if full:
        if res.get('dash_fig') is not None:
            st.markdown('**Approval strategy — interactive cut-off dashboard**')
            st.caption('Hover any cut-off to compare approval / good / default / K-S against the optimal (max-K-S) cut-off.')
            st.plotly_chart(res['dash_fig'], width='stretch', key='approval_dash')
        st.markdown('**Scorecard**')
        st.table(viz.style_table(res['scorecard']))
        st.markdown('**Visualizations**')
        for _name, _png in res['gallery']:
            st.image(_png, width='stretch')


dictionary='Full Dictionary.xlsx'
plot_type = ['ks']
title=''
new_predictors=[]
direction='C:/Users/Daniil Afanasiev/Projects/Preprocessing Pipeline'

#@st.cache
# Page layout
## Page expands to full width
st.set_page_config(page_title='Credit Scoring Custom Model App',
    #page_icon='💳',
    layout='wide', initial_sidebar_state='expanded')

st.markdown(viz.app_css(), unsafe_allow_html=True)  # apply the fintech app theme

#---------------------------------#

with st.sidebar.header('1. Type your project name'):
    project_name = st.sidebar.text_input("Project name")
    
#---------------------------------#

with st.sidebar.header('2. Type the exact target name'):
    target = st.sidebar.text_input("Target name")
    
#---------------------------------#
_project_label = f' · <b>{project_name}</b>' if project_name else ''
st.markdown(f"""
<div class="hero">
  <h1>Credit Scoring — Custom Model Studio</h1>
  <p>Build an interpretable logistic-regression scorecard with Palencia-based binning, WOE encoding and automated tuning{_project_label}.</p>
  <span class="pill">Logistic Regression</span>&nbsp;
  <span class="pill">Regularization</span>&nbsp;
  <span class="pill">WoE Binning</span>&nbsp;
  <span class="pill">Scorecard &amp; PPT</span>
</div>
""", unsafe_allow_html=True)

#---------------------------------#

# Sidebar - Collects user input features into dataframe
with st.sidebar.header('3. Upload your data either in CSV or Excel type'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=["csv", "xls", "xlsx"], accept_multiple_files=False)

# Sidebar - Specify parameter settings
with st.sidebar.header('4. Set Parameters'):
    sparse_threshold = st.sidebar.slider('Sparse threshold', 50, 100, 95, 5)
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 70, 5)
    min_iv = st.sidebar.slider('Minimum Information Value of predictor', 0.01, 0.05, 0.01, 0.005)
    corr_threshold=st.sidebar.slider('Maximum value of paired correlation', 0.3, 0.8, 0.65, 0.05)
    cat_corr_threshold=st.sidebar.slider("Maximum categorical association (Cramér's V)", 0.3, 0.95, 0.7, 0.05)
    max_cardinality=st.sidebar.slider('Max distinct values for categorical features (high-cardinality cut-off)', 10, 50, 20, 5)
    use_feature_eng=st.sidebar.checkbox('Apply power transformations (feature engineering)', value=False)
    optimization_metric=st.sidebar.radio('Optimize hyperparameters on', ['KS', 'AUC ROC'], index=0, horizontal=True)
    use_cv=st.sidebar.checkbox('Use k-fold cross-validation (more robust)', value=False)
    k_folds=st.sidebar.slider('Number of folds (k)', 2, 8, 5, 1, disabled=not use_cv)
    cv_folds=k_folds if use_cv else None

with st.sidebar.subheader('4.1. Scoring Parameters'):
    target_score = st.sidebar.slider('Target score', 300, 600, 450, 50)
    target_odds = st.sidebar.slider('Target odds', 0.5, 2.0, 1.0, 0.5)
    pts_double_odds = st.sidebar.slider('Points to double odds', 10, 100, 80, 10)

with st.sidebar.subheader('4.2. Evaluation (bootstrap confidence intervals)'):
    ci_level_pct = st.sidebar.select_slider('Confidence level (%)', options=[90, 95, 99], value=95)
    n_boot = st.sidebar.select_slider('Bootstrap resamples',
                                      options=[500, 1000, 2000, 3000, 5000], value=2000)
    ci_level = ci_level_pct / 100.0



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
    df, converted_numeric=preprocessing.coerce_numeric_columns(df, target)
    df, dropped_high_card=preprocessing.filter_high_cardinality(df, target, max_cardinality=max_cardinality)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df.head(5))
    st.write('Dataset shape:')
    st.info(df.shape)
    if converted_numeric:
        st.caption(f'Converted {len(converted_numeric)} numeric-like feature(s) from categorical to numerical (now binned as numerical).')
    if dropped_high_card:
        st.caption(f'Excluded {len(dropped_high_card)} high-cardinality categorical feature(s) (> {max_cardinality} distinct values): {dropped_high_card}')
    df_copy=df.copy()
    df_iv=preprocessing.get_init_iv(df_copy, target)
    # Predictor configuration lives inside a form so that adding/removing
    # predictors does NOT trigger a rerun (and the heavy modelling below) on
    # every click. Nothing past this point runs until "Build model" is pressed.
    with st.form('predictor_config'):
        st.markdown('**1.2. Add logic for external predictors (optional)**')
        list_numerical_desc_features, list_numerical_asc_features, list_categ_y_better, list_categ_n_better, df_logic_dict = preprocessing.generator_of_predictors_logic(dictionary)
        new_predictors=sorted(list(set(df.select_dtypes(include=['int64','float64']).columns.tolist())-set(df_logic_dict['Variable Name (ReNamed)'].tolist())-set([target]))) # features considering to be new compared to Full Dictionary
        new_predictors_asc=st.multiselect('Add external features with ascending event rate', new_predictors)
        new_predictors_desc=st.multiselect('Add external features with descending event rate', new_predictors)

        st.markdown('**1.3. Exclude inappropriate predictors (optional)**')

        predictors_to_exclude=st.multiselect('Add inappropriate features to exclude', df.columns.tolist())

        build=st.form_submit_button('🚀 Build model')

    if build:
        st.session_state.pop('cs_results', None)            # fresh build clears old results
    elif 'cs_results' in st.session_state:
        _show_results(st.session_state['cs_results'])        # download rerun -> keep results, no recompute
        st.stop()
    else:
        st.info('Configure the predictors above, then press **🚀 Build model** to run binning, model construction and scoring.')
        st.stop()

    viz.reset_gallery()  # start collecting this run's figures for the download bundle
    df=df.loc[:, ~df.columns.isin(predictors_to_exclude)]

    # Each stage runs inside its own st.status box so the pipeline unfolds
    # visibly step by step (spinner + label while running, ✓ when done), with
    # the per-stage tables, progress bars and plots streaming inside each box.
    with st.status('Step 2 — Splitting dataset into numerical & categorical…', expanded=True) as status2:
        st.subheader('2. Split dataset on numerical and categorical sub datasets')
        df_num, df_cat = preprocessing.num_cat_split(df)
        st.markdown('**2.1. Numerical sub dataset**')
        st.write(df_num.head(5))
        st.info(df_num.shape)
        st.markdown('**2.2. Categorical sub dataset**')
        st.write(df_cat.head(5))
        st.info(df_cat.shape)
        status2.update(label='Step 2 — Dataset split complete ✓', state='complete')

    with st.status('Step 3 — Feature engineering (power transforms)…', expanded=True) as status_fe:
        st.subheader('3. Feature engineering')
        if use_feature_eng:
            _n_orig = df_num.shape[1]
            eng_df, eng_asc, eng_desc = feature_engineering.engineer_numerical(df_num, list(list_numerical_asc_features) + list(new_predictors_asc), list(list_numerical_desc_features) + list(new_predictors_desc))
            df = pd.concat([df, eng_df], axis=1)
            df_num = pd.concat([df_num, eng_df], axis=1)
            new_predictors_asc = list(new_predictors_asc) + eng_asc
            new_predictors_desc = list(new_predictors_desc) + eng_desc
            st.caption(f'Generated {eng_df.shape[1]} Box-Cox / power features (λ in [-2, 2] + log) from {_n_orig} numerical features.')
            st.write(eng_df.head(5))
        else:
            st.caption('Power transformations disabled — using the original numerical features only.')
        status_fe.update(label='Step 3 — Feature engineering complete ✓', state='complete')

    with st.status('Step 4 — Palencia-based binning (this is the slow part)…', expanded=True) as status3:
        st.subheader('4. Palencia-based binning')
        st.markdown('**4.1. Extended binning chracteristics**')
        list_numerical_features, list_categorical_features, list_numerical_features_asc, list_numerical_features_desc, dictionary_feature_stat, dictionary_feature_plots=binning.feature_selection_palencia(df_num, df_cat, list_numerical_desc_features, list_numerical_asc_features, list_categ_y_better, list_categ_n_better, target=target,new_predictors_asc=new_predictors_asc, new_predictors_desc= new_predictors_desc,  min_iv=min_iv)
        st.markdown('**4.2. Selected features**')
        st.write('Categorical features:')
        st.write(list_categorical_features)
        st.write('Numerical features:')
        st.write(list_numerical_features)
        df=binning.merging_for_model(df, list_numerical_features, list_categorical_features, target, list_numerical_features_asc, list_numerical_features_desc)
        status3.update(label='Step 4 — Binning complete ✓', state='complete')

    with st.status('Step 5 — WoE encoding & correlation filtering…', expanded=True) as status4:
        st.subheader('5. WoE encoding of selected dataset')
        st.markdown("**5.1. Business-logic check (\\*Match features)**")
        df, dropped_match=preprocessing.drop_illogical_match_features(df, target)
        if dropped_match:
            st.caption(f"Dropped {len(dropped_match)} '\\*Match' feature(s) where a match has a higher bad rate than no-match (contradicts business logic): {dropped_match}")
        st.markdown("**5.2. Categorical association (Cramér's V)**")
        df=correlation.filtering_categorical(df, target, threshold=cat_corr_threshold)
        df_dum, woe_map=woe.woe_transform(df, target)
        st.markdown('**5.3. WoE correlation matrix**')
        df_dum=correlation.filtering(df_dum, target, threshold=corr_threshold)
        st.markdown('**5.4. WoE-transformed dataset**')
        st.write(df_dum.head(5))
        st.info(df_dum.shape)
        status4.update(label='Step 5 — WoE encoding complete ✓', state='complete')

    with st.status('Step 6 — Bayessian search & optimal model construction…', expanded=True) as status5:
        st.subheader('6. Bayessian search and optimal model construction')
        lr, X_dum, y_dum = model.build(df_dum, target, metric=optimization_metric, cv_folds=cv_folds)
        status5.update(label='Step 6 — Model built ✓', state='complete')

    with st.status('Step 7 — Scoring & scorecard…', expanded=True) as status6:
        df_ppt, df_scorecard, dash_fig, df_ci=scoring(df_dum, X_dum, y_dum, target, lr, woe_map=woe_map, target_score = target_score, target_odds = target_odds, pts_double_odds = pts_double_odds, n_boot = n_boot, ci_level = ci_level)
        status6.update(label='Step 7 — Scoring complete ✓', state='complete')

    _ts=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    st.session_state['cs_results']={
        'xlsx': scorecard_ppt.create(df_scorecard, df_ppt, df_missing_rate, df_iv, dictionary_feature_stat, dictionary_feature_plots, df_ci=df_ci),
        'xlsx_name': f'{project_name}_SCORECARD_PPT_satistics_{_ts}.xlsx',
        'zip': viz.gallery_zip(),
        'zip_name': f'{project_name}_VISUALIZATIONS_{_ts}.zip',
        'scorecard': df_scorecard,
        'gallery': list(viz.gallery_items()),
        'dash_fig': dash_fig,
    }
    _show_results(st.session_state['cs_results'], full=False)

else:
    
    st.info('Awaiting for the file with Dataframe to be uploaded.')
    run_example = st.button('Press to use Example Dataset')
    if 'cs_results' in st.session_state and not run_example:
        _show_results(st.session_state['cs_results'])   # download rerun -> keep results
        st.stop()
    if run_example:
        st.session_state.pop('cs_results', None)
        st.subheader('1. Dataset')
        project_name='Genesis'
        uploaded_file='Example.xlsx'
        target='PI'
        df = pd.read_excel(uploaded_file)
        df_copy=df.copy()
        df_missing_rate=preprocessing.missing_rate(df_copy)
        df=preprocessing.initial_filtering(df, sparse_threshold=sparse_threshold, target=target)
        df, converted_numeric=preprocessing.coerce_numeric_columns(df, target)
        df, dropped_high_card=preprocessing.filter_high_cardinality(df, target, max_cardinality=max_cardinality)
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df.head(5))
        st.write('Dataset shape:')
        st.info(df.shape)
        if converted_numeric:
            st.caption(f'Converted {len(converted_numeric)} numeric-like feature(s) from categorical to numerical (now binned as numerical).')
        if dropped_high_card:
            st.caption(f'Excluded {len(dropped_high_card)} high-cardinality categorical feature(s) (> {max_cardinality} distinct values): {dropped_high_card}')
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

        viz.reset_gallery()  # start collecting this run's figures for the download bundle

        with st.status('Step 2 — Splitting dataset into numerical & categorical…', expanded=True) as status2:
            st.subheader('2. Split dataset on numerical and categorical sub datasets')
            df_num, df_cat = preprocessing.num_cat_split(df)
            st.markdown('**2.1. Numerical sub dataset**')
            st.write(df_num.head(5))
            st.info(df_num.shape)
            st.markdown('**2.2. Categorical sub dataset**')
            st.write(df_cat.head(5))
            st.info(df_cat.shape)
            status2.update(label='Step 2 — Dataset split complete ✓', state='complete')

        with st.status('Step 3 — Feature engineering (power transforms)…', expanded=True) as status_fe:
            st.subheader('3. Feature engineering')
            if use_feature_eng:
                _n_orig = df_num.shape[1]
                eng_df, eng_asc, eng_desc = feature_engineering.engineer_numerical(df_num, list(list_numerical_asc_features) + list(new_predictors_asc), list(list_numerical_desc_features) + list(new_predictors_desc))
                df = pd.concat([df, eng_df], axis=1)
                df_num = pd.concat([df_num, eng_df], axis=1)
                new_predictors_asc = list(new_predictors_asc) + eng_asc
                new_predictors_desc = list(new_predictors_desc) + eng_desc
                st.caption(f'Generated {eng_df.shape[1]} Box-Cox / power features (λ in [-2, 2] + log) from {_n_orig} numerical features.')
                st.write(eng_df.head(5))
            else:
                st.caption('Power transformations disabled — using the original numerical features only.')
            status_fe.update(label='Step 3 — Feature engineering complete ✓', state='complete')

        with st.status('Step 4 — Palencia-based binning (this is the slow part)…', expanded=True) as status3:
            st.subheader('4. Palencia-based binning')
            st.markdown('**4.1. Extended binning chracteristics**')
            list_numerical_features, list_categorical_features, list_numerical_features_asc, list_numerical_features_desc, dictionary_feature_stat, dictionary_feature_plots=binning.feature_selection_palencia(df_num, df_cat, list_numerical_desc_features, list_numerical_asc_features, list_categ_y_better, list_categ_n_better, target=target,new_predictors_asc=new_predictors_asc, new_predictors_desc= new_predictors_desc,  min_iv=min_iv)
            st.markdown('**4.2. Selected features**')
            st.write('Categorical features:')
            st.write(list_categorical_features)
            st.write('Numerical features:')
            st.write(list_numerical_features)
            df=binning.merging_for_model(df, list_numerical_features, list_categorical_features, target, list_numerical_features_asc, list_numerical_features_desc)
            status3.update(label='Step 4 — Binning complete ✓', state='complete')

        with st.status('Step 5 — WoE encoding & correlation filtering…', expanded=True) as status4:
            st.subheader('5. WoE encoding of selected dataset')
            st.markdown("**5.1. Business-logic check (\\*Match features)**")
            df, dropped_match=preprocessing.drop_illogical_match_features(df, target)
            if dropped_match:
                st.caption(f"Dropped {len(dropped_match)} '\\*Match' feature(s) where a match has a higher bad rate than no-match (contradicts business logic): {dropped_match}")
            st.markdown("**5.2. Categorical association (Cramér's V)**")
            df=correlation.filtering_categorical(df, target, threshold=cat_corr_threshold)
            df_dum, woe_map=woe.woe_transform(df, target)
            st.markdown('**5.3. WoE correlation matrix**')
            df_dum=correlation.filtering(df_dum, target, threshold=corr_threshold)
            st.markdown('**5.4. WoE-transformed dataset**')
            st.write(df_dum.head(5))
            st.info(df_dum.shape)
            status4.update(label='Step 5 — WoE encoding complete ✓', state='complete')

        with st.status('Step 6 — Bayessian search & optimal model construction…', expanded=True) as status5:
            st.subheader('6. Bayessian search and optimal model construction')
            lr, X_dum, y_dum = model.build(df_dum, target, metric=optimization_metric, cv_folds=cv_folds)
            status5.update(label='Step 6 — Model built ✓', state='complete')

        with st.status('Step 7 — Scoring & scorecard…', expanded=True) as status6:
            df_ppt, df_scorecard, dash_fig, df_ci=scoring(df_dum, X_dum, y_dum, target, lr, woe_map=woe_map, target_score = target_score, target_odds = target_odds, pts_double_odds = pts_double_odds, n_boot = n_boot, ci_level = ci_level)
            status6.update(label='Step 7 — Scoring complete ✓', state='complete')

        _ts=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        st.session_state['cs_results']={
            'xlsx': scorecard_ppt.create(df_scorecard, df_ppt, df_missing_rate, df_iv, dictionary_feature_stat, dictionary_feature_plots, df_ci=df_ci),
            'xlsx_name': f'{project_name}_SCORECARD_PPT_satistics_{_ts}.xlsx',
            'zip': viz.gallery_zip(),
            'zip_name': f'{project_name}_VISUALIZATIONS_{_ts}.zip',
            'scorecard': df_scorecard,
            'gallery': list(viz.gallery_items()),
        'dash_fig': dash_fig,
        }
        _show_results(st.session_state['cs_results'], full=False)

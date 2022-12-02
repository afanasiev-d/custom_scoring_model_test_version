from io import BytesIO
import pandas as pd
from datetime import datetime
import streamlit as st

def create(df_scorecard, df_ppt, df_missing_rate, dictionary_feature_stat):  #create .xlsx file in appropriate format with 2 sheets that represent scorecard and perfomance projection table 
    
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    dfs = {'Scorecard': df_scorecard.sort_values(by=['Feature']).reset_index(drop=True), 
          'PPT': df_ppt.round(4),
          'Missing rate': df_missing_rate.reset_index(drop=True)}
    workbook=writer.book        
    cell_format=workbook.add_format({'align': 'center', 'valign': 'vcenter'})
    for sheetname, df in dfs.items():
        df.to_excel(writer, index=False, sheet_name=sheetname)
        worksheet=writer.sheets[sheetname]
        for idx, col in enumerate(df):
            series=df[col]
            max_len=max((series.astype(str).map(len).max(), len(str(series.name))))+1
            if col!='Feature':
                worksheet.set_column(idx, idx, max_len, cell_format)
            else:
                worksheet.set_column(idx, idx, max_len)
            #worksheet.set_column(idx, idx, max_len)
        for idx in range(df.shape[0]+1):
            worksheet.set_row(idx, 20)
            
    for sheetname, df in dictionary_feature_stat.items():
        df.round(4).to_excel(writer, index=False, sheet_name=sheetname)
        worksheet=writer.sheets[sheetname]
        for idx, col in enumerate(df):
            series=df[col]
            max_len=max((series.astype(str).map(len).max(), len(str(series.name))))+1
            if col!='Feature':
                worksheet.set_column(idx, idx, max_len, cell_format)
            else:
                worksheet.set_column(idx, idx, max_len)
            #worksheet.set_column(idx, idx, max_len)
        for idx in range(df.shape[0]+1):
            worksheet.set_row(idx, 20)
    
    writer.save()
    processed_data = output.getvalue()
    
    return processed_data

#---------------------------------#

def download(df_scorecard, df_ppt, df_missing_rate, project_name, dictionary_feature_stat):  #download constructed excel file
    
    data_xlsx = create(df_scorecard, df_ppt, df_missing_rate, dictionary_feature_stat)
    now=datetime.now()
    dt_string= now.strftime("%d-%m-%Y_%H-%M-%S")
    f_name=project_name+'_SCORECARD_PPT_satistics_'+dt_string+'.xlsx'
    st.download_button(label='📥 Download Current Results',
                                data=data_xlsx ,
                                file_name=f_name)

import re
from io import BytesIO
import pandas as pd
from datetime import datetime
import streamlit as st
import viz

# Fintech workbook palette (mirrors the on-screen charts).
_HDR_BG   = viz.NAVY      # header fill
_HDR_FG   = '#FFFFFF'     # header text
_BAND     = '#F1F5F9'     # zebra band fill
_BORDER   = '#D7DEE8'     # cell borders
_TEXT_L   = ('feature', 'variable', 'bin', 'value')  # left-aligned text columns


def _safe_sheet_name(name, used):
    """Excel sheet names: <=31 chars, no : \\ / ? * [ ], and unique."""
    s = re.sub(r'[:\\/?*\[\]]', '_', str(name)).strip()[:31] or 'Sheet'
    base, i = s, 1
    while s.lower() in used:
        suffix = f'_{i}'
        s = base[:31 - len(suffix)] + suffix
        i += 1
    used.add(s.lower())
    return s


def _num_format(col, series):
    """Pick an Excel number format string for a column based on its name/values."""
    name = str(col).lower()
    if not pd.api.types.is_numeric_dtype(series):
        return None
    vals = series.dropna()
    if len(vals) == 0:
        return '0'
    if 'missing' in name and 'rate' in name:        # already 0-100 scaled
        return '0.00"%"'
    is_fraction = bool(((vals >= -0.0001) & (vals <= 1.0001)).all())
    if is_fraction and any(k in name for k in ('rate', 'share', 'distribution', 'approval')):
        return '0.00%'
    if bool((vals == vals.round(0)).all()):          # whole numbers
        return '#,##0'
    return '#,##0.0000'


def _style_sheet(workbook, worksheet, df):
    """Write a dataframe to an already-created worksheet with the house style."""
    nrows, ncols = df.shape
    header_fmt = workbook.add_format({
        'bold': True, 'font_color': _HDR_FG, 'bg_color': _HDR_BG, 'font_size': 10,
        'align': 'center', 'valign': 'vcenter', 'border': 1, 'border_color': _HDR_BG,
        'text_wrap': True,
    })
    band_fmt = workbook.add_format({'bg_color': _BAND})

    # Header row
    for c, col in enumerate(df.columns):
        worksheet.write(0, c, str(col), header_fmt)
    worksheet.set_row(0, 30)

    # Per-column width + number format + alignment
    for c, col in enumerate(df.columns):
        series = df[col]
        align = 'left' if str(col).lower() in _TEXT_L else 'center'
        cell = {'valign': 'vcenter', 'align': align, 'border': 1, 'border_color': _BORDER}
        nf = _num_format(col, series)
        if nf:
            cell['num_format'] = nf
        width = int(min(max(series.astype(str).map(lambda v: len(str(v))).max(),
                            len(str(col))) + 3, 46))
        worksheet.set_column(c, c, width, workbook.add_format(cell))

    # Polish: freeze header, autofilter, zebra banding
    worksheet.freeze_panes(1, 0)
    if nrows > 0:
        worksheet.autofilter(0, 0, nrows, ncols - 1)
        worksheet.conditional_format(1, 0, nrows, ncols - 1,
                                     {'type': 'formula', 'criteria': '=MOD(ROW(),2)=0',
                                      'format': band_fmt})
    worksheet.set_default_row(18)


def create(df_scorecard, df_ppt, df_missing_rate, df_iv, dictionary_feature_stat):  #create a richly formatted .xlsx report
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    workbook = writer.book

    dfs = {'Scorecard': df_scorecard.sort_values(by=['Feature']).reset_index(drop=True),
           'PPT': df_ppt.round(4),
           'Missing rate': df_missing_rate.reset_index(drop=True),
           'Initial IV': df_iv.reset_index(drop=False).round(4)}

    used = set()
    # Summary sheets get a teal tab; per-feature binning sheets a navy tab.
    for sheetname, df in dfs.items():
        name = _safe_sheet_name(sheetname, used)
        df.to_excel(writer, index=False, header=False, startrow=1, sheet_name=name)
        ws = writer.sheets[name]
        ws.set_tab_color(viz.TEAL)
        _style_sheet(workbook, ws, df)

    for sheetname, df in dictionary_feature_stat.items():
        df = df.round(4)
        name = _safe_sheet_name(sheetname, used)
        df.to_excel(writer, index=False, header=False, startrow=1, sheet_name=name)
        ws = writer.sheets[name]
        ws.set_tab_color(viz.NAVY)
        _style_sheet(workbook, ws, df)

    writer.close()
    processed_data = output.getvalue()

    return processed_data

#---------------------------------#

def download(df_scorecard, df_ppt, df_missing_rate, df_iv, project_name, dictionary_feature_stat):  #download constructed excel file
    
    data_xlsx = create(df_scorecard, df_ppt, df_missing_rate, df_iv, dictionary_feature_stat)
    now=datetime.now()
    dt_string= now.strftime("%d-%m-%Y_%H-%M-%S")
    f_name=project_name+'_SCORECARD_PPT_satistics_'+dt_string+'.xlsx'
    st.download_button(label='📥 Download Current Results',
                                data=data_xlsx ,
                                file_name=f_name)

#---------------------------------#

def download_visuals(zip_bytes, project_name):  #download all generated visualizations as a .zip of PNGs
    if not zip_bytes:
        return
    now=datetime.now()
    dt_string= now.strftime("%d-%m-%Y_%H-%M-%S")
    f_name=project_name+'_VISUALIZATIONS_'+dt_string+'.zip'
    st.download_button(label='📊 Download All Visualizations (ZIP)',
                                data=zip_bytes,
                                file_name=f_name,
                                mime='application/zip')

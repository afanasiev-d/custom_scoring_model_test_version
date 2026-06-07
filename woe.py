import numpy as np
import pandas as pd


def woe_transform(df, target, smoothing=0.5):
    """Weight-of-Evidence encode every binned categorical column.

    Each ``{feature}_cat`` column (the output of ``binning.merging_for_model``) is
    replaced by a single numeric column holding the WoE of the category each row
    falls into. WoE is Laplace-smoothed so empty good/bad cells never blow up to
    +/- infinity::

        WoE(bin) = ln( ((good + s)/total_good) / ((bad + s)/total_bad) )

    Returns ``(df_woe, woe_map)`` where:
      * ``df_woe`` has one numeric WoE column per feature (+ the target),
      * ``woe_map[col]`` is a DataFrame ``[Category, WoE, Count, Share (%)]`` used
        to build the interpretable scorecard.
    """
    y = df[target]
    is_good = (y == 0).to_numpy().astype(float)
    is_bad = (y == 1).to_numpy().astype(float)
    total_good = float(is_good.sum()) or 1.0
    total_bad = float(is_bad.sum()) or 1.0
    n = len(df)

    out = {}
    woe_map = {}
    for c in [col for col in df.columns if col != target]:
        cat = df[c].astype('object')
        agg = (pd.DataFrame({'Category': cat.to_numpy(), 'good': is_good, 'bad': is_bad})
               .groupby('Category', observed=True)
               .agg(good=('good', 'sum'), bad=('bad', 'sum'), Count=('good', 'count')))
        woe = np.log(((agg['good'] + smoothing) / total_good) /
                     ((agg['bad'] + smoothing) / total_bad))
        woe_dict = woe.to_dict()

        # Missing values are neutral for the model: the NaN category is encoded as
        # WoE = 0 so it never contributes to the score (the *fair* WoE is still kept
        # in woe_map for display in the scorecard).
        model_dict = {k: (0.0 if str(k) == 'NaN' else float(v)) for k, v in woe_dict.items()}
        out[c] = cat.map(model_dict).astype(float).to_numpy()

        m = agg.reset_index()
        m['WoE'] = m['Category'].map(woe_dict)
        m['Share (%)'] = (m['Count'] / n * 100).round(1)
        woe_map[c] = m[['Category', 'WoE', 'Count', 'Share (%)']]

    df_woe = pd.DataFrame(out, index=df.index)
    df_woe[target] = y.to_numpy()
    return df_woe, woe_map

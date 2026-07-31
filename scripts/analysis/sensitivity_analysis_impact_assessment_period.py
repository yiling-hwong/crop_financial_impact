# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""

import os
import sys
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('..')
import utils

crops = ['maize', 'wheat', 'soy']
pred_str = "t3s3"
spei_month = "03"
weight_opt = "weighted"  # "weighted", "simple"

start_year_hist, end_year_hist = 1974, 2004  # fixed baseline period

start_year_fut = 1990
end_year_fut = 2019

assessment_periods = [[1990, 2019], [2000, 2019], [2007, 2019]]
period_labels = [f"{p[0]}-{p[1]}" for p in assessment_periods]

root_dir = f'../../data'
ar6_region = "region_ar6_dev"

year_cols_all = [str(x) for x in range(start_year_fut, end_year_fut + 1)]
years_all = list(range(start_year_fut, end_year_fut + 1))

hist_fut_suffix = f"hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"

# Valid countries
hist_file = f"{root_dir}/historical/linregress_inputs/INPUT_HISTORICAL_DETRENDED_DEMEANED_spei{spei_month}.csv"
df_hist = pd.read_csv(hist_file)
valid_countries = df_hist["country"].unique()

# IPCC regions
df_ipcc = utils.get_ipcc_region_df()
df_ipcc = df_ipcc[["Country", ar6_region]].rename(columns={"Country": "country"})

prod_weights = utils.load_fao_prod_weights(root_dir, spei_month)
gdp_weights  = utils.load_gdp_weights(root_dir)

"""
HELPERS
"""
def load_metric_dfs(folder_name, suffix):
    """Load dy/dp/degdp/de csvs (all crops) for one category (e.g. isimip3a, topbot, carbmaj)."""
    dy_dfs, dp_dfs, degdp_dfs, de_dfs = [], [], [], []
    for crop in crops:
        base = f"{root_dir}/historical/linregress_outputs/{crop}/{folder_name}"
        for fname, lst in [
            (f"{base}/dy_{suffix}",    dy_dfs),
            (f"{base}/dp_{suffix}",    dp_dfs),
            (f"{base}/degdp_{suffix}", degdp_dfs),
            (f"{base}/de_{suffix}",    de_dfs),
        ]:
            df = pd.read_csv(fname)
            df = df[df['country'].isin(valid_countries)]
            df["crop"] = crop
            lst.append(df)
    return (
        pd.concat(dy_dfs,    ignore_index=True),
        pd.concat(dp_dfs,    ignore_index=True),
        pd.concat(degdp_dfs, ignore_index=True),
        pd.concat(de_dfs,    ignore_index=True),
    )


def compute_ts(df_dy, df_dp, df_degdp, df_de, regional_method="weighted"):
    dy_ts = utils.compute_dy_weighted(df_dy, df_dp, year_cols_all, prod_weights=prod_weights, weight_opt=weight_opt)
    dp_ts = (df_dp[year_cols_all].sum(axis=0) / 1e6).values
    de_ts = (df_de[year_cols_all].sum(axis=0) / 1e9).values
    degdp_ts = utils.compute_degdp_weighted(
        df_degdp[["country"] + year_cols_all],
        year_cols_all,
        gdp_weights=gdp_weights, weight_opt=weight_opt
    )

    df_degdp_ipcc = df_degdp.merge(df_ipcc, on="country", how="left")

    if regional_method == "weighted":
        degdp_ldc_ts = utils.compute_degdp_weighted(
            df_degdp_ipcc[df_degdp_ipcc[ar6_region] == "ldc"][["country"] + year_cols_all],
            year_cols_all, gdp_weights=gdp_weights, weight_opt=weight_opt
        )
        degdp_dev_ts = utils.compute_degdp_weighted(
            df_degdp_ipcc[df_degdp_ipcc[ar6_region] == "developed"][["country"] + year_cols_all],
            year_cols_all, gdp_weights=gdp_weights, weight_opt=weight_opt
        )
    else:
        degdp_ldc_ts = (df_degdp_ipcc[df_degdp_ipcc[ar6_region] == "ldc"]
                        .groupby("country")[year_cols_all].mean().mean(axis=0).values)
        degdp_dev_ts = (df_degdp_ipcc[df_degdp_ipcc[ar6_region] == "developed"]
                        .groupby("country")[year_cols_all].mean().mean(axis=0).values)

    return {
        "dy":        dy_ts,
        "dp":        dp_ts,
        "degdp":     degdp_ts,
        "de":        de_ts,
        "degdp_ldc": degdp_ldc_ts,
        "degdp_dev": degdp_dev_ts,
    }


def period_slice(period):
    s, e = period
    return years_all.index(s), years_all.index(e) + 1


def summary_table(ts, title):
    rows = []
    row_defs = [
        ("dy [%]",                                "dy",        np.mean),
        ("degdp [%]",                             "degdp",     np.mean),
        ("dp [Mt]",                                "dp",        np.sum),
        ("de [B US$]",                              "de",        np.sum),
        ("GDP loss LDC (degdp of LDC) [%]",        "degdp_ldc", np.mean),
        ("GDP loss developed (degdp of dev.) [%]", "degdp_dev", np.mean),
    ]
    for label, key, agg_fn in row_defs:
        row = {"metric": label}
        for period, lbl in zip(assessment_periods, period_labels):
            s, e = period_slice(period)
            row[lbl] = round(float(agg_fn(ts[key][s:e])), 6)
        rows.append(row)

    ratio_row = {"metric": "GDP loss ratio LDC/dev"}
    for period, lbl in zip(assessment_periods, period_labels):
        s, e = period_slice(period)
        ldc = float(np.mean(ts["degdp_ldc"][s:e]))
        dev = float(np.mean(ts["degdp_dev"][s:e]))
        ratio_row[lbl] = round(ldc / dev, 4) if dev != 0 else float('nan')
    rows.append(ratio_row)

    df_summary = pd.DataFrame(rows).set_index("metric")
    print()
    print("=" * 90)
    print(f"IMPACT ASSESSMENT PERIOD SENSITIVITY, {title} (baseline hist{start_year_hist}-{end_year_hist})")
    print("=" * 90)
    print(df_summary.to_string())
    return df_summary


"""
GLOBAL (all crops, no emitter split)
"""
suffix_global = f"isimip3a_{pred_str}_{hist_fut_suffix}"
df_dy, df_dp, df_degdp, df_de = load_metric_dfs("isimip3a", suffix_global)
ts_global = compute_ts(df_dy, df_dp, df_degdp, df_de)
df_summary_global = summary_table(ts_global, "historical heat and drought impacts")

"""
EMITTER-LINKED DAMAGES: top10 / bot50 (topbot) and carbmaj
"""
suffix_top10 = f"top10_{pred_str}_{hist_fut_suffix}"
df_dy_top, df_dp_top, df_degdp_top, df_de_top = load_metric_dfs("topbot", suffix_top10)
ts_top10 = compute_ts(df_dy_top, df_dp_top, df_degdp_top, df_de_top, regional_method="simple")
df_summary_top10 = summary_table(ts_top10, "top10 emitters")

suffix_bot50 = f"bot50_{pred_str}_{hist_fut_suffix}"
df_dy_bot, df_dp_bot, df_degdp_bot, df_de_bot = load_metric_dfs("topbot", suffix_bot50)
ts_bot50 = compute_ts(df_dy_bot, df_dp_bot, df_degdp_bot, df_de_bot, regional_method="simple")
df_summary_bot50 = summary_table(ts_bot50, "bottom50 emitters")

suffix_carb = f"carbmaj_{pred_str}_{hist_fut_suffix}"
df_dy_carb, df_dp_carb, df_degdp_carb, df_de_carb = load_metric_dfs("carbmaj", suffix_carb)
ts_carb = compute_ts(df_dy_carb, df_dp_carb, df_degdp_carb, df_de_carb)
df_summary_carb = summary_table(ts_carb, "carbon majors")

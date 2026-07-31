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

"""
PARAMETERS
"""
crops = ['maize', 'wheat', 'soy']
pred_str = "t3s3"
spei_month = "03"
weight_opt = "weighted"  # "weighted", "simple"

baseline_periods = [[1974, 2004], [1970, 1990], [1980, 2000], [1984, 2014]]

start_year_fut = 1990
end_year_fut = 2019

start_year_analysis = 2007
end_year_analysis = 2019

root_dir = f'../../data'
ar6_region = "region_ar6_dev"

period_labels = [f"{p[0]}-{p[1]}" for p in baseline_periods]
year_cols_all = [str(x) for x in range(start_year_fut, end_year_fut + 1)]
years_all = list(range(start_year_fut, end_year_fut + 1))
start_idx = years_all.index(start_year_analysis)
end_idx = years_all.index(end_year_analysis) + 1

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
LOAD DATA AND COMPUTE TIMESERIES
"""
ts_data = {}

for baseline_period, period_label in zip(baseline_periods, period_labels):
    start_year_hist, end_year_hist = baseline_period

    dy_dfs, dp_dfs, degdp_dfs, de_dfs = [], [], [], []
    for crop in crops:
        base = f"{root_dir}/historical/linregress_outputs/{crop}/isimip3a"
        suffix = f"isimip3a_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
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

    df_dy    = pd.concat(dy_dfs,    ignore_index=True)
    df_dp    = pd.concat(dp_dfs,    ignore_index=True)
    df_degdp = pd.concat(degdp_dfs, ignore_index=True)
    df_de    = pd.concat(de_dfs,    ignore_index=True)

    # ── Global timeseries ──────────────────────────────────────────────
    # dy: FAO production-weighted timeseries
    dy_ts = utils.compute_dy_weighted(df_dy, df_dp, year_cols_all, prod_weights=prod_weights, weight_opt=weight_opt)

    # dp: sum crops per country, then sum across countries -> Mt
    dp_ts = (df_dp.groupby("country")[year_cols_all].sum().sum(axis=0) / 1e6).values

    # degdp: World Bank GDP-weighted mean
    degdp_ts = utils.compute_degdp_weighted(
        df_degdp[["country"] + year_cols_all],
        year_cols_all,
        gdp_weights=gdp_weights, weight_opt=weight_opt
    )

    # de: sum across all crop-country rows -> B USD
    de_ts = (df_de.groupby("country")[year_cols_all].sum().sum(axis=0) / 1e9).values

    # ── Regional degdp timeseries ──────────────────────────────────────
    df_degdp_ipcc = df_degdp.merge(df_ipcc, on="country", how="left")
    df_de_ipcc    = df_de.merge(df_ipcc, on="country", how="left")

    degdp_ldc_ts = utils.compute_degdp_weighted(
        df_degdp_ipcc[df_degdp_ipcc[ar6_region] == "ldc"][["country"] + year_cols_all],
        year_cols_all,
        gdp_weights=gdp_weights, weight_opt=weight_opt
    )
    degdp_dev_ts = utils.compute_degdp_weighted(
        df_degdp_ipcc[df_degdp_ipcc[ar6_region] == "developed"][["country"] + year_cols_all],
        year_cols_all,
        gdp_weights=gdp_weights, weight_opt=weight_opt
    )

    ts_data[period_label] = {
        "dy":           dy_ts,
        "dp":           dp_ts,
        "degdp":        degdp_ts,
        "de":           de_ts,
        "degdp_ldc":    degdp_ldc_ts,
        "degdp_dev":    degdp_dev_ts,
    }

"""
SUMMARY TABLE (mean/sum over study period)
"""
def period_mean(arr): return float(np.mean(arr[start_idx:end_idx]))
def period_sum(arr):  return float(np.sum(arr[start_idx:end_idx]))

rows = []
for metric_label, key, agg_fn in [
    ("dy global [%]",          "dy",        period_mean),
    ("degdp global [%]", "degdp", period_mean),
    ("dp global [Mt]",         "dp",        period_sum),
    ("de global [B US$]",      "de",        period_sum),
    ("degdp LDC [%]",          "degdp_ldc", period_mean),
    ("degdp developed [%]",    "degdp_dev", period_mean),
]:
    row = {"metric": metric_label}
    for lbl in period_labels:
        row[lbl] = round(agg_fn(ts_data[lbl][key]), 6)
    rows.append(row)

df_summary = pd.DataFrame(rows).set_index("metric")
print()
print("=" * 90)
print(f"BASELINE PERIOD SENSITIVITY, historical heat and drought impacts ({start_year_analysis}–{end_year_analysis})")
print("=" * 90)
print(df_summary.to_string())

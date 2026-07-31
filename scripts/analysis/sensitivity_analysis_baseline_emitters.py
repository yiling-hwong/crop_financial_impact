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

    # ── Accumulators ──────────────────────────────────────────────────────────
    # topbot: top10, bot50
    dy_top_dfs, dp_top_dfs, degdp_top_dfs, de_top_dfs = [], [], [], []
    dy_bot_dfs, dp_bot_dfs, degdp_bot_dfs, de_bot_dfs = [], [], [], []
    # carbmaj
    dy_carb_dfs, dp_carb_dfs, degdp_carb_dfs, de_carb_dfs = [], [], [], []

    for crop in crops:
        base_topbot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot"
        base_carb   = f"{root_dir}/historical/linregress_outputs/{crop}/carbmaj"
        hist_fut    = f"hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"

        # topbot file paths
        for estr, dy_l, dp_l, degdp_l, de_l in [
            ("top10", dy_top_dfs, dp_top_dfs, degdp_top_dfs, de_top_dfs),
            ("bot50", dy_bot_dfs, dp_bot_dfs, degdp_bot_dfs, de_bot_dfs),
        ]:
            sfx = f"{estr}_{pred_str}_{hist_fut}"
            for fname, lst in [
                (f"{base_topbot}/dy_{sfx}",    dy_l),
                (f"{base_topbot}/dp_{sfx}",    dp_l),
                (f"{base_topbot}/degdp_{sfx}", degdp_l),
                (f"{base_topbot}/de_{sfx}",    de_l),
            ]:
                df = pd.read_csv(fname)
                df = df[df['country'].isin(valid_countries)]
                df["crop"] = crop
                lst.append(df)

        # carbmaj file paths
        sfx_carb = f"carbmaj_{pred_str}_{hist_fut}"
        for fname, lst in [
            (f"{base_carb}/dy_{sfx_carb}",    dy_carb_dfs),
            (f"{base_carb}/dp_{sfx_carb}",    dp_carb_dfs),
            (f"{base_carb}/degdp_{sfx_carb}", degdp_carb_dfs),
            (f"{base_carb}/de_{sfx_carb}",    de_carb_dfs),
        ]:
            df = pd.read_csv(fname)
            df = df[df['country'].isin(valid_countries)]
            df["crop"] = crop
            lst.append(df)

    # ── Concatenate across crops ───────────────────────────────────────────────
    df_dy_top    = pd.concat(dy_top_dfs,    ignore_index=True)
    df_dp_top    = pd.concat(dp_top_dfs,    ignore_index=True)
    df_degdp_top = pd.concat(degdp_top_dfs, ignore_index=True)
    df_de_top    = pd.concat(de_top_dfs,    ignore_index=True)

    df_dy_bot    = pd.concat(dy_bot_dfs,    ignore_index=True)
    df_dp_bot    = pd.concat(dp_bot_dfs,    ignore_index=True)
    df_degdp_bot = pd.concat(degdp_bot_dfs, ignore_index=True)
    df_de_bot    = pd.concat(de_bot_dfs,    ignore_index=True)

    df_dy_carb    = pd.concat(dy_carb_dfs,    ignore_index=True)
    df_dp_carb    = pd.concat(dp_carb_dfs,    ignore_index=True)
    df_degdp_carb = pd.concat(degdp_carb_dfs, ignore_index=True)
    df_de_carb    = pd.concat(de_carb_dfs,    ignore_index=True)

    # ── topbot top10 timeseries ───────────────────────────────────────────────
    dy_top_ts    = utils.compute_dy_weighted(df_dy_top, df_dp_top, year_cols_all, prod_weights=prod_weights, weight_opt=weight_opt)
    dp_top_ts    = (df_dp_top[year_cols_all].sum(axis=0) / 1e6).values
    de_top_ts    = (df_de_top[year_cols_all].sum(axis=0) / 1e9).values

    degdp_top_ts = utils.compute_degdp_weighted(
        df_degdp_top[["country"] + year_cols_all],
        year_cols_all,
        gdp_weights=gdp_weights, weight_opt=weight_opt
    )
    degdp_bot_ts = utils.compute_degdp_weighted(
        df_degdp_bot[["country"] + year_cols_all],
        year_cols_all,
        gdp_weights=gdp_weights, weight_opt=weight_opt
    )

    # ── topbot top10 regional timeseries (LDC / developed) ───────────────────
    df_degdp_top_ipcc = df_degdp_top.merge(df_ipcc, on="country", how="left")
    df_de_top_ipcc    = df_de_top.merge(df_ipcc, on="country", how="left")

    degdp_top_ldc_ts = (df_degdp_top_ipcc[df_degdp_top_ipcc[ar6_region] == "ldc"]
                        .groupby("country")[year_cols_all].mean()
                        .mean(axis=0).values)
    degdp_top_dev_ts = (df_degdp_top_ipcc[df_degdp_top_ipcc[ar6_region] == "developed"]
                        .groupby("country")[year_cols_all].mean()
                        .mean(axis=0).values)

    # ── carbmaj global timeseries ─────────────────────────────────────────────
    dy_carb_ts    = utils.compute_dy_weighted(df_dy_carb, df_dp_carb, year_cols_all, prod_weights=prod_weights, weight_opt=weight_opt)
    dp_carb_ts    = (df_dp_carb[year_cols_all].sum(axis=0) / 1e6).values
    degdp_carb_ts = utils.compute_degdp_weighted(
        df_degdp_carb[["country"] + year_cols_all],
        year_cols_all,
        gdp_weights=gdp_weights, weight_opt=weight_opt
    )
    de_carb_ts    = (df_de_carb[year_cols_all].sum(axis=0) / 1e9).values

    # ── carbmaj regional timeseries (LDC / developed) ─────────────────────────
    df_degdp_carb_ipcc = df_degdp_carb.merge(df_ipcc, on="country", how="left")
    df_de_carb_ipcc    = df_de_carb.merge(df_ipcc, on="country", how="left")

    degdp_carb_ldc_ts = utils.compute_degdp_weighted(
        df_degdp_carb_ipcc[df_degdp_carb_ipcc[ar6_region] == "ldc"][["country"] + year_cols_all],
        year_cols_all, gdp_weights=gdp_weights, weight_opt=weight_opt)
    degdp_carb_dev_ts = utils.compute_degdp_weighted(
        df_degdp_carb_ipcc[df_degdp_carb_ipcc[ar6_region] == "developed"][["country"] + year_cols_all],
        year_cols_all, gdp_weights=gdp_weights, weight_opt=weight_opt)

    ts_data[period_label] = {
        "dy_top10":        dy_top_ts,
        "dp_top10":        dp_top_ts,
        "degdp_top10":     degdp_top_ts,
        "de_top10":        de_top_ts,
        "degdp_top10_ldc": degdp_top_ldc_ts,
        "degdp_top10_dev": degdp_top_dev_ts,
        "degdp_bot50":     degdp_bot_ts,
        "dy_carb":         dy_carb_ts,
        "dp_carb":         dp_carb_ts,
        "degdp_carb":      degdp_carb_ts,
        "de_carb":         de_carb_ts,
        "degdp_carb_ldc":  degdp_carb_ldc_ts,
        "degdp_carb_dev":  degdp_carb_dev_ts,
    }

"""
SUMMARY TABLE (mean/sum over study period)
"""
def period_mean(arr): return float(np.mean(arr[start_idx:end_idx]))
def period_sum(arr):  return float(np.sum(arr[start_idx:end_idx]))

def make_row(label, key, agg_fn):
    row = {"metric": label}
    for lbl in period_labels:
        row[lbl] = round(agg_fn(ts_data[lbl][key]), 6)
    return row

def make_ratio_row(label, key_num, key_den):
    row = {"metric": label}
    for lbl in period_labels:
        num = period_mean(ts_data[lbl][key_num])
        den = period_mean(ts_data[lbl][key_den])
        row[lbl] = round(num / den if den != 0 else float('nan'), 4)
    return row

rows = [
    # topbot top10 — global
    make_row("dy top10 [%]",              "dy_top10",        period_mean),
    make_row("degdp top10 [%]", "degdp_top10", period_mean),
    make_row("dp top10 [Mt]",             "dp_top10",        period_sum),
    make_row("de top10 [B US$]",          "de_top10",        period_sum),
    # topbot top10 — regional
    make_row("degdp top10 LDC [%]",       "degdp_top10_ldc", period_mean),
    make_row("degdp top10 developed [%]", "degdp_top10_dev", period_mean),
    # carbmaj — global
    make_row("dy carbmaj global [%]",     "dy_carb",       period_mean),
    make_row("degdp carbmaj global [%]", "degdp_carb", period_mean),
    make_row("dp carbmaj global [Mt]",    "dp_carb",       period_sum),
    make_row("de carbmaj global [B US$]", "de_carb",       period_sum),
    # carbmaj — regional
    make_row("degdp carbmaj LDC [%]",        "degdp_carb_ldc", period_mean),
    make_row("degdp carbmaj developed [%]",  "degdp_carb_dev", period_mean),
]

df_summary = pd.DataFrame(rows).set_index("metric")
print()
print("=" * 90)
print(f"BASELINE PERIOD SENSITIVITY, emitter-linked impacts ({start_year_analysis}–{end_year_analysis})")
print("=" * 90)
print(df_summary.to_string())
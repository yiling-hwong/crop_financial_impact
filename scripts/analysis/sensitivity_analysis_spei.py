# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('..')
import utils

"""
PARAMETERS
"""
crops     = ["maize","wheat","soy"]
pred_str = "t3s3"
spei_months = ["03", "06", "09", "12"]
weight_opt = "weighted"  # "weighted", "simple"

start_year_hist = 1974
end_year_hist   = 2004
start_year_fut  = 1990
end_year_fut    = 2019
start_year_analysis = 2007
end_year_analysis = 2019

root_dir = f'../../data'
hist_dir = f"{root_dir}/historical/linregress_outputs"

impacts   = ["dy", "dp", "degdp", "de"]
spei_labels = {"03": "SPEI-3", "06": "SPEI-6", "09": "SPEI-9", "12": "SPEI-12"}

"""
LOAD SUPPORT DATA
"""
hist_file      = f"{root_dir}/historical/linregress_inputs/INPUT_HISTORICAL_DETRENDED_DEMEANED_spei03.csv"
valid_countries = pd.read_csv(hist_file)["country"].unique()

prod_weights = utils.load_fao_prod_weights(root_dir, "03")
gdp_weights  = utils.load_gdp_weights(root_dir)

ar6_region = "region_ar6_dev"
df_ipcc    = utils.get_ipcc_region_df()
df_ipcc    = df_ipcc[["Country", ar6_region]].rename(columns={"Country": "country"})

years_ts      = list(range(start_year_fut, end_year_fut + 1))
years_ts_str  = [str(y) for y in years_ts]
years_plot    = list(range(start_year_analysis, end_year_analysis + 1))
start_idx = years_ts.index(start_year_analysis)
end_idx   = years_ts.index(end_year_analysis) + 1

"""
LOAD DATA AND COMPUTE GLOBAL TIMESERIES
"""
ts_store = {}  # spei_month -> {impact -> array}

for spei_month in spei_months:
    suffix = (f"isimip3a_{pred_str}"
              f"_hist{start_year_hist}-{end_year_hist}"
              f"_fut{start_year_fut}-{end_year_fut}"
              f"_spei{spei_month}.csv")

    dy_dfs, dp_dfs, degdp_dfs, de_dfs = [], [], [], []
    for crop in crops:
        base = f"{hist_dir}/{crop}/isimip3a"
        for fname, lst in [
            (f"{base}/dy_{suffix}",    dy_dfs),
            (f"{base}/dp_{suffix}",    dp_dfs),
            (f"{base}/degdp_{suffix}", degdp_dfs),
            (f"{base}/de_{suffix}",    de_dfs),
        ]:
            df = pd.read_csv(fname)
            df = df[df["country"].isin(valid_countries)]
            df["crop"] = crop
            lst.append(df)

    df_dy    = pd.concat(dy_dfs,    ignore_index=True)
    df_dp    = pd.concat(dp_dfs,    ignore_index=True)
    df_degdp = pd.concat(degdp_dfs, ignore_index=True)
    df_de    = pd.concat(de_dfs,    ignore_index=True)

    # dy: FAO production-weighted
    dy_ts = utils.compute_dy_weighted(df_dy, df_dp, years_ts_str, prod_weights=prod_weights, weight_opt=weight_opt)

    # dp: flat sum across all country rows -> Mt
    dp_ts = df_dp[years_ts_str].sum().values / 1e6

    # degdp: World Bank GDP-weighted mean
    degdp_ts = utils.compute_degdp_weighted(df_degdp, years_ts_str, gdp_weights=gdp_weights, weight_opt=weight_opt)

    # de: flat sum -> B US$
    de_ts = df_de[years_ts_str].sum().values / 1e9

    # degdp LDC / developed
    df_degdp_ipcc = df_degdp.merge(df_ipcc, on="country", how="left")
    df_de_ipcc    = df_de.merge(df_ipcc, on="country", how="left")

    degdp_ldc_ts = utils.compute_degdp_weighted(
        df_degdp_ipcc[df_degdp_ipcc[ar6_region] == "ldc"][["country"] + years_ts_str],
        years_ts_str,
        gdp_weights=gdp_weights, weight_opt=weight_opt
    )
    degdp_dev_ts = utils.compute_degdp_weighted(
        df_degdp_ipcc[df_degdp_ipcc[ar6_region] == "developed"][["country"] + years_ts_str],
        years_ts_str,
        gdp_weights=gdp_weights, weight_opt=weight_opt
    )

    ts_store[spei_month] = {
        "dy":         dy_ts,
        "dp":         dp_ts,
        "degdp":      degdp_ts,
        "de":         de_ts,
        "degdp_ldc":  degdp_ldc_ts,
        "degdp_dev":  degdp_dev_ts,
    }

"""
DIAGNOSTICS TABLE: mean/sum over study period
"""
print()
print("=" * 80)
print(f"SPEI SENSITIVITY ({start_year_analysis}–{end_year_analysis})")
print("=" * 80)

def diag_row(label, key, fn):
    row = {"metric": label}
    for spei_month in spei_months:
        sl = ts_store[spei_month][key][start_idx:end_idx]
        row[spei_labels[spei_month]] = round(float(fn(sl)), 6)
    return row

rows = [
    diag_row("DY MEAN Global [%]",       "dy",        np.mean),
    diag_row("DEGDP MEAN Global [%]",     "degdp",     np.mean),
    diag_row("DP SUM Global [Mt]",        "dp",        np.sum),
    diag_row("DE SUM Global [B US$]",     "de",        np.sum),
    diag_row("DEGDP MEAN LDC [%]",        "degdp_ldc", np.mean),
    diag_row("DEGDP MEAN Developed [%]",  "degdp_dev", np.mean),
]

df_diag = pd.DataFrame(rows).set_index("metric")
print(df_diag.to_string())
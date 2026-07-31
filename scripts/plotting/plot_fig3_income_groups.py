# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import geopandas as gpd
import cartopy.crs as ccrs
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('../')
import utils

"""
PARAMETERS
"""
crops = ["maize", "wheat", "soy"]
estring_top = "top10"
estring_bot = "bot50"
estring_tot = "total"
pred_str = "t3s3"
pred_str_irr = f"{pred_str}_irr"
spei_month = "03"
weight_opt = "weighted"  # "weighted", "simple"

###################### PLOTTING PARAMETERS ######################
# Maps parameter
cmap_dy = "Reds_r"
cmap_dp = "copper"
cmap_de = utils.get_truncated_cmap("RdPu_r")
cmap_degdp = utils.get_truncated_cmap("YlGnBu_r")
vmin_dy = -2.5
vmax_dy = 0.1
vmin_dp = -100
vmax_dp = -0.1
ticks_dp = [-0.1, -1, -10, -100]

vmin_degdp = -0.008
vmax_degdp = -0.0005
vmin_de = -10.0
vmax_de = -0.01
ticks_de = [-0.01, -0.01, -0.1, -1.0, -10.0]
start_year_avg = 2007
end_year_avg = 2019

label_dy = "Yield change [%]"
label_dp = "Production change [Mt]"
label_de = "Economic loss [B US$]"
label_degdp = "GDP loss [%]"

# Timeseries parameters
alpha_irr = 1.0
lw_irr = 1.0
ylim_top_dy = 0.2
ylim_bottom_dy = -2.5
ylim_top_dp = 3.0
ylim_bottom_dp = -45
ylim_top_degdp = 0.002
ylim_bottom_degdp = -0.006
ylim_top_de = 0.5
ylim_bottom_de = -11
start_year_plot = 2007
end_year_plot = 2019

label_alphabets = utils.generate_alphabet_list(8, option="lower")
label_alphabets = [x for x in label_alphabets]

window_size = 1
smooth_method = "numpy"
np_mode = "full"

year_str = [str(x) for x in range(start_year_plot, end_year_plot + 1)]
years_plot = [x for x in range(start_year_plot, end_year_plot + 1)]

top10_color = "darkblue"
bot50_color = "darkred"

quantile_low = 0.25
quantile_high = 0.75
##################################################################

ar6_region = "region_ar6_dev"
regions = ["ldc", "developing", "developed"]

start_year_hist = 1974
end_year_hist = 2004
start_year_fut = 1990
end_year_fut = 2019

root_dir = '../../data'
country_shape_file = f"{root_dir}/resources/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

df_dy_top_all = []
df_dp_top_all = []
df_dy_bot_all = []
df_dp_bot_all = []
df_dy_tot_all = []
df_dp_tot_all = []
df_degdp_top_all = []
df_de_top_all = []
df_degdp_bot_all = []
df_de_bot_all = []
df_degdp_tot_all = []
df_de_tot_all = []

df_dy_top_all_irr = []
df_dp_top_all_irr = []
df_dy_bot_all_irr = []
df_dp_bot_all_irr = []
df_dy_tot_all_irr = []
df_dp_tot_all_irr = []
df_degdp_top_all_irr = []
df_de_top_all_irr = []
df_degdp_bot_all_irr = []
df_de_bot_all_irr = []
df_degdp_tot_all_irr = []
df_de_tot_all_irr = []

# Load data for all crops
for crop in crops:

    # Files for top10
    dy_file_top = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dy_{estring_top}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    dp_file_top = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dp_{estring_top}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    degdp_file_top = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/degdp_{estring_top}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    de_file_top = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/de_{estring_top}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    dy_file_top_irr = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dy_{estring_top}_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    dp_file_top_irr = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dp_{estring_top}_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    degdp_file_top_irr = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/degdp_{estring_top}_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    de_file_top_irr = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/de_{estring_top}_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"

    # Files for bot50
    dy_file_bot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dy_{estring_bot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    dp_file_bot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dp_{estring_bot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    degdp_file_bot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/degdp_{estring_bot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    de_file_bot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/de_{estring_bot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    dy_file_bot_irr = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dy_{estring_bot}_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    dp_file_bot_irr = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dp_{estring_bot}_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    degdp_file_bot_irr = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/degdp_{estring_bot}_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    de_file_bot_irr = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/de_{estring_bot}_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"

    # Files for total
    dy_file_tot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dy_{estring_tot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    dp_file_tot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dp_{estring_tot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    degdp_file_tot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/degdp_{estring_tot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    de_file_tot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/de_{estring_tot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    dy_file_tot_irr = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dy_{estring_tot}_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    dp_file_tot_irr = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dp_{estring_tot}_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    degdp_file_tot_irr = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/degdp_{estring_tot}_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    de_file_tot_irr = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/de_{estring_tot}_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"

    # Read Data
    df_dy_top = pd.read_csv(dy_file_top)
    df_dp_top = pd.read_csv(dp_file_top)
    df_degdp_top = pd.read_csv(degdp_file_top)
    df_de_top = pd.read_csv(de_file_top)

    df_dy_bot = pd.read_csv(dy_file_bot)
    df_dp_bot = pd.read_csv(dp_file_bot)
    df_degdp_bot = pd.read_csv(degdp_file_bot)
    df_de_bot = pd.read_csv(de_file_bot)

    df_dy_tot = pd.read_csv(dy_file_tot)
    df_dp_tot = pd.read_csv(dp_file_tot)
    df_degdp_tot = pd.read_csv(degdp_file_tot)
    df_de_tot = pd.read_csv(de_file_tot)

    # Read irr Data
    df_dy_top_irr = pd.read_csv(dy_file_top_irr)
    df_dp_top_irr = pd.read_csv(dp_file_top_irr)
    df_degdp_top_irr = pd.read_csv(degdp_file_top_irr)
    df_de_top_irr = pd.read_csv(de_file_top_irr)

    df_dy_bot_irr = pd.read_csv(dy_file_bot_irr)
    df_dp_bot_irr = pd.read_csv(dp_file_bot_irr)
    df_degdp_bot_irr = pd.read_csv(degdp_file_bot_irr)
    df_de_bot_irr = pd.read_csv(de_file_bot_irr)

    df_dy_tot_irr = pd.read_csv(dy_file_tot_irr)
    df_dp_tot_irr = pd.read_csv(dp_file_tot_irr)
    df_degdp_tot_irr = pd.read_csv(degdp_file_tot_irr)
    df_de_tot_irr = pd.read_csv(de_file_tot_irr)

    # Add crop column
    df_dy_top['crop'] = crop
    df_dp_top['crop'] = crop
    df_degdp_top['crop'] = crop
    df_de_top['crop'] = crop

    df_dy_bot['crop'] = crop
    df_dp_bot['crop'] = crop
    df_degdp_bot['crop'] = crop
    df_de_bot['crop'] = crop

    df_dy_tot['crop'] = crop
    df_dp_tot['crop'] = crop
    df_degdp_tot['crop'] = crop
    df_de_tot['crop'] = crop

    # Add crop column (irr)
    df_dy_top_irr['crop'] = crop
    df_dp_top_irr['crop'] = crop
    df_degdp_top_irr['crop'] = crop
    df_de_top_irr['crop'] = crop

    df_dy_bot_irr['crop'] = crop
    df_dp_bot_irr['crop'] = crop
    df_degdp_bot_irr['crop'] = crop
    df_de_bot_irr['crop'] = crop

    df_dy_tot_irr['crop'] = crop
    df_dp_tot_irr['crop'] = crop
    df_degdp_tot_irr['crop'] = crop
    df_de_tot_irr['crop'] = crop

    # Append to lists
    df_dy_top_all.append(df_dy_top)
    df_dp_top_all.append(df_dp_top)
    df_dy_bot_all.append(df_dy_bot)
    df_dp_bot_all.append(df_dp_bot)
    df_dy_tot_all.append(df_dy_tot)
    df_dp_tot_all.append(df_dp_tot)

    df_degdp_top_all.append(df_degdp_top)
    df_de_top_all.append(df_de_top)
    df_degdp_bot_all.append(df_degdp_bot)
    df_de_bot_all.append(df_de_bot)
    df_degdp_tot_all.append(df_degdp_tot)
    df_de_tot_all.append(df_de_tot)

    # Append to irr lists
    df_dy_top_all_irr.append(df_dy_top_irr)
    df_dp_top_all_irr.append(df_dp_top_irr)
    df_dy_bot_all_irr.append(df_dy_bot_irr)
    df_dp_bot_all_irr.append(df_dp_bot_irr)
    df_dy_tot_all_irr.append(df_dy_tot_irr)
    df_dp_tot_all_irr.append(df_dp_tot_irr)

    df_degdp_top_all_irr.append(df_degdp_top_irr)
    df_de_top_all_irr.append(df_de_top_irr)
    df_degdp_bot_all_irr.append(df_degdp_bot_irr)
    df_de_bot_all_irr.append(df_de_bot_irr)
    df_degdp_tot_all_irr.append(df_degdp_tot_irr)
    df_de_tot_all_irr.append(df_de_tot_irr)

# Combine all crops data
df_dy_top = pd.concat(df_dy_top_all, ignore_index=True)
df_dp_top = pd.concat(df_dp_top_all, ignore_index=True)
df_dy_bot = pd.concat(df_dy_bot_all, ignore_index=True)
df_dp_bot = pd.concat(df_dp_bot_all, ignore_index=True)
df_dy_tot = pd.concat(df_dy_tot_all, ignore_index=True)
df_dp_tot = pd.concat(df_dp_tot_all, ignore_index=True)

df_degdp_top = pd.concat(df_degdp_top_all, ignore_index=True)
df_de_top = pd.concat(df_de_top_all, ignore_index=True)
df_degdp_bot = pd.concat(df_degdp_bot_all, ignore_index=True)
df_de_bot = pd.concat(df_de_bot_all, ignore_index=True)
df_degdp_tot = pd.concat(df_degdp_tot_all, ignore_index=True)
df_de_tot = pd.concat(df_de_tot_all, ignore_index=True)

# Combine all crops data (irr)
df_dy_top_irr = pd.concat(df_dy_top_all_irr, ignore_index=True)
df_dp_top_irr = pd.concat(df_dp_top_all_irr, ignore_index=True)
df_dy_bot_irr = pd.concat(df_dy_bot_all_irr, ignore_index=True)
df_dp_bot_irr = pd.concat(df_dp_bot_all_irr, ignore_index=True)
df_dy_tot_irr = pd.concat(df_dy_tot_all_irr, ignore_index=True)
df_dp_tot_irr = pd.concat(df_dp_tot_all_irr, ignore_index=True)

df_degdp_top_irr = pd.concat(df_degdp_top_all_irr, ignore_index=True)
df_de_top_irr = pd.concat(df_de_top_all_irr, ignore_index=True)
df_degdp_bot_irr = pd.concat(df_degdp_bot_all_irr, ignore_index=True)
df_de_bot_irr = pd.concat(df_de_bot_all_irr, ignore_index=True)
df_degdp_tot_irr = pd.concat(df_degdp_tot_all_irr, ignore_index=True)
df_de_tot_irr = pd.concat(df_de_tot_all_irr, ignore_index=True)

# Get valid countries
hist_file = f"{root_dir}/historical/linregress_inputs/INPUT_HISTORICAL_DETRENDED_DEMEANED_spei{spei_month}.csv"
df_hist = pd.read_csv(hist_file)
valid_countries = df_hist["country"].unique()

# Filter for valid countries
for df in [df_dy_top, df_dp_top, df_dy_bot, df_dp_bot, df_dy_tot, df_dp_tot,
           df_degdp_top, df_de_top, df_degdp_bot, df_de_bot, df_degdp_tot, df_de_tot,
           df_dy_top_irr, df_dp_top_irr, df_dy_bot_irr, df_dp_bot_irr, df_dy_tot_irr, df_dp_tot_irr,
           df_degdp_top_irr, df_de_top_irr, df_degdp_bot_irr, df_de_bot_irr, df_degdp_tot_irr, df_de_tot_irr]:
    df.drop(df[~df['country'].isin(valid_countries)].index, inplace=True)

_yr_cols_full = [str(x) for x in range(start_year_fut, end_year_fut + 1)]
_years_full = list(range(start_year_fut, end_year_fut + 1))
plot_start_idx = _years_full.index(start_year_plot)
plot_end_idx = _years_full.index(end_year_plot) + 1

# Load weights
prod_weights = utils.load_fao_prod_weights(root_dir, spei_month)
gdp_weights  = utils.load_gdp_weights(root_dir)

cols_ts_dy = ["country", "crop"] + _yr_cols_full
cols_ts_dp = ["country"] + _yr_cols_full
dy_top_ts     = utils.compute_dy_weighted(df_dy_top[cols_ts_dy],     df_dp_top[cols_ts_dp],     _yr_cols_full, prod_weights=prod_weights, weight_opt=weight_opt)
dy_bot_ts     = utils.compute_dy_weighted(df_dy_bot[cols_ts_dy],     df_dp_bot[cols_ts_dp],     _yr_cols_full, prod_weights=prod_weights, weight_opt=weight_opt)
dy_tot_ts     = utils.compute_dy_weighted(df_dy_tot[cols_ts_dy],     df_dp_tot[cols_ts_dp],     _yr_cols_full, prod_weights=prod_weights, weight_opt=weight_opt)
dy_top_irr_ts = utils.compute_dy_weighted(df_dy_top_irr[cols_ts_dy], df_dp_top_irr[cols_ts_dp], _yr_cols_full, prod_weights=prod_weights, weight_opt=weight_opt)
dy_bot_irr_ts = utils.compute_dy_weighted(df_dy_bot_irr[cols_ts_dy], df_dp_bot_irr[cols_ts_dp], _yr_cols_full, prod_weights=prod_weights, weight_opt=weight_opt)
dy_tot_irr_ts = utils.compute_dy_weighted(df_dy_tot_irr[cols_ts_dy], df_dp_tot_irr[cols_ts_dp], _yr_cols_full, prod_weights=prod_weights, weight_opt=weight_opt)

# For maps: calculate averages over specified years
year_str_avg = [str(x) for x in range(start_year_avg, end_year_avg + 1)]
cols_avg = ["country", "crop"] + year_str_avg

# Process top10 data
dy_top = df_dy_top[cols_avg]
dp_top = df_dp_top[cols_avg]
degdp_top = df_degdp_top[cols_avg]
de_top = df_de_top[cols_avg]

dy_top["dy_avg"] = dy_top.loc[:, f"{start_year_avg}":f"{end_year_avg}"].mean(axis=1)
dp_top[f"dp_sum"] = dp_top.loc[:, f"{start_year_avg}":f"{end_year_avg}"].sum(axis=1) / 1e6
degdp_top["degdp_avg"] = degdp_top.loc[:, f"{start_year_avg}":f"{end_year_avg}"].mean(axis=1)
de_top[f"de_sum"] = de_top.loc[:, f"{start_year_avg}":f"{end_year_avg}"].sum(axis=1) / 1e9

# Process bot50 data
dy_bot = df_dy_bot[cols_avg]
dp_bot = df_dp_bot[cols_avg]
degdp_bot = df_degdp_bot[cols_avg]
de_bot = df_de_bot[cols_avg]

dy_bot["dy_avg"] = dy_bot.loc[:, f"{start_year_avg}":f"{end_year_avg}"].mean(axis=1)
dp_bot[f"dp_sum"] = dp_bot.loc[:, f"{start_year_avg}":f"{end_year_avg}"].sum(axis=1) / 1e6
degdp_bot["degdp_avg"] = degdp_bot.loc[:, f"{start_year_avg}":f"{end_year_avg}"].mean(axis=1)
de_bot[f"de_sum"] = de_bot.loc[:, f"{start_year_avg}":f"{end_year_avg}"].sum(axis=1) / 1e9

# Process TOTAL data
dy_tot = df_dy_tot[cols_avg]
dp_tot = df_dp_tot[cols_avg]
degdp_tot = df_degdp_tot[cols_avg]
de_tot = df_de_tot[cols_avg]

dy_tot["dy_avg"] = dy_tot.loc[:, f"{start_year_avg}":f"{end_year_avg}"].mean(axis=1)
dp_tot[f"dp_sum"] = dp_tot.loc[:, f"{start_year_avg}":f"{end_year_avg}"].sum(axis=1) / 1e6
degdp_tot["degdp_avg"] = degdp_tot.loc[:, f"{start_year_avg}":f"{end_year_avg}"].mean(axis=1)
de_tot[f"de_sum"] = de_tot.loc[:, f"{start_year_avg}":f"{end_year_avg}"].sum(axis=1) / 1e9

# Aggregate across crops per country before computing diffs
# dy/degdp: mean across crops; dp/de: sum across crops
_dy_top_map    = dy_top.groupby("country")["dy_avg"].mean().reset_index()
_dy_bot_map    = dy_bot.groupby("country")["dy_avg"].mean().reset_index()
_dp_top_map    = dp_top.groupby("country")["dp_sum"].sum().reset_index()
_dp_bot_map    = dp_bot.groupby("country")["dp_sum"].sum().reset_index()
_degdp_top_map = degdp_top.groupby("country")["degdp_avg"].mean().reset_index()
_degdp_bot_map = degdp_bot.groupby("country")["degdp_avg"].mean().reset_index()
_de_top_map    = de_top.groupby("country")["de_sum"].sum().reset_index()
_de_bot_map    = de_bot.groupby("country")["de_sum"].sum().reset_index()

df_dy_diff = pd.merge(_dy_top_map, _dy_bot_map, on="country", suffixes=('_top', '_bot'))
df_dy_diff["dy_diff"] = df_dy_diff["dy_avg_top"] - df_dy_diff["dy_avg_bot"]

df_dp_diff = pd.merge(_dp_top_map, _dp_bot_map, on="country", suffixes=('_top', '_bot'))
df_dp_diff["dp_sum_diff"] = df_dp_diff["dp_sum_top"] - df_dp_diff["dp_sum_bot"]

df_degdp_diff = pd.merge(_degdp_top_map, _degdp_bot_map, on="country", suffixes=('_top', '_bot'))
df_degdp_diff["degdp_diff"] = df_degdp_diff["degdp_avg_top"] - df_degdp_diff["degdp_avg_bot"]

df_de_diff = pd.merge(_de_top_map, _de_bot_map, on="country", suffixes=('_top', '_bot'))
df_de_diff["de_sum_diff"] = df_de_diff["de_sum_top"] - df_de_diff["de_sum_bot"]

"""
IPCC REGIONS
"""
print("\nGetting IPCC region:", ar6_region)
df_ipcc = utils.get_ipcc_region_df()
df_ipcc = df_ipcc[["Country", ar6_region]]
df_ipcc = df_ipcc.rename(columns={"Country": "country"})

#################### Merge with IPCC regions
# --- TOP10
dy_ipcc_top = df_dy_top.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
dp_ipcc_top = df_dp_top.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
degdp_ipcc_top = df_degdp_top.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
de_ipcc_top = df_de_top.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
# --- BOT50
dy_ipcc_bot = df_dy_bot.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
dp_ipcc_bot = df_dp_bot.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
degdp_ipcc_bot = df_degdp_bot.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
de_ipcc_bot = df_de_bot.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
# --- TOTAL
dy_ipcc_tot = df_dy_tot.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
dp_ipcc_tot = df_dp_tot.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
degdp_ipcc_tot = df_degdp_tot.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
de_ipcc_tot = df_de_tot.merge(df_ipcc[["country", ar6_region]], on="country", how="left")

#################### Calculate regional means/sums for timeseries
# --- TOP10
dy_ipcc_melted_top = dy_ipcc_top.melt(id_vars=['country', "crop", ar6_region, ], var_name='year', value_name='value')
dp_ipcc_melted_top = dp_ipcc_top.melt(id_vars=['country', "countryname", "crop", ar6_region], var_name='year',
                                      value_name='value')
degdp_ipcc_melted_top = degdp_ipcc_top.melt(id_vars=['country', "crop", ar6_region], var_name='year',
                                            value_name='value')
de_ipcc_melted_top = de_ipcc_top.melt(id_vars=['country', "crop", ar6_region], var_name='year', value_name='value')

_ts_yr_cols = _yr_cols_full
dy_ipcc_stat_top = pd.DataFrame([
    {"region": r, "year": yr, "value": v}
    for r in regions
    for yr, v in zip(_ts_yr_cols, utils.compute_dy_weighted(
        dy_ipcc_top[dy_ipcc_top[ar6_region] == r],
        dp_ipcc_top[dp_ipcc_top[ar6_region] == r],
        _ts_yr_cols,
        prod_weights=prod_weights, weight_opt=weight_opt))])
degdp_ipcc_stat_top = pd.DataFrame([
    {"region": r, "year": yr, "value": v}
    for r in regions
    for yr, v in zip(_ts_yr_cols, utils.compute_degdp_weighted(
        degdp_ipcc_top[degdp_ipcc_top[ar6_region] == r],
        _ts_yr_cols,
        gdp_weights=gdp_weights, weight_opt=weight_opt))])
dp_ipcc_stat_top = dp_ipcc_melted_top.groupby([ar6_region, 'year'], as_index=False)['value'].sum()
dp_ipcc_stat_top.rename(columns={ar6_region: 'region'}, inplace=True)
de_ipcc_stat_top = de_ipcc_melted_top.groupby([ar6_region, 'year'], as_index=False)['value'].sum()
de_ipcc_stat_top.rename(columns={ar6_region: 'region'}, inplace=True)

dy_ipcc_region_top = \
dy_ipcc_stat_top.assign(year=dy_ipcc_stat_top['year'].astype(int)).query(f"{start_year_plot} <= year <= {end_year_plot}").groupby(
    "region", as_index=False)["value"].mean()
degdp_ipcc_region_top = degdp_ipcc_stat_top.assign(year=degdp_ipcc_stat_top['year'].astype(int)).query(
    f"{start_year_plot} <= year <= {end_year_plot}").groupby("region", as_index=False)["value"].mean()
dp_ipcc_region_top = \
dp_ipcc_stat_top.assign(year=dp_ipcc_stat_top['year'].astype(int)).query(f"{start_year_plot} <= year <= {end_year_plot}").groupby(
    "region", as_index=False)["value"].sum()
de_ipcc_region_top = \
de_ipcc_stat_top.assign(year=de_ipcc_stat_top['year'].astype(int)).query(f"{start_year_plot} <= year <= {end_year_plot}").groupby(
    "region", as_index=False)["value"].sum()

# --- BOT50
dy_ipcc_melted_bot = dy_ipcc_bot.melt(id_vars=['country', "crop", ar6_region, ], var_name='year', value_name='value')
dp_ipcc_melted_bot = dp_ipcc_bot.melt(id_vars=['country', "countryname", "crop", ar6_region], var_name='year',
                                      value_name='value')
degdp_ipcc_melted_bot = degdp_ipcc_bot.melt(id_vars=['country', "crop", ar6_region], var_name='year',
                                            value_name='value')
de_ipcc_melted_bot = de_ipcc_bot.melt(id_vars=['country', "crop", ar6_region], var_name='year', value_name='value')

dy_ipcc_stat_bot = pd.DataFrame([
    {"region": r, "year": yr, "value": v}
    for r in regions
    for yr, v in zip(_ts_yr_cols, utils.compute_dy_weighted(
        dy_ipcc_bot[dy_ipcc_bot[ar6_region] == r],
        dp_ipcc_bot[dp_ipcc_bot[ar6_region] == r],
        _ts_yr_cols,
        prod_weights=prod_weights, weight_opt=weight_opt))])
degdp_ipcc_stat_bot = pd.DataFrame([
    {"region": r, "year": yr, "value": v}
    for r in regions
    for yr, v in zip(_ts_yr_cols, utils.compute_degdp_weighted(
        degdp_ipcc_bot[degdp_ipcc_bot[ar6_region] == r],
        _ts_yr_cols,
        gdp_weights=gdp_weights, weight_opt=weight_opt))])
dp_ipcc_stat_bot = dp_ipcc_melted_bot.groupby([ar6_region, 'year'], as_index=False)['value'].sum()
dp_ipcc_stat_bot.rename(columns={ar6_region: 'region'}, inplace=True)
de_ipcc_stat_bot = de_ipcc_melted_bot.groupby([ar6_region, 'year'], as_index=False)['value'].sum()
de_ipcc_stat_bot.rename(columns={ar6_region: 'region'}, inplace=True)

dy_ipcc_region_bot = \
dy_ipcc_stat_bot.assign(year=dy_ipcc_stat_bot['year'].astype(int)).query(f"{start_year_plot} <= year <= {end_year_plot}").groupby(
    "region", as_index=False)["value"].mean()
degdp_ipcc_region_bot = degdp_ipcc_stat_bot.assign(year=degdp_ipcc_stat_bot['year'].astype(int)).query(
    f"{start_year_plot} <= year <= {end_year_plot}").groupby("region", as_index=False)["value"].mean()
dp_ipcc_region_bot = \
dp_ipcc_stat_bot.assign(year=dp_ipcc_stat_bot['year'].astype(int)).query(f"{start_year_plot} <= year <= {end_year_plot}").groupby(
    "region", as_index=False)["value"].sum()
de_ipcc_region_bot = \
de_ipcc_stat_bot.assign(year=de_ipcc_stat_bot['year'].astype(int)).query(f"{start_year_plot} <= year <= {end_year_plot}").groupby(
    "region", as_index=False)["value"].sum()

# ---- TOTAL
dy_ipcc_melted_tot = dy_ipcc_tot.melt(id_vars=['country', "crop", ar6_region, ], var_name='year', value_name='value')
dp_ipcc_melted_tot = dp_ipcc_tot.melt(id_vars=['country', "countryname", "crop", ar6_region], var_name='year',
                                      value_name='value')
degdp_ipcc_melted_tot = degdp_ipcc_tot.melt(id_vars=['country', "crop", ar6_region], var_name='year',
                                            value_name='value')
de_ipcc_melted_tot = de_ipcc_tot.melt(id_vars=['country', "crop", ar6_region], var_name='year', value_name='value')

dy_ipcc_stat_tot = pd.DataFrame([
    {"region": r, "year": yr, "value": v}
    for r in regions
    for yr, v in zip(_ts_yr_cols, utils.compute_dy_weighted(
        dy_ipcc_tot[dy_ipcc_tot[ar6_region] == r],
        dp_ipcc_tot[dp_ipcc_tot[ar6_region] == r],
        _ts_yr_cols,
        prod_weights=prod_weights, weight_opt=weight_opt))])
degdp_ipcc_stat_tot = pd.DataFrame([
    {"region": r, "year": yr, "value": v}
    for r in regions
    for yr, v in zip(_ts_yr_cols, utils.compute_degdp_weighted(
        degdp_ipcc_tot[degdp_ipcc_tot[ar6_region] == r],
        _ts_yr_cols,
        gdp_weights=gdp_weights, weight_opt=weight_opt))])
dp_ipcc_stat_tot = dp_ipcc_melted_tot.groupby([ar6_region, 'year'], as_index=False)['value'].sum()
dp_ipcc_stat_tot.rename(columns={ar6_region: 'region'}, inplace=True)
de_ipcc_stat_tot = de_ipcc_melted_tot.groupby([ar6_region, 'year'], as_index=False)['value'].sum()
de_ipcc_stat_tot.rename(columns={ar6_region: 'region'}, inplace=True)

dy_ipcc_region_tot = \
dy_ipcc_stat_tot.assign(year=dy_ipcc_stat_tot['year'].astype(int)).query(f"{start_year_plot} <= year <= {end_year_plot}").groupby(
    "region", as_index=False)["value"].mean()
degdp_ipcc_region_tot = degdp_ipcc_stat_tot.assign(year=degdp_ipcc_stat_tot['year'].astype(int)).query(
    f"{start_year_plot} <= year <= {end_year_plot}").groupby("region", as_index=False)["value"].mean()
dp_ipcc_region_tot = \
dp_ipcc_stat_tot.assign(year=dp_ipcc_stat_tot['year'].astype(int)).query(f"{start_year_plot} <= year <= {end_year_plot}").groupby(
    "region", as_index=False)["value"].sum()
de_ipcc_region_tot = \
de_ipcc_stat_tot.assign(year=de_ipcc_stat_tot['year'].astype(int)).query(f"{start_year_plot} <= year <= {end_year_plot}").groupby(
    "region", as_index=False)["value"].sum()

############# Combine top10, bot50, total into single DataFrame
dp_ipcc_region_all = dp_ipcc_region_top[['region', 'value']].rename(columns={'value': 'top10'}) \
    .merge(dp_ipcc_region_bot[['region', 'value']].rename(columns={'value': 'bot50'}), on='region') \
    .merge(dp_ipcc_region_tot[['region', 'value']].rename(columns={'value': 'total'}), on='region')

dp_ipcc_region_all.iloc[:, 1:] /= 1e6

de_ipcc_region_all = de_ipcc_region_top[['region', 'value']].rename(columns={'value': 'top10'}) \
    .merge(de_ipcc_region_bot[['region', 'value']].rename(columns={'value': 'bot50'}), on='region') \
    .merge(de_ipcc_region_tot[['region', 'value']].rename(columns={'value': 'total'}), on='region')

de_ipcc_region_all.iloc[:, 1:] /= 1e9

"""
PLOT
"""
world = gpd.read_file(country_shape_file)
world = world.rename(columns={'ADM0_A3': 'country'})

fig = plt.figure(figsize=(9, 10))
gs = fig.add_gridspec(4, 2, height_ratios=[1.8, 1.8, 1, 1])

# First row: dy and dp maps
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())

# Second row: de and degdp maps
ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson())
ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())

# Third row: dy and dp timeseries
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])

# Fourth row: de and degdp timeseries
ax7 = fig.add_subplot(gs[3, 0])
ax8 = fig.add_subplot(gs[3, 1])

# Plot maps
# TOP 10
map_data = [
    (ax1, df_dy_diff, "dy_avg_top", label_alphabets[0], cmap_dy, vmin_dy, vmax_dy, label_dy),
    (ax2, df_dp_diff, f"dp_sum_top", label_alphabets[1], cmap_dp, vmin_dp, vmax_dp, label_dp),
    (ax3, df_degdp_diff, "degdp_avg_top", label_alphabets[2], cmap_degdp, vmin_degdp, vmax_degdp, label_degdp),
    (ax4, df_de_diff, f"de_sum_top", label_alphabets[3], cmap_de, vmin_de, vmax_de, label_de)
]

title_fontsize = 12
label_fontsize = 12
tick_fontsize = 11

print()
print("Plotting maps..")

for ax, data, column, label, cmap, vmin, vmax, cbar_label in map_data:

    world_data = world.merge(data, on='country', how='left')
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    if ax == ax2:
        norm = colors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=vmin, vmax=vmax, base=10)

    if ax == ax4:
        norm = colors.SymLogNorm(linthresh=0.001, linscale=0.001, vmin=vmin, vmax=vmax, base=10)

    # Plot map
    world_data = world_data.to_crs(ccrs.Robinson().proj4_init)
    world_data.plot(ax=ax, color='white', edgecolor='gray', linewidth=0.5)
    world_data.plot(column=column, ax=ax,
                    norm=norm,
                    cmap=cmap, missing_kwds={'color': 'white', 'label': 'No Data'},
                    vmin=vmin, vmax=vmax)

    ax.text(-0.03, 1.0, label, transform=ax.transAxes, fontweight='bold', fontsize=label_fontsize + 1)
    ax.set_global()
    ax.gridlines(alpha=0.2)

    ###### COLORBAR
    pos = ax.get_position()

    if ax == ax1:
        cax = fig.add_axes([pos.x0 + 0.045, pos.y0 + 0.08, pos.width - 0.07, 0.005])
    if ax == ax2:
        cax = fig.add_axes([pos.x0 + 0.105, pos.y0 + 0.08, pos.width - 0.07, 0.005])
    if ax == ax3:
        cax = fig.add_axes([pos.x0 + 0.045, pos.y0 + 0.06, pos.width - 0.07, 0.005])
    if ax == ax4:
        cax = fig.add_axes([pos.x0 + 0.105, pos.y0 + 0.06, pos.width - 0.07, 0.005])

    if ax == ax2:
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          format=utils.format_fn,
                          ticks=ticks_dp,
                          cax=cax, orientation='horizontal', extend='both')

    elif ax == ax4:
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          format=utils.format_fn,
                          ticks=ticks_de,
                          cax=cax, orientation='horizontal', extend='both')

    else:
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          format=utils.format_fn,
                          cax=cax, orientation='horizontal', extend='both')

    cb.set_label(cbar_label, size=label_fontsize)
    cb.ax.tick_params(labelsize=tick_fontsize)

print("Plotting timeseries..")

#----------------- DY (production-weighted mean)
ax5.plot(years_plot, dy_bot_ts[plot_start_idx:plot_end_idx], color=bot50_color, linestyle='-', label='Bottom 50%')
ax5.plot(years_plot, dy_top_ts[plot_start_idx:plot_end_idx], color=top10_color, linestyle='-', label='Top 10%')
ax5.plot(years_plot, dy_bot_irr_ts[plot_start_idx:plot_end_idx], color=bot50_color, linestyle='--', alpha=alpha_irr, linewidth=lw_irr)
ax5.plot(years_plot, dy_top_irr_ts[plot_start_idx:plot_end_idx], color=top10_color, linestyle='--', alpha=alpha_irr, linewidth=lw_irr)
ax5.set_xlim(left=start_year_plot)
ax5.set_xlim(right=end_year_plot + 0.5)
ax5.set_ylim(ylim_bottom_dy, ylim_top_dy)
ax5.text(-0.03, 1.1, label_alphabets[4], transform=ax5.transAxes, fontweight='bold', fontsize=label_fontsize + 1)
ax5.set_ylabel(label_dy, fontsize=label_fontsize)
ax5.tick_params(labelsize=tick_fontsize)
ax5.legend(fontsize=label_fontsize, ncol=2)
ax5.grid(True)

# ----------------- ADD CONFIDENCE INTERVAL (DY) -----------------
year_2019 = years_plot[-1]
# ---TOP10
dy_2019_top = df_dy_top["2019"]
dy_mean_top = dy_2019_top.mean()
ci_low_top = dy_2019_top.quantile(quantile_low)
ci_high_top = dy_2019_top.quantile(quantile_high)
ci_low_interval_top = abs(dy_mean_top - ci_low_top)
ci_high_interval_top = abs(dy_mean_top - ci_high_top)
final_val_top = dy_top_ts[plot_end_idx - 1]
ci_low_top = final_val_top - ci_low_interval_top
ci_high_top = final_val_top + ci_high_interval_top
# ---BOT50
dy_2019_bot = df_dy_bot["2019"]
dy_mean_bot = dy_2019_bot.mean()
ci_low_bot = dy_2019_bot.quantile(quantile_low)
ci_high_bot = dy_2019_bot.quantile(quantile_high)
ci_low_interval_bot = abs(dy_mean_bot - ci_low_bot)
ci_high_interval_bot = abs(dy_mean_bot - ci_high_bot)
final_val_bot = dy_bot_ts[plot_end_idx - 1]
ci_low_bot = final_val_bot - ci_low_interval_bot
ci_high_bot = final_val_bot + ci_high_interval_bot

# Plot CI
ax5.vlines(year_2019 + 0.1, ci_low_top, ci_high_top,
           color=top10_color, linestyle='-', linewidth=0.8)
ax5.plot(year_2019 + 0.1, final_val_top, 'o',
         color=top10_color, markersize=1)
ax5.vlines(year_2019 + 0.1, ci_low_bot, ci_high_bot,
           color=bot50_color, linestyle='-', linewidth=0.8)
ax5.plot(year_2019 + 0.1, final_val_bot, 'o',
         color=bot50_color, markersize=1)

# ----------------------------------------------------------

# ----------------- DP
ax6.plot(years_plot, df_dp_bot[year_str].sum() / 1e6, color=bot50_color, linestyle='-', label='Bottom 50%')
ax6.plot(years_plot, df_dp_top[year_str].sum() / 1e6, color=top10_color, linestyle='-', label='Top 10%')
ax6.plot(years_plot, df_dp_bot_irr[year_str].sum() / 1e6, color=bot50_color, linestyle='--', alpha=alpha_irr, linewidth=lw_irr)
ax6.plot(years_plot, df_dp_top_irr[year_str].sum() / 1e6, color=top10_color, linestyle='--', alpha=alpha_irr, linewidth=lw_irr)
ax6.set_xlim(left=start_year_plot)
ax6.set_xlim(right=end_year_plot + 0.5)
ax6.set_ylim(ylim_bottom_dp, ylim_top_dp)
ax6.text(-0.03, 1.1, label_alphabets[5], transform=ax6.transAxes, fontweight='bold', fontsize=label_fontsize + 1)
ax6.set_ylabel(label_dp, fontsize=label_fontsize)
ax6.tick_params(labelsize=tick_fontsize)
ax6.grid(True)

degdp_top_val = np.nan_to_num(utils.compute_degdp_weighted(df_degdp_top, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt), nan=0)
degdp_bot_val = np.nan_to_num(utils.compute_degdp_weighted(df_degdp_bot, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt), nan=0)

# ----------------- DEGDP
ax7.plot(years_plot, degdp_bot_val, color=bot50_color, linestyle='-', label='Bottom 50%')
ax7.plot(years_plot, degdp_top_val, color=top10_color, linestyle='-', label='Top 10%')
degdp_top_val_irr = np.nan_to_num(utils.compute_degdp_weighted(df_degdp_top_irr, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt), nan=0)
degdp_bot_val_irr = np.nan_to_num(utils.compute_degdp_weighted(df_degdp_bot_irr, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt), nan=0)
ax7.plot(years_plot, degdp_bot_val_irr, color=bot50_color, linestyle='--', alpha=alpha_irr, linewidth=lw_irr)
ax7.plot(years_plot, degdp_top_val_irr, color=top10_color, linestyle='--', alpha=alpha_irr, linewidth=lw_irr)
ax7.set_xlim(left=start_year_plot)
ax7.set_xlim(right=end_year_plot + 0.5)
ax7.set_ylim(ylim_bottom_degdp, ylim_top_degdp)
ax7.text(-0.03, 1.1, label_alphabets[6], transform=ax7.transAxes, fontweight='bold', fontsize=label_fontsize + 1)
ax7.set_ylabel(label_degdp, fontsize=label_fontsize)
ax7.tick_params(labelsize=tick_fontsize)
ax7.grid(True)

# ----------------- ADD CONFIDENCE INTERVAL (DEGDP) -----------------
year_2019 = years_plot[-1]
# ---TOP10
degdp_2019_top = df_degdp_top["2019"]
degdp_mean_top = degdp_2019_top.mean()
ci_low_top = degdp_2019_top.quantile(quantile_low)
ci_high_top = degdp_2019_top.quantile(quantile_high)
ci_low_interval_top = abs(degdp_mean_top - ci_low_top)
ci_high_interval_top = abs(degdp_mean_top - ci_high_top)
final_val_top = degdp_top_val[-1]
ci_low_top = final_val_top - ci_low_interval_top
ci_high_top = final_val_top + ci_high_interval_top
# ---BOT50
degdp_2019_bot = df_degdp_bot["2019"]
degdp_mean_bot = degdp_2019_bot.mean()
ci_low_bot = degdp_2019_bot.quantile(quantile_low)
ci_high_bot = degdp_2019_bot.quantile(quantile_high)
ci_low_interval_bot = abs(degdp_mean_bot - ci_low_bot)
ci_high_interval_bot = abs(degdp_mean_bot - ci_high_bot)
final_val_bot = degdp_bot_val[-1]
ci_low_bot = final_val_bot - ci_low_interval_bot
ci_high_bot = final_val_bot + ci_high_interval_bot

# Plot CI
ax7.vlines(year_2019 + 0.2, ci_low_top, ci_high_top,
           color=top10_color, linestyle='-', linewidth=0.8)
ax7.plot(year_2019 + 0.2, final_val_top, 'o',
         color=top10_color, markersize=1)
ax7.vlines(year_2019 + 0.0, ci_low_bot, ci_high_bot,
           color=bot50_color, linestyle='-', linewidth=0.8)
ax7.plot(year_2019 + 0.0, final_val_bot, 'o',
         color=bot50_color, markersize=1)

# ----------------------------------------------------------

# ----------------- DE
ax8.plot(years_plot, df_de_bot[year_str].sum() / 1e9, color=bot50_color, linestyle='-', label='Bottom 50%')
ax8.plot(years_plot, df_de_top[year_str].sum() / 1e9, color=top10_color, linestyle='-', label='Top 10%')
ax8.plot(years_plot, df_de_bot_irr[year_str].sum() / 1e9, color=bot50_color, linestyle='--', alpha=alpha_irr, linewidth=lw_irr)
ax8.plot(years_plot, df_de_top_irr[year_str].sum() / 1e9, color=top10_color, linestyle='--', alpha=alpha_irr, linewidth=lw_irr)
ax8.set_xlim(left=start_year_plot)
ax8.set_xlim(right=end_year_plot + 0.5)
ax8.set_ylim(ylim_bottom_de, ylim_top_de)
ax8.text(-0.03, 1.1, label_alphabets[7], transform=ax8.transAxes, fontweight='bold', fontsize=label_fontsize + 1)
ax8.set_ylabel(label_de, fontsize=label_fontsize)
ax8.tick_params(labelsize=tick_fontsize)
ax8.grid(True)

print()
print("--------- FINAL YEAR VALUE:")
print("TOP10 dy:", dy_top_ts[plot_end_idx - 1])
print("BOT50 dy:", dy_bot_ts[plot_end_idx - 1])
print("TOTAL dy:", dy_tot_ts[plot_end_idx - 1])
print("TOP10 dp:", df_dp_top[year_str].sum().tolist()[-1] / 1e6)
print("BOT50 dp:", df_dp_bot[year_str].sum().tolist()[-1] / 1e6)
print("TOTAL dp:", df_dp_tot[year_str].sum().tolist()[-1] / 1e6)
print("TOP10 degdp:", utils.compute_degdp_weighted(df_degdp_top, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt)[-1])
print("BOT50 degdp:", utils.compute_degdp_weighted(df_degdp_bot, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt)[-1])
print("TOTAL degdp:", utils.compute_degdp_weighted(df_degdp_tot, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt)[-1])
print("TOP10 de:", df_de_top[year_str].sum().tolist()[-1] / 1e9)
print("BOT50 de:", df_de_bot[year_str].sum().tolist()[-1] / 1e9)
print("TOTAL de:", df_de_tot[year_str].sum().tolist()[-1] / 1e9)

print()
print("--------- FINAL YEAR VALUE (IRR):")
print("TOP10 dy (irr):", dy_top_irr_ts[plot_end_idx - 1])
print("BOT50 dy (irr):", dy_bot_irr_ts[plot_end_idx - 1])
print("TOTAL dy (irr):", dy_tot_irr_ts[plot_end_idx - 1])
print("TOP10 dp (irr):", df_dp_top_irr[year_str].sum().tolist()[-1] / 1e6)
print("BOT50 dp (irr):", df_dp_bot_irr[year_str].sum().tolist()[-1] / 1e6)
print("TOTAL dp (irr):", df_dp_tot_irr[year_str].sum().tolist()[-1] / 1e6)
print("TOP10 degdp (irr):", utils.compute_degdp_weighted(df_degdp_top_irr, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt)[-1])
print("BOT50 degdp (irr):", utils.compute_degdp_weighted(df_degdp_bot_irr, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt)[-1])
print("TOTAL degdp (irr):", utils.compute_degdp_weighted(df_degdp_tot_irr, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt)[-1])
print("TOP10 de (irr):", df_de_top_irr[year_str].sum().tolist()[-1] / 1e9)
print("BOT50 de (irr):", df_de_bot_irr[year_str].sum().tolist()[-1] / 1e9)
print("TOTAL de (irr):", df_de_tot_irr[year_str].sum().tolist()[-1] / 1e9)

print()
print("------------- AGGREGATED / MEAN OVER YEARS: ")
print("Number of years:", len(year_str))
mean_dy_top10 = float(dy_top_ts[plot_start_idx:plot_end_idx].mean())
mean_dy_bot50 = float(dy_bot_ts[plot_start_idx:plot_end_idx].mean())
mean_dy_total = float(dy_tot_ts[plot_start_idx:plot_end_idx].mean())
mean_degdp_top10 = float(utils.compute_degdp_weighted(df_degdp_top, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt).mean())
mean_degdp_bot50 = float(utils.compute_degdp_weighted(df_degdp_bot, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt).mean())
mean_degdp_total = float(utils.compute_degdp_weighted(df_degdp_tot, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt).mean())

sum_dp_top10 = sum(df_dp_top[year_str].sum().tolist()) / 1e6
sum_dp_bot50 = sum(df_dp_bot[year_str].sum().tolist()) / 1e6
sum_dp_total = sum(df_dp_tot[year_str].sum().tolist()) / 1e6
sum_de_top10 = sum(df_de_top[year_str].sum().tolist()) / 1e9
sum_de_bot50 = sum(df_de_bot[year_str].sum().tolist()) / 1e9
sum_de_total = sum(df_de_tot[year_str].sum().tolist()) / 1e9

print("TOP10 dy MEAN:", mean_dy_top10)
print("BOT50 dy MEAN:", mean_dy_bot50)
print("TOTAL dy MEAN:", mean_dy_total)
print("TOP10 degdp MEAN:", mean_degdp_top10)
print("BOT50 degdp MEAN:", mean_degdp_bot50)
print("TOTAL degdp MEAN:", mean_degdp_total)

print("TOP10 dp SUM:", sum_dp_top10)
print("BOT50 dp SUM:", sum_dp_bot50)
print("TOTAL dp SUM:", sum_dp_total)
print("TOP10 de SUM:", sum_de_top10)
print("BOT50 de SUM:", sum_de_bot50)
print("TOTAL de SUM:", sum_de_total)

print()
print("------------- AGGREGATED / MEAN OVER YEARS (IRR): ")
mean_dy_top10_irr = float(dy_top_irr_ts[plot_start_idx:plot_end_idx].mean())
mean_dy_bot50_irr = float(dy_bot_irr_ts[plot_start_idx:plot_end_idx].mean())
mean_dy_total_irr = float(dy_tot_irr_ts[plot_start_idx:plot_end_idx].mean())
mean_degdp_top10_irr = float(utils.compute_degdp_weighted(df_degdp_top_irr, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt).mean())
mean_degdp_bot50_irr = float(utils.compute_degdp_weighted(df_degdp_bot_irr, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt).mean())
mean_degdp_total_irr = float(utils.compute_degdp_weighted(df_degdp_tot_irr, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt).mean())

sum_dp_top10_irr = sum(df_dp_top_irr[year_str].sum().tolist()) / 1e6
sum_dp_bot50_irr = sum(df_dp_bot_irr[year_str].sum().tolist()) / 1e6
sum_dp_total_irr = sum(df_dp_tot_irr[year_str].sum().tolist()) / 1e6
sum_de_top10_irr = sum(df_de_top_irr[year_str].sum().tolist()) / 1e9
sum_de_bot50_irr = sum(df_de_bot_irr[year_str].sum().tolist()) / 1e9
sum_de_total_irr = sum(df_de_tot_irr[year_str].sum().tolist()) / 1e9

print("TOP10 dy MEAN (irr):", mean_dy_top10_irr)
print("BOT50 dy MEAN (irr):", mean_dy_bot50_irr)
print("TOTAL dy MEAN (irr):", mean_dy_total_irr)
print("TOP10 degdp MEAN (irr):", mean_degdp_top10_irr)
print("BOT50 degdp MEAN (irr):", mean_degdp_bot50_irr)
print("TOTAL degdp MEAN (irr):", mean_degdp_total_irr)

print("TOP10 dp SUM (irr):", sum_dp_top10_irr)
print("BOT50 dp SUM (irr):", sum_dp_bot50_irr)
print("TOTAL dp SUM (irr):", sum_dp_total_irr)
print("TOP10 de SUM (irr):", sum_de_top10_irr)
print("BOT50 de SUM (irr):", sum_de_bot50_irr)
print("TOTAL de SUM (irr):", sum_de_total_irr)

# Add x-label to bottom plots
ax7.set_xlabel('Year', fontsize=label_fontsize)
ax8.set_xlabel('Year', fontsize=label_fontsize)

# Adjust individual subplot positions
# First row (maps)
ax1.set_position([0.12, 0.78, 0.38, 0.2])  # [left, bottom, width, height]
ax2.set_position([0.6, 0.78, 0.38, 0.2])

# Second row (maps)
ax3.set_position([0.12, 0.51, 0.38, 0.2])  # [left, bottom, width, height]
ax4.set_position([0.6, 0.51, 0.38, 0.2])

# Third row (timeseries)
ax5.set_position([0.12, 0.28, 0.38, 0.13])  # [left, bottom, width, height]
ax6.set_position([0.6, 0.28, 0.38, 0.13])

# Fourth row (timeseries)
ax7.set_position([0.12, 0.06, 0.38, 0.13])  # [left, bottom, width, height]
ax8.set_position([0.6, 0.06, 0.38, 0.13])


plt.show()

"""
PRINT OUT IPCC SPECIFIC DIAGNOSTICS
"""
_yr_plot = [str(y) for y in range(start_year_plot, end_year_plot + 1)]

def _unweighted_region(df_ipcc, suffix_top, suffix_bot, suffix_tot):
    rows = []
    for r in regions:
        row = {"region": r}
        for label, df in [(suffix_top, dy_ipcc_top), (suffix_bot, dy_ipcc_bot), (suffix_tot, dy_ipcc_tot)]:
            row[label] = float(
                df[df[ar6_region] == r]
                .groupby("country")[_yr_plot].mean()
                .mean().mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)

dy_ipcc_region_all = pd.DataFrame([
    {"region": r,
     "top10": float(dy_ipcc_top[dy_ipcc_top[ar6_region] == r].groupby("country")[_yr_plot].mean().mean().mean()),
     "bot50": float(dy_ipcc_bot[dy_ipcc_bot[ar6_region] == r].groupby("country")[_yr_plot].mean().mean().mean()),
     "total": float(dy_ipcc_tot[dy_ipcc_tot[ar6_region] == r].groupby("country")[_yr_plot].mean().mean().mean())}
    for r in regions])

degdp_ipcc_region_all = pd.DataFrame([
    {"region": r,
     "top10": float(degdp_ipcc_top[degdp_ipcc_top[ar6_region] == r].groupby("country")[_yr_plot].mean().mean().mean()),
     "bot50": float(degdp_ipcc_bot[degdp_ipcc_bot[ar6_region] == r].groupby("country")[_yr_plot].mean().mean().mean()),
     "total": float(degdp_ipcc_tot[degdp_ipcc_tot[ar6_region] == r].groupby("country")[_yr_plot].mean().mean().mean())}
    for r in regions])

print()
print("###########################")
print(f"Region specific impact: {start_year_plot}-{end_year_plot}")

print("--------DY [%]")
print(dy_ipcc_region_all)
print("--------DEGDP [%]")
print(degdp_ipcc_region_all)
print("--------DP [Mt]")
print(dp_ipcc_region_all)
print("--------DE [B USD]")
print(de_ipcc_region_all)

"""
COMPARISON TABLE: RAINFED vs IRR
"""
print()
print("=" * 80)
print(f"COMPARISON TABLE: RAINFED vs IRR ({start_year_plot}-{end_year_plot})")
print("=" * 80)

comparison_data = {
    "Metric": [
        "DY MEAN Top10 [%]", "DY MEAN Bot50 [%]", "DY MEAN Total [%]",
        "DEGDP MEAN Top10 [%]", "DEGDP MEAN Bot50 [%]", "DEGDP MEAN Total [%]",
        "DP SUM Top10 [Mt]", "DP SUM Bot50 [Mt]", "DP SUM Total [Mt]",
        "DE SUM Top10 [B US$]", "DE SUM Bot50 [B US$]", "DE SUM Total [B US$]",
    ],
    "Rainfed": [
        mean_dy_top10, mean_dy_bot50, mean_dy_total,
        mean_degdp_top10, mean_degdp_bot50, mean_degdp_total,
        sum_dp_top10, sum_dp_bot50, sum_dp_total,
        sum_de_top10, sum_de_bot50, sum_de_total,
    ],
    "Irr": [
        mean_dy_top10_irr, mean_dy_bot50_irr, mean_dy_total_irr,
        mean_degdp_top10_irr, mean_degdp_bot50_irr, mean_degdp_total_irr,
        sum_dp_top10_irr, sum_dp_bot50_irr, sum_dp_total_irr,
        sum_de_top10_irr, sum_de_bot50_irr, sum_de_total_irr,
    ],
}
df_comparison = pd.DataFrame(comparison_data)
df_comparison["Diff"] = df_comparison["Irr"] - df_comparison["Rainfed"]
df_comparison["Diff (%)"] = ((df_comparison["Irr"] - df_comparison["Rainfed"]) / df_comparison["Rainfed"].abs()) * 100
print(df_comparison.to_string(index=False))
# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 300
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import geopandas as gpd
import cartopy
import cartopy.crs as ccrs
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('../')
import utils

"""
PARAMETERS
"""
crops = ["maize", "wheat", "soy"]  # all three crops
estring_top = "top10"
estring_bot = "bot50"
estring_tot = "total"
pred_str = "t3s3"

###################### PLOTTING PARAMETERS ######################
# Maps parameter
cmap_dy = "Reds_r"  # YlOrBr_r, YlOrRd_r
cmap_dp = "copper"
cmap_de = "Blues_r"  # YlGnBu_r, PuBuGn_r, YlOrBr_r, YlGnBu_r
cmap_degdp = "Purples_r"
vmin_dy = -2.5
vmax_dy = 0.1
vmin_dp = -10
vmax_dp = -0.001
ticks_dp = [-0.1, -1, -10]

vmin_degdp = -3.5e-3
vmax_degdp = 0.0
vmin_de = -0.1
vmax_de = -0.001
ticks_de = [-0.001, -0.01, -0.1]
start_year_avg = 2000  # 2000, 2007
end_year_avg = 2019

label_dy = "Yield change [%]"
label_dp = "Production change [Mt]"
label_de = "Economic loss [B US$]"
label_degdp = "GDP loss [%]"

# Timeseries parameters
ylim_top_dy = 0.2
ylim_bottom_dy = -2.3
ylim_top_dp = 3.0
ylim_bottom_dp = -45
ylim_top_degdp = 0.002
ylim_bottom_degdp = -0.014
ylim_top_de = 0.5
ylim_bottom_de = -10
start_year_plot = 2000
end_year_plot = 2019

label_alphabets = utils.generate_alphabet_list(8, option="lower")
label_alphabets = [x for x in label_alphabets]

window_size = 1
smooth_method = "numpy"
np_mode = "full"

year_str = [str(x) for x in range(start_year_plot, end_year_plot + 1)]
years_plot = [x for x in range(start_year_plot, end_year_plot + 1)]

dy_dp_color = "darkred"
de_degdp_color = "darkblue"

quantile_low = 0.25
quantile_high = 0.75
##################################################################

ar6_region = "region_ar6_dev"
regions = ["ldc", "developing", "developed"]

# For input file names
start_year_hist = 1971
end_year_hist = 1989
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

# Load data for all crops
for crop in crops:

    # Files for top10
    dy_file_top = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dy_{estring_top}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    dp_file_top = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dp_{estring_top}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    degdp_file_top = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/degdp_{estring_top}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    de_file_top = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/de_{estring_top}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"

    # Files for bot50
    dy_file_bot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dy_{estring_bot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    dp_file_bot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dp_{estring_bot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    degdp_file_bot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/degdp_{estring_bot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    de_file_bot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/de_{estring_bot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"

    # Files for total
    dy_file_tot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dy_{estring_tot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    dp_file_tot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dp_{estring_tot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    degdp_file_tot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/degdp_{estring_tot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    de_file_tot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/de_{estring_tot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"

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

# Calculate differences for mapping
df_dy_diff = pd.merge(dy_top[["country", "crop", "dy_avg"]], dy_bot[["country", "crop", "dy_avg"]],
                      on=["country", "crop"], suffixes=('_top', '_bot'))
df_dy_diff["dy_diff"] = df_dy_diff["dy_avg_top"] - df_dy_diff["dy_avg_bot"]

df_dp_diff = pd.merge(dp_top[["country", "crop", f"dp_sum"]], dp_bot[["country", "crop", f"dp_sum"]],
                      on=["country", "crop"], suffixes=('_top', '_bot'))
df_dp_diff[f"dp_sum_diff"] = df_dp_diff[f"dp_sum_top"] - df_dp_diff[f"dp_sum_bot"]

df_degdp_diff = pd.merge(degdp_top[["country", "crop", "degdp_avg"]], degdp_bot[["country", "crop", "degdp_avg"]],
                         on=["country", "crop"], suffixes=('_top', '_bot'))
df_degdp_diff["degdp_diff"] = df_degdp_diff["degdp_avg_top"] - df_degdp_diff["degdp_avg_bot"]

df_de_diff = pd.merge(de_top[["country", "crop", f"de_sum"]], de_bot[["country", "crop", f"de_sum"]],
                      on=["country", "crop"], suffixes=('_top', '_bot'))
df_de_diff[f"de_sum_diff"] = df_de_diff[f"de_sum_top"] - df_de_diff[f"de_sum_bot"]

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

dy_ipcc_stat_top = dy_ipcc_melted_top.groupby([ar6_region, 'year'], as_index=False)['value'].mean()
dy_ipcc_stat_top.rename(columns={ar6_region: 'region'}, inplace=True)
degdp_ipcc_stat_top = degdp_ipcc_melted_top.groupby([ar6_region, 'year'], as_index=False)['value'].mean()
degdp_ipcc_stat_top.rename(columns={ar6_region: 'region'}, inplace=True)
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

dy_ipcc_stat_bot = dy_ipcc_melted_bot.groupby([ar6_region, 'year'], as_index=False)['value'].mean()
dy_ipcc_stat_bot.rename(columns={ar6_region: 'region'}, inplace=True)
degdp_ipcc_stat_bot = degdp_ipcc_melted_bot.groupby([ar6_region, 'year'], as_index=False)['value'].mean()
degdp_ipcc_stat_bot.rename(columns={ar6_region: 'region'}, inplace=True)
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

dy_ipcc_stat_tot = dy_ipcc_melted_tot.groupby([ar6_region, 'year'], as_index=False)['value'].mean()
dy_ipcc_stat_tot.rename(columns={ar6_region: 'region'}, inplace=True)
degdp_ipcc_stat_tot = degdp_ipcc_melted_tot.groupby([ar6_region, 'year'], as_index=False)['value'].mean()
degdp_ipcc_stat_tot.rename(columns={ar6_region: 'region'}, inplace=True)
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

de_ipcc_region_all = de_ipcc_region_top[['region', 'value']].rename(columns={'value': 'top10'}) \
    .merge(de_ipcc_region_bot[['region', 'value']].rename(columns={'value': 'bot50'}), on='region') \
    .merge(de_ipcc_region_tot[['region', 'value']].rename(columns={'value': 'total'}), on='region')

dy_ipcc_region_all = dy_ipcc_region_top[['region', 'value']].rename(columns={'value': 'top10'}) \
    .merge(dy_ipcc_region_bot[['region', 'value']].rename(columns={'value': 'bot50'}), on='region') \
    .merge(dy_ipcc_region_tot[['region', 'value']].rename(columns={'value': 'total'}), on='region')

degdp_ipcc_region_all = degdp_ipcc_region_top[['region', 'value']].rename(columns={'value': 'top10'}) \
    .merge(degdp_ipcc_region_bot[['region', 'value']].rename(columns={'value': 'bot50'}), on='region') \
    .merge(degdp_ipcc_region_tot[['region', 'value']].rename(columns={'value': 'total'}), on='region')

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
    # dy_avg_top OR dy_avg_diff
    (ax2, df_dp_diff, f"dp_sum_top", label_alphabets[1], cmap_dp, vmin_dp, vmax_dp, label_dp),
    # dp_sum_top OR dp_sum_diff
    (ax3, df_degdp_diff, "degdp_avg_top", label_alphabets[2], cmap_degdp, vmin_degdp, vmax_degdp, label_degdp),
    # degdp_avg_top OR degdp_avg_diff
    (ax4, df_de_diff, f"de_sum_top", label_alphabets[3], cmap_de, vmin_de, vmax_de, label_de)
    # de_sum_top OR de_sum_diff
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

#----------------- DY
ax5.plot(years_plot, df_dy_top[year_str].mean(), color=dy_dp_color, label='Top10')
ax5.plot(years_plot, df_dy_bot[year_str].mean(), color=dy_dp_color, linestyle='--', label='Bot50')
ax5.set_xlim(right=end_year_plot + 0.5)
ax5.set_ylim(ylim_bottom_dy, ylim_top_dy)
ax5.text(-0.03, 1.1, label_alphabets[4], transform=ax5.transAxes, fontweight='bold', fontsize=label_fontsize + 1)
ax5.set_ylabel(label_dy, fontsize=label_fontsize)
ax5.tick_params(labelsize=tick_fontsize)
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
final_val_top = df_dy_top[year_str].mean()[-1]
ci_low_top = final_val_top - ci_low_interval_top
ci_high_top = final_val_top + ci_high_interval_top
# ---BOT50
dy_2019_bot = df_dy_bot["2019"]
dy_mean_bot = dy_2019_bot.mean()
ci_low_bot = dy_2019_bot.quantile(quantile_low)
ci_high_bot = dy_2019_bot.quantile(quantile_high)
ci_low_interval_bot = abs(dy_mean_bot - ci_low_bot)
ci_high_interval_bot = abs(dy_mean_bot - ci_high_bot)
final_val_bot = df_dy_bot[year_str].mean()[-1]
ci_low_bot = final_val_bot - ci_low_interval_bot
ci_high_bot = final_val_bot + ci_high_interval_bot

# Plot CI
ax5.vlines(year_2019 + 0.1, ci_low_top, ci_high_top,
           color=dy_dp_color, linestyle='-', linewidth=0.8)
ax5.plot(year_2019 + 0.1, final_val_top, 'o',
         color=dy_dp_color, markersize=1)
ax5.vlines(year_2019 + 0.1, ci_low_bot, ci_high_bot,
           color=dy_dp_color, linestyle='-', linewidth=0.8)
ax5.plot(year_2019 + 0.1, final_val_bot, 'o',
         color=dy_dp_color, markersize=1)

# ----------------------------------------------------------

# ----------------- DP
ax6.plot(years_plot, df_dp_top[year_str].sum() / 1e6, color=dy_dp_color, label='Top10')
ax6.plot(years_plot, df_dp_bot[year_str].sum() / 1e6, color=dy_dp_color, linestyle='--', label='Bot50')
ax6.set_xlim(right=end_year_plot + 0.5)
ax6.set_ylim(ylim_bottom_dp, ylim_top_dp)
ax6.text(-0.03, 1.1, label_alphabets[5], transform=ax6.transAxes, fontweight='bold', fontsize=label_fontsize + 1)
ax6.set_ylabel(label_dp, fontsize=label_fontsize)
ax6.tick_params(labelsize=tick_fontsize)
ax6.legend(fontsize=tick_fontsize)
ax6.grid(True)

degdp_top_val = df_degdp_top[year_str].mean()
degdp_top_val = np.nan_to_num(degdp_top_val, nan=0)
degdp_bot_val = df_degdp_bot[year_str].mean()
degdp_bot_val = np.nan_to_num(degdp_bot_val, nan=0)

# ----------------- DEGDP
ax7.plot(years_plot, degdp_top_val, color=de_degdp_color, label='Top10')
ax7.plot(years_plot, degdp_bot_val, color=de_degdp_color, linestyle='--', label='Bot50')
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
final_val_top = df_degdp_top[year_str].mean()[-1]
ci_low_top = final_val_top - ci_low_interval_top
ci_high_top = final_val_top + ci_high_interval_top
# ---BOT50
degdp_2019_bot = df_degdp_bot["2019"]
degdp_mean_bot = degdp_2019_bot.mean()
ci_low_bot = degdp_2019_bot.quantile(quantile_low)
ci_high_bot = degdp_2019_bot.quantile(quantile_high)
ci_low_interval_bot = abs(degdp_mean_bot - ci_low_bot)
ci_high_interval_bot = abs(degdp_mean_bot - ci_high_bot)
final_val_bot = df_degdp_bot[year_str].mean()[-1]
ci_low_bot = final_val_bot - ci_low_interval_bot
ci_high_bot = final_val_bot + ci_high_interval_bot

# Plot CI
ax7.vlines(year_2019 + 0.2, ci_low_top, ci_high_top,
           color=de_degdp_color, linestyle='-', linewidth=0.8)
ax7.plot(year_2019 + 0.2, final_val_top, 'o',
         color=de_degdp_color, markersize=1)
ax7.vlines(year_2019 + 0.0, ci_low_bot, ci_high_bot,
           color=de_degdp_color, linestyle='-', linewidth=0.8)
ax7.plot(year_2019 + 0.0, final_val_bot, 'o',
         color=de_degdp_color, markersize=1)

# ----------------------------------------------------------

# ----------------- DE
ax8.plot(years_plot, df_de_top[year_str].sum() / 1e9, color=de_degdp_color, label='Top10')
ax8.plot(years_plot, df_de_bot[year_str].sum() / 1e9, color=de_degdp_color, linestyle='--', label='Bot50')
ax8.set_xlim(right=end_year_plot + 0.5)
ax8.set_ylim(ylim_bottom_de, ylim_top_de)
ax8.text(-0.03, 1.1, label_alphabets[7], transform=ax8.transAxes, fontweight='bold', fontsize=label_fontsize + 1)
ax8.set_ylabel(label_de, fontsize=label_fontsize)
ax8.tick_params(labelsize=tick_fontsize)
ax8.legend(fontsize=tick_fontsize)
ax8.grid(True)

print()
print("--------- FINAL YEAR VALUE:")
print("TOP10 dy:", df_dy_top[year_str].mean().tolist()[-1])
print("BOT50 dy:", df_dy_bot[year_str].mean().tolist()[-1])
print("TOTAL dy:", df_dy_tot[year_str].mean().tolist()[-1])
print("TOP10 dp:", df_dp_top[year_str].sum().tolist()[-1] / 1e6)
print("BOT50 dp:", df_dp_bot[year_str].sum().tolist()[-1] / 1e6)
print("TOTAL dp:", df_dp_tot[year_str].sum().tolist()[-1] / 1e6)
print("TOP10 degdp:", df_degdp_top[year_str].mean().tolist()[-1])
print("BOT50 degdp:", df_degdp_bot[year_str].mean().tolist()[-1])
print("TOTAL degdp:", df_degdp_tot[year_str].mean().tolist()[-1])
print("TOP10 de:", df_de_top[year_str].sum().tolist()[-1] / 1e9)
print("BOT50 de:", df_de_bot[year_str].sum().tolist()[-1] / 1e9)
print("TOTAL de:", df_de_tot[year_str].sum().tolist()[-1] / 1e9)

print()
print("------------- AGGREGATED / MEAN OVER YEARS: ")
print("Number of years:", len(year_str))
mean_dy_top10 = np.mean(df_dy_top[year_str].mean().tolist())
mean_dy_bot50 = np.mean(df_dy_bot[year_str].mean().tolist())
mean_dy_total = np.mean(df_dy_tot[year_str].mean().tolist())
mean_degdp_top10 = np.mean(df_degdp_top[year_str].mean().tolist())
mean_degdp_bot50 = np.mean(df_degdp_bot[year_str].mean().tolist())
mean_degdp_total = np.mean(df_degdp_tot[year_str].mean().tolist())

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
print()
print("###########################")
print(f"Region specific impact: {start_year_plot}-{end_year_plot}")

print("--------DY")
print(dy_ipcc_region_all)
print("--------DEGDP")
print(degdp_ipcc_region_all)
print("--------DP")
print(dp_ipcc_region_all)
print("--------DE")
print(de_ipcc_region_all)
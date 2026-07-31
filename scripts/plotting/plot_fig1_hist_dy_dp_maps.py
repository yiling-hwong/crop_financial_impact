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
import geopandas as gpd
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('..')
import utils

"""
PARAMETERS
"""
crops = ["maize", "wheat", "soy"]
pred_str = "t3s3"
pred_str_irr = f"{pred_str}_irr"
spei_month = "03"
weight_opt = "weighted"  # "weighted", "simple"

###################### PLOTTING PARAMETERS ######################
# Map parameters
cmap_dy = "Reds_r"
cmap_dp = "copper"
vmin_dy = -9
vmax_dy = 0.0
vmin_dp = -20
vmax_dp = 0.0
ticks_dp = [0, -0.005, -1, -20]
start_year_avg = 2007
end_year_avg = 2019

# Timeseries parameters
ylim_top_dy = 0.8
ylim_bottom_dy = -8.5
ylim_top_dp = 10.0
ylim_bottom_dp = -100
start_year_plot = 2007
end_year_plot = 2019

label_dy = "Yield change [%]"
label_dp = "Production change [Mt]"

window_size = 5
smooth_method = "numpy"
np_mode = "full"

ipcc_colors = {"ldc": "crimson", "developing": "goldenrod", "developed": "dodgerblue"}
ipcc_labels = {"ldc": "LDC", "developing": "Developing", "developed": "Developed"}
labels_alphabets = utils.generate_alphabet_list(4, option="lower")

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

prod_weights = utils.load_fao_prod_weights(root_dir, spei_month)

df_dy_all = []
df_dp_all = []
df_dy_all_irr = []
df_dp_all_irr = []

for crop in crops:
    dy_file = f"{root_dir}/historical/linregress_outputs/{crop}/isimip3a/dy_isimip3a_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    dp_file = f"{root_dir}/historical/linregress_outputs/{crop}/isimip3a/dp_isimip3a_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"

    dy_file_irr = f"{root_dir}/historical/linregress_outputs/{crop}/isimip3a/dy_isimip3a_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    dp_file_irr = f"{root_dir}/historical/linregress_outputs/{crop}/isimip3a/dp_isimip3a_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"

    # Read Data
    df_dy = pd.read_csv(dy_file)
    df_dp = pd.read_csv(dp_file)
    df_dy_irr = pd.read_csv(dy_file_irr)
    df_dp_irr = pd.read_csv(dp_file_irr)

    # Add crop column
    df_dy['crop'] = crop
    df_dp['crop'] = crop
    df_dy_irr['crop'] = crop
    df_dp_irr['crop'] = crop

    # Append to lists
    df_dy_all.append(df_dy)
    df_dp_all.append(df_dp)
    df_dy_all_irr.append(df_dy_irr)
    df_dp_all_irr.append(df_dp_irr)

# Combine all crops data
df_dy = pd.concat(df_dy_all, ignore_index=True)
df_dp = pd.concat(df_dp_all, ignore_index=True)
df_dy_irr = pd.concat(df_dy_all_irr, ignore_index=True)
df_dp_irr = pd.concat(df_dp_all_irr, ignore_index=True)

# Get valid countries
hist_file = f"{root_dir}/historical/linregress_inputs/INPUT_HISTORICAL_DETRENDED_DEMEANED_spei{spei_month}.csv"
df_hist = pd.read_csv(hist_file)
valid_countries = df_hist["country"].unique()

# Filter for valid countries
df_dy = df_dy[df_dy['country'].isin(valid_countries)]
df_dp = df_dp[df_dp['country'].isin(valid_countries)]
df_dy_irr = df_dy_irr[df_dy_irr['country'].isin(valid_countries)]
df_dp_irr = df_dp_irr[df_dp_irr['country'].isin(valid_countries)]

# Prepare data for maps
year_str = [str(x) for x in range(start_year_avg, end_year_avg + 1)]
cols = ["country"] + year_str

dy_all = df_dy[cols]
dp_all = df_dp[cols]
dy_all_irr = df_dy_irr[cols]
dp_all_irr = df_dp_irr[cols]

# Calculate averages for each country
dy_all["dy_avg"] = dy_all.loc[:, f"{start_year_avg}":f"{end_year_avg}"].mean(axis=1)
dp_all[f"dp_sum"] = dp_all.loc[:, f"{start_year_avg}":f"{end_year_avg}"].sum(axis=1)
dy_all_irr["dy_avg"] = dy_all_irr.loc[:, f"{start_year_avg}":f"{end_year_avg}"].mean(axis=1)
dp_all_irr[f"dp_sum"] = dp_all_irr.loc[:, f"{start_year_avg}":f"{end_year_avg}"].sum(axis=1)

df1 = dy_all[["country", "dy_avg"]]
df2 = dp_all[["country", f"dp_sum"]]

# Average or sum across all crops and per country
df1 = df1.groupby("country", as_index=False)["dy_avg"].mean()
df2 = df2.groupby("country", as_index=False)[f"dp_sum"].sum()

# Load the world map shapefile
world = gpd.read_file(country_shape_file)
world = world.merge(df1.merge(df2, on="country"), how="left", left_on="ADM0_A3", right_on="country")

dy_max = world["dy_avg"].max()
dy_min = world["dy_avg"].min()
dp_max = (world[f"dp_sum"] / 1e6).max()
dp_min = (world[f"dp_sum"] / 1e6).min()

print("----------------")
print("dy max and min:", dy_max, dy_min)
print("dp max and min:", dp_max, dp_min)

# Prepare data for timeseries
years = [x for x in range(start_year_fut, end_year_fut + 1)]
years_plot = [x for x in range(start_year_plot, end_year_plot + 1)]
plot_start_idx = years.index(start_year_plot)
plot_end_idx = years.index(end_year_plot) + 1

_yr_str = [str(x) for x in range(start_year_fut, end_year_fut + 1)]
cols_ts_dy = ["country", "crop"] + _yr_str
cols_ts_dp = ["country"] + _yr_str
dy_all_ts = df_dy[cols_ts_dy]
dp_all_ts = df_dp[cols_ts_dp]
dy_all_ts_irr = df_dy_irr[cols_ts_dy]
dp_all_ts_irr = df_dp_irr[cols_ts_dp]

# Calculate global means/sums for timeseries
_yr_cols = _yr_str
dy_global = utils.compute_dy_weighted(dy_all_ts, dp_all_ts, _yr_cols, prod_weights=prod_weights, weight_opt=weight_opt)
dp_global = dp_all_ts.iloc[:, 1:].sum()
dy_global_irr = utils.compute_dy_weighted(dy_all_ts_irr, dp_all_ts_irr, _yr_cols, prod_weights=prod_weights, weight_opt=weight_opt)
dp_global_irr = dp_all_ts_irr.iloc[:, 1:].sum()

"""
GET IPCC REGION DATA
"""
print("----------------")
print("Getting ipcc region:", ar6_region)
df_ipcc = utils.get_ipcc_region_df()
df_ipcc = df_ipcc[["Country", ar6_region]]
df_ipcc = df_ipcc.rename(columns={"Country": "country"})

# Merge with IPCC regions
dy_ipcc = dy_all_ts.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
dp_ipcc = dp_all_ts.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
dy_ipcc_irr = dy_all_ts_irr.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
dp_ipcc_irr = dp_all_ts_irr.merge(df_ipcc[["country", ar6_region]], on="country", how="left")

# Calculate regional means/sums for timeseries
dy_ipcc_melted = dy_ipcc.melt(id_vars=['country', 'crop', 'region_ar6_dev'], var_name='year', value_name='value')
dy_ipcc_melted['year'] = dy_ipcc_melted['year'].astype(int)  # Convert year to integer
dp_ipcc_melted = dp_ipcc.melt(id_vars=['country', 'region_ar6_dev'], var_name='year', value_name='value')
dp_ipcc_melted['year'] = dp_ipcc_melted['year'].astype(int)  # Convert year to integer

dy_ipcc_melted_irr = dy_ipcc_irr.melt(id_vars=['country', 'crop', 'region_ar6_dev'], var_name='year', value_name='value')
dy_ipcc_melted_irr['year'] = dy_ipcc_melted_irr['year'].astype(int)  # Convert year to integer
dp_ipcc_melted_irr = dp_ipcc_irr.melt(id_vars=['country', 'region_ar6_dev'], var_name='year', value_name='value')
dp_ipcc_melted_irr['year'] = dp_ipcc_melted_irr['year'].astype(int)  # Convert year to integer

dy_ipcc_mean = pd.DataFrame([
    {"region": r, "year": int(yr), "value": v}
    for r in regions
    for yr, v in zip(_yr_cols, utils.compute_dy_weighted(
        dy_ipcc[dy_ipcc[ar6_region] == r][["country", "crop"] + _yr_cols],
        dp_ipcc[dp_ipcc[ar6_region] == r][["country"] + _yr_cols],
        _yr_cols,
        prod_weights=prod_weights, weight_opt=weight_opt))])
dp_ipcc_stat = dp_ipcc_melted.groupby(['region_ar6_dev', 'year'], as_index=False)['value'].sum()
dp_ipcc_stat.rename(columns={'region_ar6_dev': 'region'}, inplace=True)

dy_ipcc_mean_irr = pd.DataFrame([
    {"region": r, "year": int(yr), "value": v}
    for r in regions
    for yr, v in zip(_yr_cols, utils.compute_dy_weighted(
        dy_ipcc_irr[dy_ipcc_irr[ar6_region] == r][["country", "crop"] + _yr_cols],
        dp_ipcc_irr[dp_ipcc_irr[ar6_region] == r][["country"] + _yr_cols],
        _yr_cols,
        prod_weights=prod_weights, weight_opt=weight_opt))])
dp_ipcc_stat_irr = dp_ipcc_melted_irr.groupby(['region_ar6_dev', 'year'], as_index=False)['value'].sum()
dp_ipcc_stat_irr.rename(columns={'region_ar6_dev': 'region'}, inplace=True)

"""
PLOT
"""
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
title_fontsize = 12
label_fontsize = 12
tick_fontsize = 12

if len(crops) == 1:
    crop_plot = crops
else:
    crop_plot = "ALL"

######################## DY MAP
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
var = "dy_avg"

world_plot = world.copy()

# Reproject world geometries to Robinson
world_plot = world_plot.to_crs(ccrs.Robinson().proj4_init)
world_plot.plot(ax=ax1, color='white', edgecolor='gray', linewidth=0.5)

# Create a ScalarMappable for the colorbar
norm = plt.Normalize(vmin=vmin_dy, vmax=vmax_dy)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_dy)
sm.set_array([])

# Plot the data
plot = world_plot.plot(column=var, ax=ax1, cmap=cmap_dy,
                       vmin=vmin_dy, vmax=vmax_dy,
                       missing_kwds={"color": "white", "label": "No Data"})

# Add colorbar at the bottom
pos = ax1.get_position()
cax = fig.add_axes([pos.x0 + 0.01, pos.y0 + 0.07, pos.width - 0.07, 0.01])
cbar = plt.colorbar(sm, cax=cax, orientation='horizontal', extend='both')
cbar.set_label(label_dy, size=label_fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)
ax1.text(0.0, 1.15, "a", transform=ax1.transAxes,fontsize=label_fontsize+1, fontweight='bold', va='top', ha='right')
ax1.set_global()
ax1.gridlines(alpha=0.2)

######################## DP MAP
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
var = f"dp_sum"

world_plot = world.copy()

# Reproject world geometries to Robinson
world_plot = world_plot.to_crs(ccrs.Robinson().proj4_init)
world_plot.plot(ax=ax2, color='white', edgecolor='gray', linewidth=0.5)
world_plot[var] = world_plot[var] / 1e6  # convert to Mt

norm = plt.Normalize(vmin=vmin_dp, vmax=vmax_dp)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_dp)
sm.set_array([])

# Plot the data
plot = world_plot.plot(column=var, ax=ax2, cmap=cmap_dp,
                       norm=norm,
                       vmin=vmin_dp, vmax=vmax_dp,
                       missing_kwds={"color": "white", "label": "No Data"})

# Add colorbar at the bottom
pos = ax2.get_position()
cax = fig.add_axes([pos.x0 + 0.09, pos.y0 + 0.07, pos.width - 0.07, 0.01])
cbar = plt.colorbar(sm, cax=cax, orientation='horizontal', extend='both'
                    , format=utils.format_fn
                    )
cbar.set_label(label_dp, size=label_fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)
ax2.text(0.0, 1.15, "b", transform=ax2.transAxes,fontsize=label_fontsize+1, fontweight='bold', va='top', ha='right')
ax2.set_global()
ax2.gridlines(alpha=0.2)

######################## DY TIMESERIES
dy_region_list = []
ax3 = fig.add_subplot(gs[1, 0])
values = dy_global
values_irr = dy_global_irr

smoothed_values = utils.get_smoothed_values(values, window_size=window_size,method=smooth_method, np_mode=np_mode)
smoothed_values_irr = utils.get_smoothed_values(values_irr, window_size=window_size,method=smooth_method, np_mode=np_mode)
line1, = ax3.plot(years_plot, smoothed_values[plot_start_idx:plot_end_idx],color="black", linestyle='-', linewidth=0.8, label="Global")
line11, = ax3.plot(years_plot, smoothed_values_irr[plot_start_idx:plot_end_idx],color="black", linestyle='--', linewidth=0.8, alpha=0.6)
dy_global_final_year = smoothed_values[plot_end_idx - 1]

# Add global confidence intervals for 2019
year_2019 = years_plot[-1]
dy_2019 = dy_ipcc_melted[dy_ipcc_melted['year'] == 2019]
dy_mean = dy_2019["value"].mean()
ci_low = dy_2019['value'].quantile(quantile_low)
ci_high = dy_2019['value'].quantile(quantile_high)
ci_low_interval = abs(dy_mean - ci_low)
ci_high_interval = abs(dy_mean - ci_high)
final_val = smoothed_values[plot_end_idx - 1]
ci_low = final_val - ci_low_interval
ci_high = final_val + ci_high_interval

# Plot global CI
offsets = {'global': -0.4, 'ldc': -0.2, 'developing': 0.0, 'developed': 0.2}
ax3.vlines(year_2019 + offsets['global'], ci_low, ci_high,
           color="black", linestyle='-', linewidth=0.8)
ax3.plot(year_2019 + offsets['global'], final_val, 'o',
         color="black", markersize=1)

# IPCC REGIONS
for region in regions:
    values_ipcc = dy_ipcc_mean[dy_ipcc_mean["region"] == region]["value"]
    values_ipcc_irr = dy_ipcc_mean_irr[dy_ipcc_mean_irr["region"] == region]["value"]
    smoothed_values_ipcc = utils.get_smoothed_values(values_ipcc, window_size=window_size, method=smooth_method,np_mode=np_mode)
    smoothed_values_ipcc_irr = utils.get_smoothed_values(values_ipcc_irr, window_size=window_size, method=smooth_method,np_mode=np_mode)
    line3, = ax3.plot(years_plot, smoothed_values_ipcc[plot_start_idx:plot_end_idx],
                      color=ipcc_colors[region], linestyle="solid", linewidth=0.8, label=ipcc_labels[region])
    line33, = ax3.plot(years_plot, smoothed_values_ipcc_irr[plot_start_idx:plot_end_idx],
                      color=ipcc_colors[region], linestyle="dashed", linewidth=0.8, alpha=0.6)

    dy_region_list.append(values_ipcc[plot_start_idx:plot_end_idx])

    # Add confidence intervals for 2019
    year_2019 = years_plot[-1]
    dy_2019 = dy_ipcc_melted[dy_ipcc_melted['year'] == 2019]

    # Add CI for each region
    dy_region = dy_2019[dy_2019['region_ar6_dev'].str.lower() == region.lower()]['value']
    if len(dy_region) > 0:  # Only plot if we have data
        region_mean = dy_region.mean()
        ci_low = dy_region.quantile(quantile_low)
        ci_high = dy_region.quantile(quantile_high)
        ci_low_interval = abs(region_mean-ci_low)
        ci_high_interval = abs(region_mean-ci_high)
        final_val = smoothed_values_ipcc[plot_end_idx - 1]
        ci_low = final_val - ci_low_interval
        ci_high = final_val + ci_high_interval

        # Plot vertical line with CI
        ax3.vlines(year_2019 + offsets[region.lower()], ci_low, ci_high,
                   color=ipcc_colors[region], linestyle='-', linewidth=0.8)
        ax3.plot(year_2019 + offsets[region.lower()], final_val, 'o',
                 color=ipcc_colors[region], markersize=1)

if ylim_top_dy != None:
    ax3.set_ylim(top=ylim_top_dy)
if ylim_bottom_dy != None:
    ax3.set_ylim(bottom=ylim_bottom_dy)

ax3.set_xlim(left=start_year_plot)
ax3.set_xlim(right=end_year_plot+0.5)
ax3.text(0.0, 1.25, "c", transform=ax3.transAxes,fontsize=label_fontsize+1, fontweight='bold', va='top', ha='right')
ax3.axhline(0, color="black", linestyle="dashed", lw=0.9)
ax3.set_xlabel('Year', fontsize=label_fontsize)
ax3.set_ylabel(label_dy, fontsize=label_fontsize)
ax3.tick_params(axis='both', labelsize=tick_fontsize)
ax3.grid(True)

######################## DP TIMESERIES
dp_region_list = []
ax4 = fig.add_subplot(gs[1, 1])
values = dp_global.values / 1e6
values_irr = dp_global_irr.values / 1e6

smoothed_values = utils.get_smoothed_values(values, window_size=window_size,method=smooth_method, np_mode=np_mode)
smoothed_values_irr = utils.get_smoothed_values(values_irr, window_size=window_size,method=smooth_method, np_mode=np_mode)
ax4.plot(years_plot, smoothed_values[plot_start_idx:plot_end_idx],color="black", linestyle='-', linewidth=0.8, label="Global")
ax4.plot(years_plot, smoothed_values_irr[plot_start_idx:plot_end_idx],color="black", linestyle='--', linewidth=0.8, alpha=0.6)
dp_global_final_year = smoothed_values[plot_end_idx - 1]

# IPCC REGIONS
for region in regions:
    values_ipcc = dp_ipcc_stat[dp_ipcc_stat["region"] == region]["value"] / 1e6
    values_ipcc_irr = dp_ipcc_stat_irr[dp_ipcc_stat_irr["region"] == region]["value"] / 1e6
    smoothed_values_ipcc = utils.get_smoothed_values(values_ipcc, window_size=window_size, method=smooth_method,np_mode=np_mode)
    smoothed_values_ipcc_irr = utils.get_smoothed_values(values_ipcc_irr, window_size=window_size, method=smooth_method,np_mode=np_mode)
    line4, = ax4.plot(years_plot, smoothed_values_ipcc[plot_start_idx:plot_end_idx],
                      color=ipcc_colors[region], linestyle="solid", linewidth=0.8, label=ipcc_labels[region])
    line44, = ax4.plot(years_plot, smoothed_values_ipcc_irr[plot_start_idx:plot_end_idx],
                      color=ipcc_colors[region], linestyle="dashed", linewidth=0.8, alpha=0.6)

    dp_region_list.append(values_ipcc[plot_start_idx:plot_end_idx])


if ylim_top_dp != None:
    ax4.set_ylim(top=ylim_top_dp)
if ylim_bottom_dp != None:
    ax4.set_ylim(bottom=ylim_bottom_dp)

ax4.set_xlim(left=start_year_plot)
ax4.set_xlim(right=end_year_plot+0.5)
ax4.text(0.0, 1.25, "d", transform=ax4.transAxes,fontsize=label_fontsize+1, fontweight='bold', va='top', ha='right')
ax4.axhline(0, color="black", linestyle="dashed", lw=0.9)
ax4.set_xlabel('Year', fontsize=label_fontsize)
ax4.set_ylabel(label_dp, fontsize=label_fontsize)
ax4.tick_params(axis='both', labelsize=tick_fontsize)
ax4.grid(True)

# LEGEND
ax4.legend(bbox_to_anchor=(-0.2, -0.6), loc='center', fontsize=tick_fontsize - 1, ncol=4)

fig.subplots_adjust(left=0.1,
                    bottom=0.18,
                    right=0.98,  # 0.83 (with legend), 0.9 (no legend)
                    top=0.99,
                    wspace=0.25,
                    hspace=0.35)

plt.show()

"""
PRINT DIAGNOSTICS
"""
print("\n" + "="*50)
print("DY top 10 countries")
print("="*50)
dy_top10 = df1.sort_values(by='dy_avg', ascending=True).head(10)
print (dy_top10)

print("\n" + "="*50)
print("DP top 10 countries")
print("="*50)
dp_top10 = df2.sort_values(by='dp_sum', ascending=True).head(10)
print (dp_top10)

#----------------------- ANALYZE value of specific country
start_year = 2007
end_year = 2019
country = "ALL" #ALL, MOZ,CAF,MLI, MWI; CHN, BRA, FRA; GNB, ETH, TZA

year_str = [str(x) for x in range(start_year, end_year + 1)]
cols = ["country"] + year_str

dy_years = df_dy[cols]
dp_years = df_dp[cols]

dp_allcrop = dp_years.groupby('country').sum(numeric_only=True).reset_index()
dy_allcrop = dy_years.groupby('country').mean(numeric_only=True).reset_index()

dp_allcrop[f"dp_sum"] = dp_allcrop.loc[:, f"{start_year}":f"{end_year}"].sum(axis=1)
dy_allcrop["dy_avg"] = dy_allcrop.loc[:, f"{start_year}":f"{end_year}"].mean(axis=1)

if country == "ALL":
    dp_cntry = dp_allcrop["dp_sum"].sum()
    dy_cntry = float(dy_global[plot_start_idx:plot_end_idx].mean())
    dp_cntry = dp_cntry / 1e6
else:
    dp_cntry_1 = dp_allcrop[dp_allcrop["country"] == country]
    dy_cntry_1 = dy_allcrop[dy_allcrop["country"] == country]
    dp_cntry = dp_cntry_1["dp_sum"].values / 1e6
    dy_cntry = dy_cntry_1["dy_avg"].values
    dp_cntry = dp_cntry[0]
    dy_cntry = dy_cntry[0]

# GET REGION DATA
print ()
print ("--------------------")
dy_ldc_mean = dy_region_list[0].mean()
dy_developing_mean = dy_region_list[1].mean()
dy_dev_mean = dy_region_list[2].mean()

dp_ldc_sum = dp_region_list[0].sum()
dp_developing_sum = dp_region_list[1].sum()
dp_dev_sum = dp_region_list[2].sum()
print ("LENGTHS:",len(dy_region_list[0]),len(dp_region_list[0]))
print (f"TOTAL DP for LDC, DEVELOPING, DEVELOPED over {start_year}-{end_year} (Mt): {dp_ldc_sum}, {dp_developing_sum}, {dp_dev_sum}")
print (f"MEAN DY for LDC, DEVELOPING, DEVELOPED over {start_year}-{end_year} (%): {dy_ldc_mean}, {dy_developing_mean}, {dy_dev_mean}")

print ("--------------------")
print (f"TOTAL DP for {country} over {start_year}-{end_year} (Mt): {dp_cntry}")
print (f"MEAN DY for {country} over {start_year}-{end_year} (%): {dy_cntry}")
print (f"TOTAL DP for GLOBAL in {end_year} (Mt): {dp_global_final_year}")
print (f"MEAN DY for GLOBAL in {end_year} (%): {dy_global_final_year}")

#----------------------- ANALYZE _irr value of specific country
dy_years_irr = df_dy_irr[cols]
dp_years_irr = df_dp_irr[cols]

dp_allcrop_irr = dp_years_irr.groupby('country').sum(numeric_only=True).reset_index()
dy_allcrop_irr = dy_years_irr.groupby('country').mean(numeric_only=True).reset_index()

dp_allcrop_irr[f"dp_sum"] = dp_allcrop_irr.loc[:, f"{start_year}":f"{end_year}"].sum(axis=1)
dy_allcrop_irr["dy_avg"] = dy_allcrop_irr.loc[:, f"{start_year}":f"{end_year}"].mean(axis=1)

if country == "ALL":
    dp_cntry_irr = dp_allcrop_irr["dp_sum"].sum()
    dy_cntry_irr = float(dy_global_irr[plot_start_idx:plot_end_idx].mean())
    dp_cntry_irr = dp_cntry_irr / 1e6
else:
    dp_cntry_irr = dp_allcrop_irr[dp_allcrop_irr["country"] == country]
    dy_cntry_irr = dy_allcrop_irr[dy_allcrop_irr["country"] == country]
    dp_cntry_irr = dp_cntry_irr["dp_sum"].values / 1e6
    dy_cntry_irr = dy_cntry_irr["dy_avg"].values
    dp_cntry_irr = dp_cntry_irr[0]
    dy_cntry_irr = dy_cntry_irr[0]

# GET _irr REGION DATA
dy_ldc_mean_irr = dy_ipcc_mean_irr[(dy_ipcc_mean_irr["region"] == "ldc") & (dy_ipcc_mean_irr["year"].between(start_year, end_year))]["value"].mean()
dy_developing_mean_irr = dy_ipcc_mean_irr[(dy_ipcc_mean_irr["region"] == "developing") & (dy_ipcc_mean_irr["year"].between(start_year, end_year))]["value"].mean()
dy_dev_mean_irr = dy_ipcc_mean_irr[(dy_ipcc_mean_irr["region"] == "developed") & (dy_ipcc_mean_irr["year"].between(start_year, end_year))]["value"].mean()

dp_ldc_sum_irr = (dp_ipcc_stat_irr[(dp_ipcc_stat_irr["region"] == "ldc") & (dp_ipcc_stat_irr["year"].between(start_year, end_year))]["value"] / 1e6).sum()
dp_developing_sum_irr = (dp_ipcc_stat_irr[(dp_ipcc_stat_irr["region"] == "developing") & (dp_ipcc_stat_irr["year"].between(start_year, end_year))]["value"] / 1e6).sum()
dp_dev_sum_irr = (dp_ipcc_stat_irr[(dp_ipcc_stat_irr["region"] == "developed") & (dp_ipcc_stat_irr["year"].between(start_year, end_year))]["value"] / 1e6).sum()

"""
COMPARISON TABLE: RAINFED vs IRR
"""
print()
print("=" * 80)
print(f"COMPARISON TABLE: RAINFED vs IRR ({start_year}-{end_year})")
print("=" * 80)

comparison_data = {
    "Metric": [
        "DY MEAN LDC [%]", "DY MEAN Developing [%]", "DY MEAN Developed [%]", f"DY MEAN {country} [%]",
        "DP SUM LDC [Mt]", "DP SUM Developing [Mt]", "DP SUM Developed [Mt]", f"DP SUM {country} [Mt]",
    ],
    "Rainfed": [
        dy_ldc_mean, dy_developing_mean, dy_dev_mean, dy_cntry,
        dp_ldc_sum, dp_developing_sum, dp_dev_sum, dp_cntry,
    ],
    "Irr": [
        dy_ldc_mean_irr, dy_developing_mean_irr, dy_dev_mean_irr, dy_cntry_irr,
        dp_ldc_sum_irr, dp_developing_sum_irr, dp_dev_sum_irr, dp_cntry_irr,
    ],
}
df_comparison = pd.DataFrame(comparison_data)
df_comparison["Diff"] = df_comparison["Irr"] - df_comparison["Rainfed"]
df_comparison["Diff (%)"] = ((df_comparison["Irr"] - df_comparison["Rainfed"]) / df_comparison["Rainfed"].abs()) * 100
print(df_comparison.to_string(index=False))
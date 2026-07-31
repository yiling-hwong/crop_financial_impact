"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import geopandas as gpd
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('../')
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
cmap_de = utils.get_truncated_cmap("RdPu_r")
cmap_degdp = utils.get_truncated_cmap("YlGnBu_r")
vmin_dy = -3.5
vmax_dy = 0.0
vmin_dp = -100
vmax_dp = -0.1
ticks_dp = [-0.1, -1, -10, -100, -100]

vmin_degdp = -0.01
vmax_degdp = -0.0015
vmin_de = -10.0
vmax_de = -0.01
ticks_de = [-0.01, -0.1, -1.0, -10]

start_year_avg = 2007
end_year_avg = 2019

# Timeseries parameters
ylim_top_dy = -0.5
ylim_bottom_dy = -3.2
ylim_top_dp = 5.0
ylim_bottom_dp = -70
ylim_top_degdp = 0.01
ylim_bottom_degdp = -0.04
ylim_top_de = 2.0
ylim_bottom_de = -15
start_year_plot = 2007
end_year_plot = 2019

label_dy = "Yield change [%]"
label_dp = "Production change [Mt]"
label_de = "Economic loss [B US$]"
label_degdp = "GDP loss [%]"

ipcc_colors = {"ldc": "crimson", "developing": "goldenrod", "developed": "dodgerblue"}
ipcc_labels = {"ldc": "LDC", "developing": "Developing", "developed": "Developed"}

label_alphabets = utils.generate_alphabet_list(8, option="lower")
label_alphabets = [x for x in label_alphabets]

window_size = 1
smooth_method = "numpy"
np_mode = "same"

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
gdp_weights = utils.load_gdp_weights(root_dir)

df_dy_all = []
df_dp_all = []
df_degdp_all = []
df_de_all = []

df_dy_all_irr = []
df_dp_all_irr = []
df_degdp_all_irr = []
df_de_all_irr = []

for crop in crops:
    dy_file = f"{root_dir}/historical/linregress_outputs/{crop}/carbmaj/dy_carbmaj_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    dp_file = f"{root_dir}/historical/linregress_outputs/{crop}/carbmaj/dp_carbmaj_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    degdp_file = f"{root_dir}/historical/linregress_outputs/{crop}/carbmaj/degdp_carbmaj_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    de_file = f"{root_dir}/historical/linregress_outputs/{crop}/carbmaj/de_carbmaj_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"

    dy_file_irr = f"{root_dir}/historical/linregress_outputs/{crop}/carbmaj/dy_carbmaj_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    dp_file_irr = f"{root_dir}/historical/linregress_outputs/{crop}/carbmaj/dp_carbmaj_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    degdp_file_irr = f"{root_dir}/historical/linregress_outputs/{crop}/carbmaj/degdp_carbmaj_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    de_file_irr = f"{root_dir}/historical/linregress_outputs/{crop}/carbmaj/de_carbmaj_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"

    # Read Data
    df_dy = pd.read_csv(dy_file)
    df_dp = pd.read_csv(dp_file)
    df_degdp = pd.read_csv(degdp_file)
    df_de = pd.read_csv(de_file)

    # Read _irr Data
    df_dy_irr = pd.read_csv(dy_file_irr)
    df_dp_irr = pd.read_csv(dp_file_irr)
    df_degdp_irr = pd.read_csv(degdp_file_irr)
    df_de_irr = pd.read_csv(de_file_irr)

    # Add crop column
    df_dy['crop'] = crop
    df_dp['crop'] = crop
    df_degdp['crop'] = crop
    df_de['crop'] = crop

    # Add crop column _irr
    df_dy_irr['crop'] = crop
    df_dp_irr['crop'] = crop
    df_degdp_irr['crop'] = crop
    df_de_irr['crop'] = crop

    # Append to lists
    df_dy_all.append(df_dy)
    df_dp_all.append(df_dp)
    df_degdp_all.append(df_degdp)
    df_de_all.append(df_de)

    # Append _irr to lists
    df_dy_all_irr.append(df_dy_irr)
    df_dp_all_irr.append(df_dp_irr)
    df_degdp_all_irr.append(df_degdp_irr)
    df_de_all_irr.append(df_de_irr)

# Combine all crops data
df_dy = pd.concat(df_dy_all, ignore_index=True)
df_dp = pd.concat(df_dp_all, ignore_index=True)
df_degdp = pd.concat(df_degdp_all, ignore_index=True)
df_de = pd.concat(df_de_all, ignore_index=True)

# Combine all crops _irr data
df_dy_irr = pd.concat(df_dy_all_irr, ignore_index=True)
df_dp_irr = pd.concat(df_dp_all_irr, ignore_index=True)
df_degdp_irr = pd.concat(df_degdp_all_irr, ignore_index=True)
df_de_irr = pd.concat(df_de_all_irr, ignore_index=True)

# Get valid countries
hist_file = f"{root_dir}/historical/linregress_inputs/INPUT_HISTORICAL_DETRENDED_DEMEANED_spei{spei_month}.csv"
df_hist = pd.read_csv(hist_file)
valid_countries = df_hist["country"].unique()

# Filter for valid countries
df_dy = df_dy[df_dy['country'].isin(valid_countries)]
df_dp = df_dp[df_dp['country'].isin(valid_countries)]
df_degdp = df_degdp[df_degdp['country'].isin(valid_countries)]
df_de = df_de[df_de['country'].isin(valid_countries)]

# Filter _irr for valid countries
df_dy_irr = df_dy_irr[df_dy_irr['country'].isin(valid_countries)]
df_dp_irr = df_dp_irr[df_dp_irr['country'].isin(valid_countries)]
df_degdp_irr = df_degdp_irr[df_degdp_irr['country'].isin(valid_countries)]
df_de_irr = df_de_irr[df_de_irr['country'].isin(valid_countries)]

# Prepare data for maps
year_str = [str(x) for x in range(start_year_avg, end_year_avg + 1)]
cols = ["country"] + year_str

dy_all = df_dy[cols]
dp_all = df_dp[cols]
degdp_all = df_degdp[cols]
de_all = df_de[cols]

# Calculate averages/sums for each country
dy_all["dy_avg"] = dy_all.loc[:, f"{start_year_avg}":f"{end_year_avg}"].mean(axis=1)
degdp_all["degdp_avg"] = degdp_all.loc[:, f"{start_year_avg}":f"{end_year_avg}"].mean(axis=1)
dp_all[f"dp_sum"] = dp_all.loc[:, f"{start_year_avg}":f"{end_year_avg}"].sum(axis=1)
de_all[f"de_sum"] = de_all.loc[:, f"{start_year_avg}":f"{end_year_avg}"].sum(axis=1)

# Prepare data for maps
df1 = dy_all[["country", "dy_avg"]]
df2 = dp_all[["country", f"dp_sum"]]
df3 = degdp_all[["country", "degdp_avg"]]
df4 = de_all[["country", f"de_sum"]]

# Average across crops for dy and degdp
df1 = df1.groupby("country", as_index=False)["dy_avg"].mean()
df2 = df2.groupby("country", as_index=False)[f"dp_sum"].sum()
df3 = df3.groupby("country", as_index=False)["degdp_avg"].mean()
df4 = df4.groupby("country", as_index=False)[f"de_sum"].sum()

# Load the world map shapefile
world = gpd.read_file(country_shape_file)
world = world.rename(columns={'ADM0_A3': 'country'})

# Merge all data with world geometries
df_merged = df1.merge(df2, on="country").merge(df3, on="country").merge(df4, on="country")
world = world.merge(df_merged, on='country', how='left')

# Print min/max values for reference
print("\nValue ranges:")
print(f"dy: {world['dy_avg'].min():.2f} to {world['dy_avg'].max():.2f}")
print(f"dp: {(world[f'dp_sum'] / 1e6).min():.2f} to {(world[f'dp_sum'] / 1e6).max():.2f}")
print(f"degdp: {world['degdp_avg'].min():.2f} to {world['degdp_avg'].max():.2f}")
print(f"de: {(world[f'de_sum'] / 1e9).min():.2f} to {(world[f'de_sum'] / 1e9).max():.2f}")

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
degdp_all_ts = df_degdp[cols_ts_dp]
de_all_ts = df_de[cols_ts_dp]

# _irr timeseries
dy_all_ts_irr = df_dy_irr[cols_ts_dy]
dp_all_ts_irr = df_dp_irr[cols_ts_dp]
degdp_all_ts_irr = df_degdp_irr[cols_ts_dp]
de_all_ts_irr = df_de_irr[cols_ts_dp]

# Calculate global means/sums for timeseries
_yr_cols = _yr_str
dy_global_mean = utils.compute_dy_weighted(dy_all_ts, dp_all_ts, _yr_cols, prod_weights=prod_weights, weight_opt=weight_opt)
degdp_global_mean = utils.compute_degdp_weighted(degdp_all_ts, _yr_cols, gdp_weights=gdp_weights, weight_opt=weight_opt)
dp_global_sum = dp_all_ts.iloc[:, 1:].sum()
de_global_sum = de_all_ts.iloc[:, 1:].sum()

# Calculate global means/sums for timeseries _irr
dy_global_mean_irr = utils.compute_dy_weighted(dy_all_ts_irr, dp_all_ts_irr, _yr_cols, prod_weights=prod_weights, weight_opt=weight_opt)
degdp_global_mean_irr = utils.compute_degdp_weighted(degdp_all_ts_irr, _yr_cols, gdp_weights=gdp_weights, weight_opt=weight_opt)
dp_global_sum_irr = dp_all_ts_irr.iloc[:, 1:].sum()
de_global_sum_irr = de_all_ts_irr.iloc[:, 1:].sum()

"""
GET IPCC REGION DATA
"""
print ("--------------------")
print("Getting IPCC region:", ar6_region)
df_ipcc = utils.get_ipcc_region_df()
df_ipcc = df_ipcc[["Country", ar6_region]]
df_ipcc = df_ipcc.rename(columns={"Country": "country"})

# Merge with IPCC regions
dy_ipcc = dy_all_ts.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
dp_ipcc = dp_all_ts.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
degdp_ipcc = degdp_all_ts.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
de_ipcc = de_all_ts.merge(df_ipcc[["country", ar6_region]], on="country", how="left")

# Merge _irr with IPCC regions
dy_ipcc_irr = dy_all_ts_irr.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
dp_ipcc_irr = dp_all_ts_irr.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
degdp_ipcc_irr = degdp_all_ts_irr.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
de_ipcc_irr = de_all_ts_irr.merge(df_ipcc[["country", ar6_region]], on="country", how="left")

# Calculate regional means/sums for timeseries
dy_ipcc_melted = dy_ipcc.melt(id_vars=['country', 'crop', ar6_region], var_name='year', value_name='value')
dp_ipcc_melted = dp_ipcc.melt(id_vars=['country', ar6_region], var_name='year', value_name='value')
degdp_ipcc_melted = degdp_ipcc.melt(id_vars=['country', ar6_region], var_name='year', value_name='value')
de_ipcc_melted = de_ipcc.melt(id_vars=['country', ar6_region], var_name='year', value_name='value')

_ts_yr_cols = [str(x) for x in range(start_year_fut, end_year_fut + 1)]
dy_ipcc_stat = pd.DataFrame([
    {"region": r, "year": int(yr), "value": v}
    for r in regions
    for yr, v in zip(_ts_yr_cols, utils.compute_dy_weighted(
        dy_ipcc[dy_ipcc[ar6_region] == r][["country", "crop"] + _ts_yr_cols],
        dp_ipcc[dp_ipcc[ar6_region] == r][["country"] + _ts_yr_cols],
        _ts_yr_cols,
        prod_weights=prod_weights, weight_opt=weight_opt))])
degdp_ipcc_stat = pd.DataFrame([
    {"region": r, "year": yr, "value": v}
    for r in regions
    for yr, v in zip(_ts_yr_cols, utils.compute_degdp_weighted(
        degdp_ipcc[degdp_ipcc[ar6_region] == r],
        _ts_yr_cols,
        gdp_weights=gdp_weights, weight_opt=weight_opt))
])
dp_ipcc_stat = dp_ipcc_melted.groupby([ar6_region, 'year'], as_index=False)['value'].sum()
dp_ipcc_stat.rename(columns={ar6_region: 'region'}, inplace=True)
de_ipcc_stat = de_ipcc_melted.groupby([ar6_region, 'year'], as_index=False)['value'].sum()
de_ipcc_stat.rename(columns={ar6_region: 'region'}, inplace=True)

# Calculate regional means/sums for timeseries _irr
dy_ipcc_melted_irr = dy_ipcc_irr.melt(id_vars=['country', 'crop', ar6_region], var_name='year', value_name='value')
dp_ipcc_melted_irr = dp_ipcc_irr.melt(id_vars=['country', ar6_region], var_name='year', value_name='value')
degdp_ipcc_melted_irr = degdp_ipcc_irr.melt(id_vars=['country', ar6_region], var_name='year', value_name='value')
de_ipcc_melted_irr = de_ipcc_irr.melt(id_vars=['country', ar6_region], var_name='year', value_name='value')

dy_ipcc_stat_irr = pd.DataFrame([
    {"region": r, "year": int(yr), "value": v}
    for r in regions
    for yr, v in zip(_ts_yr_cols, utils.compute_dy_weighted(
        dy_ipcc_irr[dy_ipcc_irr[ar6_region] == r][["country", "crop"] + _ts_yr_cols],
        dp_ipcc_irr[dp_ipcc_irr[ar6_region] == r][["country"] + _ts_yr_cols],
        _ts_yr_cols,
        prod_weights=prod_weights, weight_opt=weight_opt))])
degdp_ipcc_stat_irr = pd.DataFrame([
    {"region": r, "year": yr, "value": v}
    for r in regions
    for yr, v in zip(_ts_yr_cols, utils.compute_degdp_weighted(
        degdp_ipcc_irr[degdp_ipcc_irr[ar6_region] == r],
        _ts_yr_cols,
        gdp_weights=gdp_weights, weight_opt=weight_opt))
])
dp_ipcc_stat_irr = dp_ipcc_melted_irr.groupby([ar6_region, 'year'], as_index=False)['value'].sum()
dp_ipcc_stat_irr.rename(columns={ar6_region: 'region'}, inplace=True)
de_ipcc_stat_irr = de_ipcc_melted_irr.groupby([ar6_region, 'year'], as_index=False)['value'].sum()
de_ipcc_stat_irr.rename(columns={ar6_region: 'region'}, inplace=True)

"""
PLOT
"""
fig = plt.figure(figsize=(9, 10))
gs = fig.add_gridspec(4, 2, height_ratios=[1.8, 1.8, 1, 1])

# First row: dy and dp maps
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())

# Second row: degdp and de maps
ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson())
ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())

# Third row: dy and dp timeseries
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])

# Fourth row: degdp and de timeseries
ax7 = fig.add_subplot(gs[3, 0])
ax8 = fig.add_subplot(gs[3, 1])

# Plot maps
map_data = [
    (ax1, world, "dy_avg", label_alphabets[0], cmap_dy, vmin_dy, vmax_dy, label_dy),
    (ax2, world, f"dp_sum", label_alphabets[1], cmap_dp, vmin_dp, vmax_dp, label_dp),
    (ax3, world, "degdp_avg", label_alphabets[2], cmap_degdp, vmin_degdp, vmax_degdp, label_degdp),
    (ax4, world, f"de_sum", label_alphabets[3], cmap_de, vmin_de, vmax_de, label_de)
]

title_fontsize = 12
label_fontsize = 12
tick_fontsize = 11

print()
print("Plotting maps..")

for ax, data, column, label, cmap, vmin, vmax, cbar_label in map_data:
    world_data = data.copy()
    world_data = world_data.to_crs(ccrs.Robinson().proj4_init)
    world_data.plot(ax=ax, color='white', edgecolor='gray', linewidth=0.5)

    # Convert units for dp and de
    if column == f"dp_sum":
        world_data[column] = world_data[column] / 1e6  # Convert to Mt
    elif column == f"de_sum":
        world_data[column] = world_data[column] / 1e9  # Convert to Billion USD

    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    if ax == ax2:
        norm = colors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=vmin, vmax=vmax, base=10)

    if ax == ax4:
        norm = colors.SymLogNorm(linthresh=0.01, linscale=0.01, vmin=vmin, vmax=vmax, base=10)

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
                          cax=cax, orientation='horizontal', extend='both')

    cb.set_label(cbar_label, size=label_fontsize)
    cb.ax.tick_params(labelsize=tick_fontsize)

print("Plotting timeseries..")
#-------------------------- DY
dy_region_list = []
dy_global_vals = dy_global_mean[plot_start_idx:plot_end_idx]
ax5.plot(years_plot, dy_global_vals, color="black", label='Global')
ax5.plot(years_plot, dy_global_mean_irr[plot_start_idx:plot_end_idx], color="black", linestyle='--', linewidth=0.8, alpha=0.6)

print ()
print ("-------------")
print (f"DY global values ({start_year_plot} & {end_year_plot}):",dy_global_vals[0],dy_global_vals[-1])

# Add global confidence intervals for 2019
year_2019 = years_plot[-1]
dy_2019 = dy_ipcc_melted[dy_ipcc_melted['year'] == "2019"]
dy_mean = dy_2019["value"].mean()
ci_low = dy_2019['value'].quantile(quantile_low)
ci_high = dy_2019['value'].quantile(quantile_high)
ci_low_interval = abs(dy_mean - ci_low)
ci_high_interval = abs(dy_mean - ci_high)
final_val = dy_global_mean[plot_end_idx - 1]
ci_low = final_val - ci_low_interval
ci_high = final_val + ci_high_interval

# Plot global CI
offsets = {'global': -0.4, 'ldc': -0.2, 'developing': 0.0, 'developed': 0.2}
ax5.vlines(year_2019 + offsets['global'], ci_low, ci_high,
           color="black", linestyle='-', linewidth=0.8)
ax5.plot(year_2019 + offsets['global'], final_val, 'o',
         color="black", markersize=1)

# IPCC REGIONS
for region in regions:
    values_ipcc = dy_ipcc_stat[dy_ipcc_stat["region"] == region]["value"]
    smoothed_values_ipcc = utils.get_smoothed_values(values_ipcc, window_size=window_size, method=smooth_method,
                                               np_mode=np_mode)
    ax5.plot(years_plot, smoothed_values_ipcc[plot_start_idx:plot_end_idx],
             color=ipcc_colors[region], linestyle="solid", linewidth=0.8, label=ipcc_labels[region])

    values_ipcc_irr = dy_ipcc_stat_irr[dy_ipcc_stat_irr["region"] == region]["value"]
    smoothed_values_ipcc_irr = utils.get_smoothed_values(values_ipcc_irr, window_size=window_size, method=smooth_method,
                                               np_mode=np_mode)
    ax5.plot(years_plot, smoothed_values_ipcc_irr[plot_start_idx:plot_end_idx],
             color=ipcc_colors[region], linestyle="dashed", linewidth=0.8, alpha=0.8)

    dy_region_list.append(smoothed_values_ipcc[plot_start_idx:plot_end_idx])

    # Add confidence intervals for 2019
    year_2019 = years_plot[-1]
    dy_2019 = dy_ipcc_melted[dy_ipcc_melted['year'] == "2019"]

    # Add CI for each region
    dy_region = dy_2019[dy_2019['region_ar6_dev'].str.lower() == region.lower()]['value']
    if len(dy_region) > 0:  # Only plot if we have data
        region_mean = dy_region.mean()
        ci_low = dy_region.quantile(quantile_low)
        ci_high = dy_region.quantile(quantile_high)
        ci_low_interval = abs(region_mean - ci_low)
        ci_high_interval = abs(region_mean - ci_high)
        final_val = smoothed_values_ipcc[plot_end_idx - 1]
        ci_low = final_val - ci_low_interval
        ci_high = final_val + ci_high_interval

        # Plot vertical line with CI
        ax5.vlines(year_2019 + offsets[region.lower()], ci_low, ci_high,
                   color=ipcc_colors[region], linestyle='-', linewidth=0.8)
        ax5.plot(year_2019 + offsets[region.lower()], final_val, 'o',
                 color=ipcc_colors[region], markersize=1)

ax5.set_xlim(left=start_year_plot)
ax5.set_xlim(right=end_year_plot + 0.5)
ax5.set_ylim(ylim_bottom_dy, ylim_top_dy)
ax5.axhline(0, color="black", linestyle="dashed", lw=0.9)
ax5.text(-0.03, 1.1, label_alphabets[4], transform=ax5.transAxes, fontweight='bold', fontsize=label_fontsize + 1)
ax5.set_ylabel(label_dy, fontsize=label_fontsize)
ax5.tick_params(labelsize=tick_fontsize)
ax5.grid(True)

#-------------------------- DP
dp_region_list = []
dp_global_vals = dp_global_sum[plot_start_idx:plot_end_idx] / 1e6
ax6.plot(years_plot, dp_global_vals, color="black", label='Global')
ax6.plot(years_plot, dp_global_sum_irr[plot_start_idx:plot_end_idx] / 1e6, color="black", linestyle='--', linewidth=0.8, alpha=0.6)

print (f"DP global values ({start_year_plot} & {end_year_plot}):",dp_global_vals[0],dp_global_vals[-1])

# IPCC REGIONS
for region in regions:
    values_ipcc = dp_ipcc_stat[dp_ipcc_stat["region"] == region]["value"] / 1e6
    smoothed_values_ipcc = utils.get_smoothed_values(values_ipcc, window_size=window_size, method=smooth_method,
                                               np_mode=np_mode)
    ax6.plot(years_plot, smoothed_values_ipcc[plot_start_idx:plot_end_idx],
             color=ipcc_colors[region], linestyle="solid", linewidth=0.8, label=ipcc_labels[region])

    values_ipcc_irr = dp_ipcc_stat_irr[dp_ipcc_stat_irr["region"] == region]["value"] / 1e6
    smoothed_values_ipcc_irr = utils.get_smoothed_values(values_ipcc_irr, window_size=window_size, method=smooth_method,
                                               np_mode=np_mode)
    ax6.plot(years_plot, smoothed_values_ipcc_irr[plot_start_idx:plot_end_idx],
             color=ipcc_colors[region], linestyle="dashed", linewidth=0.8, alpha=0.6)

    dp_region_list.append(smoothed_values_ipcc[plot_start_idx:plot_end_idx])

ax6.set_xlim(left=start_year_plot)
ax6.set_xlim(right=end_year_plot + 0.5)
ax6.set_ylim(ylim_bottom_dp, ylim_top_dp)
ax6.axhline(0, color="black", linestyle="dashed", lw=0.9)
ax6.text(-0.03, 1.1, label_alphabets[5], transform=ax6.transAxes, fontweight='bold', fontsize=label_fontsize + 1)
ax6.set_ylabel(label_dp, fontsize=label_fontsize)
ax6.tick_params(labelsize=tick_fontsize)
ax6.legend(fontsize=tick_fontsize - 1, loc="best", ncol=2)
ax6.grid(True)

#-------------------------- DEGDP
degdp_region_list = []
degdp_global_vals = degdp_global_mean[plot_start_idx:plot_end_idx]
degdp_global_vals = np.nan_to_num(degdp_global_vals, nan=0)  # convert NaN to zeros
ax7.plot(years_plot, degdp_global_vals, color="black", label='Global')
degdp_global_vals_irr = np.nan_to_num(degdp_global_mean_irr[plot_start_idx:plot_end_idx], nan=0)
ax7.plot(years_plot, degdp_global_vals_irr, color="black", linestyle='--', linewidth=0.8, alpha=0.6)

print (f"DEGDP global values ({start_year_plot} & {end_year_plot}):",degdp_global_vals[0],degdp_global_vals[-1])

# Add global confidence intervals for 2019
year_2019 = years_plot[-1]
degdp_2019 = degdp_ipcc_melted[degdp_ipcc_melted['year'] == "2019"]
degdp_mean = degdp_2019["value"].mean()
ci_low = degdp_2019['value'].quantile(quantile_low)
ci_high = degdp_2019['value'].quantile(quantile_high)
ci_low_interval = abs(degdp_mean - ci_low)
ci_high_interval = abs(degdp_mean - ci_high)
final_val = degdp_global_mean[plot_end_idx - 1]
ci_low = final_val - ci_low_interval
ci_high = final_val + ci_high_interval

# Plot global CI
offsets = {'global': -0.4, 'ldc': -0.2, 'developing': 0.0, 'developed': 0.2}
ax7.vlines(year_2019 + offsets['global'], ci_low, ci_high,
           color="black", linestyle='-', linewidth=0.8)
ax7.plot(year_2019 + offsets['global'], final_val, 'o',
         color="black", markersize=1)

# IPCC REGIONS
for region in regions:
    values_ipcc = degdp_ipcc_stat[degdp_ipcc_stat["region"] == region]["value"]
    values_ipcc = np.nan_to_num(values_ipcc, nan=0)  # convert NaN to zeros
    smoothed_values_ipcc = utils.get_smoothed_values(values_ipcc, window_size=window_size, method=smooth_method,
                                               np_mode=np_mode)
    ax7.plot(years_plot, smoothed_values_ipcc[plot_start_idx:plot_end_idx],
             color=ipcc_colors[region], linestyle="solid", linewidth=0.8, label=ipcc_labels[region])

    values_ipcc_irr = degdp_ipcc_stat_irr[degdp_ipcc_stat_irr["region"] == region]["value"]
    values_ipcc_irr = np.nan_to_num(values_ipcc_irr, nan=0)
    smoothed_values_ipcc_irr = utils.get_smoothed_values(values_ipcc_irr, window_size=window_size, method=smooth_method,
                                               np_mode=np_mode)

    ax7.plot(years_plot, smoothed_values_ipcc_irr[plot_start_idx:plot_end_idx],
             color=ipcc_colors[region], linestyle="dashed", linewidth=0.8, alpha=0.6)

    degdp_region_list.append(smoothed_values_ipcc[plot_start_idx:plot_end_idx])

    # Add confidence intervals for 2019
    year_2019 = years_plot[-1]
    degdp_2019 = degdp_ipcc_melted[degdp_ipcc_melted['year'] == "2019"]

    # Add CI for each region
    degdp_region = degdp_2019[degdp_2019['region_ar6_dev'].str.lower() == region.lower()]['value']
    if len(degdp_region) > 0:  # Only plot if we have data
        region_mean = degdp_region.mean()
        ci_low = degdp_region.quantile(quantile_low)
        ci_high = degdp_region.quantile(quantile_high)
        ci_low_interval = abs(region_mean - ci_low)
        ci_high_interval = abs(region_mean - ci_high)
        final_val = smoothed_values_ipcc[plot_end_idx - 1]
        ci_low = final_val - ci_low_interval
        ci_high = final_val + ci_high_interval

        # Plot vertical line with CI
        ax7.vlines(year_2019 + offsets[region.lower()], ci_low, ci_high,
                   color=ipcc_colors[region], linestyle='-', linewidth=0.8)
        ax7.plot(year_2019 + offsets[region.lower()], final_val, 'o',
                 color=ipcc_colors[region], markersize=1)

ax7.set_xlim(left=start_year_plot)
ax7.set_xlim(right=end_year_plot + 0.5)
ax7.set_ylim(ylim_bottom_degdp, ylim_top_degdp)
ax7.axhline(0, color="black", linestyle="dashed", lw=0.9)
ax7.text(-0.03, 1.1, label_alphabets[6], transform=ax7.transAxes, fontweight='bold', fontsize=label_fontsize + 1)
ax7.set_ylabel(label_degdp, fontsize=label_fontsize)
ax7.tick_params(labelsize=tick_fontsize)
ax7.grid(True)

#-------------------------- DE
de_region_list = []
de_global_vals = de_global_sum[plot_start_idx:plot_end_idx] / 1e9
ax8.plot(years_plot, de_global_vals, color="black", label='Global')
ax8.plot(years_plot, de_global_sum_irr[plot_start_idx:plot_end_idx] / 1e9, color="black", linestyle='--', linewidth=0.8, alpha=0.6)

print (f"DE global values ({start_year_plot} & {end_year_plot}):",de_global_vals[0],de_global_vals[-1])

# IPCC REGIONS
for region in regions:
    values_ipcc = de_ipcc_stat[de_ipcc_stat["region"] == region]["value"] / 1e9
    smoothed_values_ipcc = utils.get_smoothed_values(values_ipcc, window_size=window_size, method=smooth_method,
                                               np_mode=np_mode)
    ax8.plot(years_plot, smoothed_values_ipcc[plot_start_idx:plot_end_idx],
             color=ipcc_colors[region], linestyle="solid", linewidth=0.8, label=ipcc_labels[region])

    values_ipcc_irr = de_ipcc_stat_irr[de_ipcc_stat_irr["region"] == region]["value"] / 1e9
    smoothed_values_ipcc_irr = utils.get_smoothed_values(values_ipcc_irr, window_size=window_size, method=smooth_method,
                                               np_mode=np_mode)

    ax8.plot(years_plot, smoothed_values_ipcc_irr[plot_start_idx:plot_end_idx],
             color=ipcc_colors[region], linestyle="dashed", linewidth=0.8, alpha=0.6)

    de_region_list.append(smoothed_values_ipcc[plot_start_idx:plot_end_idx])

ax8.set_xlim(left=start_year_plot)
ax8.set_xlim(right=end_year_plot + 0.5)
ax8.set_ylim(ylim_bottom_de, ylim_top_de)
ax8.axhline(0, color="black", linestyle="dashed", lw=0.9)
ax8.text(-0.03, 1.1, label_alphabets[7], transform=ax8.transAxes, fontweight='bold', fontsize=label_fontsize + 1)
ax8.set_ylabel(label_de, fontsize=label_fontsize)
ax8.tick_params(labelsize=tick_fontsize)
ax8.grid(True)

# Add x-label to bottom plots
ax7.set_xlabel('Year', fontsize=label_fontsize)
ax8.set_xlabel('Year', fontsize=label_fontsize)

"""
ADJUST SUBPLOT POSITIONS
"""
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
PRINT OUT DIAGNOSTICS
"""
dy_global_mean = float(dy_global_vals.mean())
degdp_global_mean = float(utils.compute_degdp_weighted(df_degdp, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt).mean())
dy_ldc_mean     = float(dy_ipcc_stat[(dy_ipcc_stat['region'] == 'ldc') & dy_ipcc_stat['year'].between(start_year_plot, end_year_plot)]['value'].mean())
dy_developing_mean = float(dy_ipcc_stat[(dy_ipcc_stat['region'] == 'developing') & dy_ipcc_stat['year'].between(start_year_plot, end_year_plot)]['value'].mean())
dy_dev_mean     = float(dy_ipcc_stat[(dy_ipcc_stat['region'] == 'developed') & dy_ipcc_stat['year'].between(start_year_plot, end_year_plot)]['value'].mean())
df_degdp_reg = df_degdp.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
df_de_reg    = df_de.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
degdp_ldc_mean = float(utils.compute_degdp_weighted(
    df_degdp_reg[df_degdp_reg[ar6_region] == 'ldc'],
    year_str, gdp_weights=gdp_weights, weight_opt=weight_opt).mean())
degdp_developing_mean = float(utils.compute_degdp_weighted(
    df_degdp_reg[df_degdp_reg[ar6_region] == 'developing'],
    year_str, gdp_weights=gdp_weights, weight_opt=weight_opt).mean())
degdp_dev_mean = float(utils.compute_degdp_weighted(
    df_degdp_reg[df_degdp_reg[ar6_region] == 'developed'],
    year_str, gdp_weights=gdp_weights, weight_opt=weight_opt).mean())
print()
print("------------- AGGREGATED / MEAN OVER YEARS VALUES")
print("MEAN dy GLOBAL:", dy_global_mean)
print("MEAN dy ldc:", dy_ldc_mean)
print("MEAN dy developing:", dy_developing_mean)
print("MEAN dy developed:", dy_dev_mean)
print("----------------")
print("MEAN degdp GLOBAL:", degdp_global_mean)
print("MEAN degdp ldc:", degdp_ldc_mean)
print("MEAN degdp developing:", degdp_developing_mean)
print("MEAN degdp developed:", degdp_dev_mean)

dp_global_sum = dp_global_vals.sum()
dp_ldc_sum = np.sum(dp_region_list[0])
dp_developing_sum = np.sum(dp_region_list[1])
dp_dev_sum = np.sum(dp_region_list[2])
de_global_sum = de_global_vals.sum()
de_ldc_sum = np.sum(de_region_list[0])
de_developing_sum = np.sum(de_region_list[1])
de_dev_sum = np.sum(de_region_list[2])
print("----------------")
print("SUM dp GLOBAL:", dp_global_sum)
print("SUM dp ldc:", dp_ldc_sum)
print("SUM dp developing:", dp_developing_sum)
print("SUM dp developed:", dp_dev_sum)
print("----------------")
print("SUM de GLOBAL:", de_global_sum)
print("SUM de ldc:", de_ldc_sum)
print("SUM de developing:", de_developing_sum)
print("SUM de developed:", de_dev_sum)

# Compute _irr aggregated values
dy_global_mean_irr_agg = float(dy_global_mean_irr[plot_start_idx:plot_end_idx].mean())
degdp_global_mean_irr_agg = float(utils.compute_degdp_weighted(df_degdp_irr, year_str, gdp_weights=gdp_weights, weight_opt=weight_opt).mean())
dp_global_sum_irr_agg = (dp_global_sum_irr[plot_start_idx:plot_end_idx] / 1e6).sum()
de_global_sum_irr_agg = (de_global_sum_irr[plot_start_idx:plot_end_idx] / 1e9).sum()

# Compute _irr regional values
df_degdp_irr_reg = df_degdp_irr.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
df_de_irr_reg = df_de_irr.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
dy_ldc_mean_irr = float(dy_ipcc_stat_irr[(dy_ipcc_stat_irr['region'] == 'ldc') & dy_ipcc_stat_irr['year'].between(start_year_plot, end_year_plot)]['value'].mean())
dy_developing_mean_irr = float(dy_ipcc_stat_irr[(dy_ipcc_stat_irr['region'] == 'developing') & dy_ipcc_stat_irr['year'].between(start_year_plot, end_year_plot)]['value'].mean())
dy_dev_mean_irr = float(dy_ipcc_stat_irr[(dy_ipcc_stat_irr['region'] == 'developed') & dy_ipcc_stat_irr['year'].between(start_year_plot, end_year_plot)]['value'].mean())

degdp_ldc_mean_irr = float(utils.compute_degdp_weighted(
    df_degdp_irr_reg[df_degdp_irr_reg[ar6_region] == 'ldc'],
    year_str, gdp_weights=gdp_weights, weight_opt=weight_opt).mean())
degdp_developing_mean_irr = float(utils.compute_degdp_weighted(
    df_degdp_irr_reg[df_degdp_irr_reg[ar6_region] == 'developing'],
    year_str, gdp_weights=gdp_weights, weight_opt=weight_opt).mean())
degdp_dev_mean_irr = float(utils.compute_degdp_weighted(
    df_degdp_irr_reg[df_degdp_irr_reg[ar6_region] == 'developed'],
    year_str, gdp_weights=gdp_weights, weight_opt=weight_opt).mean())

dp_ldc_sum_irr = (dp_ipcc_stat_irr[dp_ipcc_stat_irr["region"] == "ldc"]["value"].iloc[plot_start_idx:plot_end_idx] / 1e6).sum()
dp_developing_sum_irr = (dp_ipcc_stat_irr[dp_ipcc_stat_irr["region"] == "developing"]["value"].iloc[plot_start_idx:plot_end_idx] / 1e6).sum()
dp_dev_sum_irr = (dp_ipcc_stat_irr[dp_ipcc_stat_irr["region"] == "developed"]["value"].iloc[plot_start_idx:plot_end_idx] / 1e6).sum()

de_ldc_sum_irr = (de_ipcc_stat_irr[de_ipcc_stat_irr["region"] == "ldc"]["value"].iloc[plot_start_idx:plot_end_idx] / 1e9).sum()
de_developing_sum_irr = (de_ipcc_stat_irr[de_ipcc_stat_irr["region"] == "developing"]["value"].iloc[plot_start_idx:plot_end_idx] / 1e9).sum()
de_dev_sum_irr = (de_ipcc_stat_irr[de_ipcc_stat_irr["region"] == "developed"]["value"].iloc[plot_start_idx:plot_end_idx] / 1e9).sum()

"""
COMPARISON TABLE: RAINFED vs IRR
"""
print()
print("=" * 80)
print(f"COMPARISON TABLE: RAINFED vs IRR ({start_year_plot}-{end_year_plot})")
print("=" * 80)

comparison_data = {
    "Metric": [
        "DY MEAN Global [%]", "DY MEAN LDC [%]", "DY MEAN Developing [%]", "DY MEAN Developed [%]",
        "DEGDP MEAN Global [%]", "DEGDP MEAN LDC [%]", "DEGDP MEAN Developing [%]", "DEGDP MEAN Developed [%]",
        "DP SUM Global [Mt]", "DP SUM LDC [Mt]", "DP SUM Developing [Mt]", "DP SUM Developed [Mt]",
        "DE SUM Global [B US$]", "DE SUM LDC [B US$]", "DE SUM Developing [B US$]", "DE SUM Developed [B US$]",
    ],
    "Rainfed": [
        dy_global_mean, dy_ldc_mean, dy_developing_mean, dy_dev_mean,
        degdp_global_mean, degdp_ldc_mean, degdp_developing_mean, degdp_dev_mean,
        dp_global_sum, dp_ldc_sum, dp_developing_sum, dp_dev_sum,
        de_global_sum, de_ldc_sum, de_developing_sum, de_dev_sum,
    ],
    "Irr": [
        dy_global_mean_irr_agg, dy_ldc_mean_irr, dy_developing_mean_irr, dy_dev_mean_irr,
        degdp_global_mean_irr_agg, degdp_ldc_mean_irr, degdp_developing_mean_irr, degdp_dev_mean_irr,
        dp_global_sum_irr_agg, dp_ldc_sum_irr, dp_developing_sum_irr, dp_dev_sum_irr,
        de_global_sum_irr_agg, de_ldc_sum_irr, de_developing_sum_irr, de_dev_sum_irr,
    ],
}
df_comparison = pd.DataFrame(comparison_data)
df_comparison["Diff"] = df_comparison["Irr"] - df_comparison["Rainfed"]
df_comparison["Diff (%)"] = ((df_comparison["Irr"] - df_comparison["Rainfed"]) / df_comparison["Rainfed"].abs()) * 100
print(df_comparison.to_string(index=False))
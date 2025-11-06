"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 300
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

###################### PLOTTING PARAMETERS ######################
# Map parameters
cmap_dy = "Reds_r"
cmap_dp = "copper"
cmap_de = "Blues_r"
cmap_degdp = "Purples_r"
vmin_dy = -3.0
vmax_dy = 0.0
vmin_dp = -1000
vmax_dp = -0.1
ticks_dp = [-0.1, -1, -10, -100, -1000]

vmin_degdp = -0.015
vmax_degdp = 0.0
vmin_de = -10.0
vmax_de = -0.01
ticks_de = [-0.01, -0.1, -1.0, -10]

start_year_avg = 2000  # 2000,2007
end_year_avg = 2019

# Timeseries parameters
ylim_top_dy = 0.3
ylim_bottom_dy = -4.0
ylim_top_dp = 5.0
ylim_bottom_dp = -70
ylim_top_degdp = 0.005
ylim_bottom_degdp = -0.05
ylim_top_de = 2.0
ylim_bottom_de = -15
start_year_plot = 2000
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
smooth_method = "numpy"  # options: numpy, pandas, scipy
np_mode = "same"  # full, same, valid

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

df_dy_all = []
df_dp_all = []
df_degdp_all = []
df_de_all = []

for crop in crops:
    dy_file = f"{root_dir}/historical/linregress_outputs/{crop}/carbmaj/dy_carbmaj_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    dp_file = f"{root_dir}/historical/linregress_outputs/{crop}/carbmaj/dp_carbmaj_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    degdp_file = f"{root_dir}/historical/linregress_outputs/{crop}/carbmaj/degdp_carbmaj_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    de_file = f"{root_dir}/historical/linregress_outputs/{crop}/carbmaj/de_carbmaj_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"

    # Read Data
    df_dy = pd.read_csv(dy_file)
    df_dp = pd.read_csv(dp_file)
    df_degdp = pd.read_csv(degdp_file)
    df_de = pd.read_csv(de_file)

    # Add crop column
    df_dy['crop'] = crop
    df_dp['crop'] = crop
    df_degdp['crop'] = crop
    df_de['crop'] = crop

    # Append to lists
    df_dy_all.append(df_dy)
    df_dp_all.append(df_dp)
    df_degdp_all.append(df_degdp)
    df_de_all.append(df_de)

# Combine all crops data
df_dy = pd.concat(df_dy_all, ignore_index=True)
df_dp = pd.concat(df_dp_all, ignore_index=True)
df_degdp = pd.concat(df_degdp_all, ignore_index=True)
df_de = pd.concat(df_de_all, ignore_index=True)

# Get valid countries
hist_file = f"{root_dir}/historical/linregress_inputs/INPUT_HISTORICAL_DETRENDED_DEMEANED.csv"
df_hist = pd.read_csv(hist_file)
valid_countries = df_hist["country"].unique()

# Filter for valid countries
df_dy = df_dy[df_dy['country'].isin(valid_countries)]
df_dp = df_dp[df_dp['country'].isin(valid_countries)]
df_degdp = df_degdp[df_degdp['country'].isin(valid_countries)]
df_de = df_de[df_de['country'].isin(valid_countries)]

# Remove unreasonable data
df_dy = df_dy[df_dy['country'] != 'BGD']
df_dp = df_dp[df_dp['country'] != 'BGD']
df_degdp = df_degdp[df_degdp['country'] != 'BGD']
df_de = df_de[df_de['country'] != 'BGD']

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

cols_ts = ["country"] + [str(x) for x in range(start_year_fut, end_year_fut + 1)]
dy_all_ts = df_dy[cols_ts]
dp_all_ts = df_dp[cols_ts]
degdp_all_ts = df_degdp[cols_ts]
de_all_ts = df_de[cols_ts]

# Calculate global means/sums for timeseries
dy_global_mean = dy_all_ts.iloc[:, 1:].mean()
degdp_global_mean = degdp_all_ts.iloc[:, 1:].mean()
dp_global_sum = dp_all_ts.iloc[:, 1:].sum()
de_global_sum = de_all_ts.iloc[:, 1:].sum()

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

# Calculate regional means/sums for timeseries
dy_ipcc_melted = dy_ipcc.melt(id_vars=['country', ar6_region], var_name='year', value_name='value')
dp_ipcc_melted = dp_ipcc.melt(id_vars=['country', ar6_region], var_name='year', value_name='value')
degdp_ipcc_melted = degdp_ipcc.melt(id_vars=['country', ar6_region], var_name='year', value_name='value')
de_ipcc_melted = de_ipcc.melt(id_vars=['country', ar6_region], var_name='year', value_name='value')

dy_ipcc_stat = dy_ipcc_melted.groupby([ar6_region, 'year'], as_index=False)['value'].mean()
dy_ipcc_stat.rename(columns={ar6_region: 'region'}, inplace=True)

degdp_ipcc_stat = degdp_ipcc_melted.groupby([ar6_region, 'year'], as_index=False)['value'].mean()
degdp_ipcc_stat.rename(columns={ar6_region: 'region'}, inplace=True)
dp_ipcc_stat = dp_ipcc_melted.groupby([ar6_region, 'year'], as_index=False)['value'].sum()
dp_ipcc_stat.rename(columns={ar6_region: 'region'}, inplace=True)
de_ipcc_stat = de_ipcc_melted.groupby([ar6_region, 'year'], as_index=False)['value'].sum()
de_ipcc_stat.rename(columns={ar6_region: 'region'}, inplace=True)

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

# IPCC REGIONS
for region in regions:
    values_ipcc = dp_ipcc_stat[dp_ipcc_stat["region"] == region]["value"] / 1e6
    smoothed_values_ipcc = utils.get_smoothed_values(values_ipcc, window_size=window_size, method=smooth_method,
                                               np_mode=np_mode)
    ax6.plot(years_plot, smoothed_values_ipcc[plot_start_idx:plot_end_idx],
             color=ipcc_colors[region], linestyle="solid", linewidth=0.8, label=ipcc_labels[region])

    dp_region_list.append(smoothed_values_ipcc[plot_start_idx:plot_end_idx])

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

# IPCC REGIONS
for region in regions:
    values_ipcc = de_ipcc_stat[de_ipcc_stat["region"] == region]["value"] / 1e9
    smoothed_values_ipcc = utils.get_smoothed_values(values_ipcc, window_size=window_size, method=smooth_method,
                                               np_mode=np_mode)
    ax8.plot(years_plot, smoothed_values_ipcc[plot_start_idx:plot_end_idx],
             color=ipcc_colors[region], linestyle="solid", linewidth=0.8, label=ipcc_labels[region])

    de_region_list.append(smoothed_values_ipcc[plot_start_idx:plot_end_idx])

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
dy_global_mean = dy_global_vals.mean()
dy_ldc_mean = np.mean(dy_region_list[0])
dy_developing_mean = np.mean(dy_region_list[1])
dy_dev_mean = np.mean(dy_region_list[2])
degdp_global_mean = degdp_global_vals.mean()
degdp_ldc_mean = np.mean(degdp_region_list[0])
degdp_developing_mean = np.mean(degdp_region_list[1])
degdp_dev_mean = np.mean(degdp_region_list[2])
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
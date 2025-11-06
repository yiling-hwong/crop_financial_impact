# -*- coding: utf-8 -*-
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

###################### PLOTTING PARAMETERS ######################
# Map parameters
cmap_de = "Blues_r"
cmap_degdp = "Purples_r"
vmin_degdp = -0.03
vmax_degdp = 0.0
vmin_de = -5
vmax_de = -0.01
start_year_avg = 2000 #2007, 2000
end_year_avg = 2019

# Timeseries parameters
ylim_top_degdp = 0.02
ylim_bottom_degdp = -0.25
ylim_top_de = 3.0
ylim_bottom_de = -35
start_year_plot = 2000 #1990,2000
end_year_plot = 2019

label_de = "Economic loss [B US$]"
label_degdp = "GDP loss [%]"

window_size = 5
smooth_method = "numpy"
np_mode = "full"

ipcc_colors = {"ldc":"crimson","developing":"goldenrod","developed":"dodgerblue"} #ldc,developing,developed
ipcc_labels = {"ldc": "LDC", "developing": "Developing", "developed": "Developed"}
labels_alphabets = utils.generate_alphabet_list(4, option="lower")

quantile_low = 0.25
quantile_high = 0.75
##################################################################

ar6_region = "region_ar6_dev"
regions = ["ldc","developing","developed"]

# For input file names
start_year_hist = 1971
end_year_hist = 1989
start_year_fut = 1990
end_year_fut = 2019

root_dir = '../../data'
country_shape_file = f"{root_dir}/resources/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

df_degdp_all = []
df_de_all = []
df_degdp_corey_all = []
df_de_corey_all = []

for crop in crops:
    degdp_file = f"{root_dir}/historical/linregress_outputs/{crop}/isimip3a/degdp_isimip3a_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    de_file = f"{root_dir}/historical/linregress_outputs/{crop}/isimip3a/de_isimip3a_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"

    # Read Data
    df_degdp = pd.read_csv(degdp_file)
    df_de = pd.read_csv(de_file)

    # Add crop column
    df_degdp['crop'] = crop
    df_de['crop'] = crop

    # Append to lists
    df_degdp_all.append(df_degdp)
    df_de_all.append(df_de)

# Combine all crops data
df_degdp = pd.concat(df_degdp_all, ignore_index=True)
df_de = pd.concat(df_de_all, ignore_index=True)

# Get valid countries
hist_file = f"{root_dir}/historical/linregress_inputs/INPUT_HISTORICAL_DETRENDED_DEMEANED.csv"
df_hist = pd.read_csv(hist_file)
valid_countries = df_hist["country"].unique()

# Filter for valid countries
df_degdp = df_degdp[df_degdp['country'].isin(valid_countries)]
df_de = df_de[df_de['country'].isin(valid_countries)]

# Remove unreasonable data
df_degdp = df_degdp[df_degdp['country'] != 'BGD']
df_de = df_de[df_de['country'] != 'BGD']

# Prepare data for maps
year_str = [str(x) for x in range(start_year_avg, end_year_avg + 1)]
cols = ["country"] + year_str

degdp_all = df_degdp[cols]
de_all = df_de[cols]

# Calculate averages for each country
degdp_all["degdp_avg"] = degdp_all.loc[:, f"{start_year_avg}":f"{end_year_avg}"].mean(axis=1)
de_all[f"de_sum"] = de_all.loc[:, f"{start_year_avg}":f"{end_year_avg}"].sum(axis=1)

df1 = degdp_all[["country", "degdp_avg"]]
df2 = de_all[["country", f"de_sum"]]

# Average or sum across all crops and per country
df1 = df1.groupby("country", as_index=False)["degdp_avg"].mean()
df2 = df2.groupby("country", as_index=False)[f"de_sum"].sum()

# Load the world map shapefile
world = gpd.read_file(country_shape_file)
world = world.merge(df1.merge(df2, on="country"), how="left", left_on="ADM0_A3", right_on="country")

degdp_max = world["degdp_avg"].max()
degdp_min = world["degdp_avg"].min()
de_max = (world[f"de_sum"]/1e9).max()
de_min = (world[f"de_sum"]/1e9).min()

print("----------------")
print ("degdp max and min:",degdp_max,degdp_min)
print ("de max and min:",de_max,de_min)

# Prepare data for timeseries
years = [x for x in range(start_year_fut, end_year_fut + 1)]
years_plot = [x for x in range(start_year_plot, end_year_plot + 1)]
plot_start_idx = years.index(start_year_plot)
plot_end_idx = years.index(end_year_plot) + 1

cols_ts = ["country"] + [str(x) for x in range(start_year_fut, end_year_fut + 1)]
degdp_all_ts = df_degdp[cols_ts]
de_all_ts = df_de[cols_ts]

# Calculate global means/sums for timeseries
degdp_global = degdp_all_ts.iloc[:, 1:].mean()
de_global = de_all_ts.iloc[:, 1:].sum()

"""
GET IPCC REGION DATA
"""
print ("----------------")
print("Getting ipcc region:", ar6_region)
df_ipcc = utils.get_ipcc_region_df()
df_ipcc = df_ipcc[["Country", ar6_region]]
df_ipcc = df_ipcc.rename(columns={"Country": "country"})

# Merge with IPCC regions
degdp_ipcc = degdp_all_ts.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
de_ipcc = de_all_ts.merge(df_ipcc[["country", ar6_region]], on="country", how="left")

# Calculate regional means/sums for timeseries
degdp_ipcc_melted = degdp_ipcc.melt(id_vars=['country', 'region_ar6_dev'], var_name='year', value_name='value')
de_ipcc_melted = de_ipcc.melt(id_vars=['country', 'region_ar6_dev'], var_name='year', value_name='value')

degdp_ipcc_mean = degdp_ipcc_melted.groupby(['region_ar6_dev', 'year'], as_index=False)['value'].mean()
degdp_ipcc_mean.rename(columns={'region_ar6_dev': 'region'}, inplace=True)
de_ipcc_stat = de_ipcc_melted.groupby(['region_ar6_dev', 'year'], as_index=False)['value'].sum()
de_ipcc_stat.rename(columns={'region_ar6_dev': 'region'}, inplace=True)

"""
PLOT
"""
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
title_fontsize = 12
label_fontsize = 12
tick_fontsize = 12

######################## degdp MAP
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
var = "degdp_avg"

world_plot = world.copy()

# Reproject world geometries to Robinson
world_plot = world_plot.to_crs(ccrs.Robinson().proj4_init)
world_plot.plot(ax=ax1, color='white', edgecolor='gray', linewidth=0.5)

# Create a ScalarMappable for the colorbar
norm = plt.Normalize(vmin=vmin_degdp, vmax=vmax_degdp)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_degdp)
sm.set_array([])

# Plot the data
plot = world_plot.plot(column=var, ax=ax1, cmap=cmap_degdp,
                       vmin=vmin_degdp, vmax=vmax_degdp,
                       missing_kwds={"color": "white", "label": "No Data"})

# Add colorbar at the bottom
pos = ax1.get_position()
cax = fig.add_axes([pos.x0 + 0.04, pos.y0 + 0.07, pos.width - 0.07, 0.01])
cbar = plt.colorbar(sm, cax=cax, orientation='horizontal', extend='both')
cbar.set_label(label_degdp, size=label_fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)

ax1.text(0.0, 1.15, "a", transform=ax1.transAxes,fontsize=label_fontsize+1, fontweight='bold', va='top', ha='right')
ax1.set_global()
ax1.gridlines(alpha=0.2)

######################## de MAP
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
var = f"de_sum"

world_plot = world.copy()

# Reproject world geometries to Robinson
world_plot = world_plot.to_crs(ccrs.Robinson().proj4_init)
world_plot.plot(ax=ax2, color='white', edgecolor='gray', linewidth=0.5)
world_plot[var] = world_plot[var] / 1e9  # convert to Mt

norm = colors.SymLogNorm(linthresh=0.01, linscale=0.01, vmin=vmin_de, vmax=vmax_de, base=10)
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_de)

# Plot the data
plot = world_plot.plot(column=var, ax=ax2, cmap=cmap_de,
                       norm=norm,
                       vmin=vmin_de, vmax=vmax_de,
                       missing_kwds={"color": "white", "label": "No Data"})

# Add colorbar at the bottom
pos = ax2.get_position()
cax_de = fig.add_axes([pos.x0 + 0.09, pos.y0 + 0.07, pos.width - 0.07, 0.01]) #left, bottom, width, height
cbar = plt.colorbar(sm, cax=cax_de, orientation='horizontal', extend='both'
                    , format=utils.format_fn
                    )
cbar.set_label(label_de, size=label_fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)

ax2.text(0.0, 1.15, "b", transform=ax2.transAxes,fontsize=label_fontsize+1, fontweight='bold', va='top', ha='right')
ax2.set_global()
ax2.gridlines(alpha=0.2)

######################## degdp TIMESERIES
degdp_region_list = []
ax3 = fig.add_subplot(gs[1, 0])
values = degdp_global.values
values = np.nan_to_num(values, nan=0)
smoothed_values = utils.get_smoothed_values(values, window_size=window_size,method=smooth_method, np_mode=np_mode)
line1, = ax3.plot(years_plot, smoothed_values[plot_start_idx:plot_end_idx],
                  color="black", linestyle='-', linewidth=1.2, label=f"Global")

# Add global confidence intervals for 2019
year_2019 = years_plot[-1]
degdp_2019 = degdp_ipcc_melted[degdp_ipcc_melted['year'] == "2019"]
degdp_mean = degdp_2019["value"].mean()
ci_low = degdp_2019['value'].quantile(quantile_low)
ci_high = degdp_2019['value'].quantile(quantile_high)
ci_low_interval = abs(degdp_mean - ci_low)
ci_high_interval = abs(degdp_mean - ci_high)
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
    values_ipcc = degdp_ipcc_mean[degdp_ipcc_mean["region"] == region]["value"]
    smoothed_values_ipcc = utils.get_smoothed_values(values_ipcc, window_size=window_size, method=smooth_method, np_mode=np_mode)
    smoothed_values_ipcc = np.nan_to_num(smoothed_values_ipcc, nan=0)  # convert NaN to zeros
    line3, = ax3.plot(years_plot, smoothed_values_ipcc[plot_start_idx:plot_end_idx],
                      color=ipcc_colors[region], linestyle="solid", linewidth=0.8, label=ipcc_labels[region])

    degdp_region_list.append(values_ipcc[plot_start_idx:plot_end_idx])

    # Add confidence intervals for 2019
    year_2019 = years_plot[-1]
    degdp_2019 = degdp_ipcc_melted[degdp_ipcc_melted['year'] == "2019"]

    # Add CI for each region
    degdp_region = degdp_2019[degdp_2019['region_ar6_dev'].str.lower() == region.lower()]['value']

    if len(degdp_region) > 0:  # Only plot if we have data
        region_mean = degdp_region.mean()
        ci_low = degdp_region.quantile(quantile_low)
        ci_high = degdp_region.quantile(quantile_high)
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

if ylim_top_degdp != None:
    ax3.set_ylim(top=ylim_top_degdp)
if ylim_bottom_degdp != None:
    ax3.set_ylim(bottom=ylim_bottom_degdp)

ax3.set_xlim(right=end_year_plot+0.5)
ax3.text(0.0, 1.25, "c", transform=ax3.transAxes,fontsize=label_fontsize+1, fontweight='bold', va='top', ha='right')
ax3.axhline(0, color="black", linestyle="dashed", lw=0.9)
ax3.set_xlabel('Year', fontsize=label_fontsize)
ax3.set_ylabel(label_degdp, fontsize=label_fontsize)
ax3.tick_params(axis='both', labelsize=tick_fontsize)
ax3.grid(True)

######################## de TIMESERIES
de_region_list = []
ax4 = fig.add_subplot(gs[1, 1])
values = de_global.values / 1e9
smoothed_values = utils.get_smoothed_values(values, window_size=window_size,
                                      method=smooth_method, np_mode=np_mode)
ax4.plot(years_plot, smoothed_values[plot_start_idx:plot_end_idx],
         color="black", linestyle='-', linewidth=1.2, label=f"Global")

# IPCC REGIONS
for region in regions:
    values_ipcc = de_ipcc_stat[de_ipcc_stat["region"] == region]["value"] / 1e9
    smoothed_values_ipcc = utils.get_smoothed_values(values_ipcc, window_size=window_size, method=smooth_method,
                                               np_mode=np_mode)
    line4, = ax4.plot(years_plot, smoothed_values_ipcc[plot_start_idx:plot_end_idx],
                      color=ipcc_colors[region], linestyle="solid", linewidth=0.8, label=ipcc_labels[region])
    de_region_list.append(values_ipcc[plot_start_idx:plot_end_idx])

if ylim_top_de != None:
    ax4.set_ylim(top=ylim_top_de)
if ylim_bottom_de != None:
    ax4.set_ylim(bottom=ylim_bottom_de)

ax4.set_xlim(right=end_year_plot+0.5)
ax4.text(0.0, 1.25, "d", transform=ax4.transAxes,fontsize=label_fontsize+1, fontweight='bold', va='top', ha='right')
ax4.axhline(0, color="black", linestyle="dashed", lw=0.9)
ax4.set_xlabel('Year', fontsize=label_fontsize)
ax4.set_ylabel(label_de, fontsize=label_fontsize)
ax4.tick_params(axis='both', labelsize=tick_fontsize)
ax4.grid(True)

# LEGEND
ax4.legend(bbox_to_anchor=(-0.2, -0.6), loc='center',fontsize=tick_fontsize-1, ncol=4)

fig.subplots_adjust(left=0.11,
                    bottom=0.18,
                    right=0.97,  # 0.83 (with legend), 0.9 (no legend)
                    top=0.99,
                    wspace=0.25,
                    hspace=0.35)

plt.show()

"""
PRINT DIAGNOSTICS
"""
print("\n" + "="*50)
print("DEGDP top 10 countries")
print("="*50)
degdp_top10 = df1.sort_values(by='degdp_avg', ascending=True).head(10)
print (degdp_top10)

print("\n" + "="*50)
print("DE top 10 countries")
print("="*50)
de_top10 = df2.sort_values(by='de_sum', ascending=True).head(10)
print (de_top10)

#----------------------- ANALYZE value of specific country
start_year = 2000
end_year = 2019
country = "ALL" #ALL, MOZ,CAF,MLI, MWI; CHN, BRA, FRA; GNB, ETH, TZA

year_str = [str(x) for x in range(start_year, end_year + 1)]
cols = ["country"] + year_str

degdp_years = df_degdp[cols]
de_years = df_de[cols]

de_allcrop = de_years.groupby('country').sum(numeric_only=True).reset_index()
degdp_allcrop = degdp_years.groupby('country').mean(numeric_only=True).reset_index()

de_allcrop[f"de_sum"] = de_allcrop.loc[:, f"{start_year}":f"{end_year}"].sum(axis=1)
degdp_allcrop["degdp_avg"] = degdp_allcrop.loc[:, f"{start_year}":f"{end_year}"].mean(axis=1)

if country == "ALL":
    de_cntry = de_allcrop["de_sum"].sum()
    degdp_cntry = degdp_allcrop["degdp_avg"].mean()
    de_cntry = de_cntry / 1e9
else:
    de_cntry = de_allcrop[de_allcrop["country"] == country]
    degdp_cntry = degdp_allcrop[degdp_allcrop["country"] == country]
    de_cntry = de_cntry["de_sum"].values / 1e9
    degdp_cntry = degdp_cntry["degdp_avg"].values
    de_cntry = de_cntry[0]
    degrp_cntry = degdp_cntry[0]

# GET REGION DATA
print ()
print ("--------------------")
degdp_ldc_mean = degdp_region_list[0].mean()
degdp_developing_mean = degdp_region_list[1].mean()
degdp_dev_mean = degdp_region_list[2].mean()

de_ldc_sum = de_region_list[0].sum()
de_developing_sum = de_region_list[1].sum()
de_dev_sum = de_region_list[2].sum()
print ("LENGTHS:",len(degdp_region_list[0]),len(de_region_list[0]))
print (f"TOTAL DE for LDC, DEVELOPING, DEVELOPED over {start_year}-{end_year} (B US$): {de_ldc_sum}, {de_developing_sum}, {de_dev_sum}")
print (f"MEAN DEGDP for LDC, DEVELOPING, DEVELOPED over {start_year}-{end_year} (%): {degdp_ldc_mean}, {degdp_developing_mean}, {degdp_dev_mean}")

print ()
print ("--------------------")
print (f"TOTAL DE for {country} over {start_year}-{end_year} (B US$): {de_cntry}")
print (f"MEAN DEGDP for {country} over {start_year}-{end_year} (%GDP): {degdp_cntry}")
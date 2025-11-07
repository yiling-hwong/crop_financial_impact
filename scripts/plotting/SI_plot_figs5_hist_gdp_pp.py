# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 300
import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from scipy.stats import t
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('../')
import utils

"""
PARAMETERS
"""
crops = ["maize","wheat","soy"]
ar6_region = "region_ar6_dev"
regions = ["ldc", "developing", "developed"]

root_dir = '../../data'
gdp_file = f"{root_dir}/resources/gdp_world_bank_2015_usd.csv"

######################## PLOTTING PARAMETERS ########################
ylim_top_gdp = 8000.0
ylim_bottom_gdp = 0
ylim_top_pp = 1.8
ylim_bottom_pp = -0.1

label_gdp = f"GDP [B US\$]"
label_pp = f"Producer price [k US\$/ton]"

ipcc_colors = {"ldc": "crimson", "developing": "goldenrod", "developed": "dodgerblue"}  # ldc,developing,developed
ipcc_labels = {"ldc": "LDC", "developing": "Developing", "developed": "Developed"}
labels_alphabets = utils.generate_alphabet_list(4, option="lower")

quantile_low = 0.10
quantile_high = 0.90
#####################################################################

start_year = 2000
end_year = 2019
years = [x for x in range(start_year, end_year + 1)]
plot_start_idx = years.index(start_year)
plot_end_idx = years.index(end_year) + 1

window_size = 1
smooth_method = "numpy"
np_mode = "full"

"""
GET GDP
"""
gdp = pd.read_csv(gdp_file)
gdp = gdp.drop(columns = ["Country Name","Indicator Name","Indicator Code","Unnamed: 68"])
gdp = gdp.rename(columns={"Country Code": "country"})
gdp = gdp.set_index("country")

"""
GET PRODUCER PRICES
"""
faopp_maize = utils.get_fao_producer_prices("maize",start_year,end_year)
faopp_wheat = utils.get_fao_producer_prices("wheat",start_year,end_year)
faopp_soy = utils.get_fao_producer_prices("soy",start_year,end_year)

df_maize = faopp_maize.pivot(index="iso",columns="year",values="pp")
df_maize.index.name = 'country'
df_maize.columns = df_maize.columns.astype(int)
df_maize = df_maize.sort_index(axis=1)
df_maize.columns.name = None

df_wheat = faopp_wheat.pivot(index="iso",columns="year",values="pp")
df_wheat.index.name = 'country'
df_wheat.columns = df_wheat.columns.astype(int)
df_wheat = df_wheat.sort_index(axis=1)
df_wheat.columns.name = None

df_soy = faopp_soy.pivot(index="iso",columns="year",values="pp")
df_soy.index.name = 'country'
df_soy.columns = df_soy.columns.astype(int)
df_soy = df_soy.sort_index(axis=1)
df_soy.columns.name = None

"""
GET IPCC REGIONS
"""
print("----------------")
print("Getting ipcc region:", ar6_region)
df_ipcc = utils.get_ipcc_region_df()
df_ipcc = df_ipcc[["Country", ar6_region]]
df_ipcc = df_ipcc.rename(columns={"Country": "country"})

gdp_ipcc = gdp.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
maize_ipcc = df_maize.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
wheat_ipcc = df_wheat.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
soy_ipcc = df_soy.merge(df_ipcc[["country", ar6_region]], on="country", how="left")

#---GDP
gdp_ipcc_melted = gdp_ipcc.melt(id_vars=['country', 'region_ar6_dev'], var_name='year', value_name='value')
gdp_ipcc_melted['year'] = gdp_ipcc_melted['year'].astype(int)  # Convert year to integer
gdp_ipcc_melted = gdp_ipcc_melted[(gdp_ipcc_melted['year'] >= start_year) & (gdp_ipcc_melted['year'] <= end_year)]
gdp_ipcc_melted["value"] = gdp_ipcc_melted["value"] / 1e9
df_gdp = gdp_ipcc_melted.groupby(['region_ar6_dev', 'year'], as_index=False)['value'].mean()
df_gdp.rename(columns={'region_ar6_dev': 'region'}, inplace=True)

#---MAIZE
maize_ipcc_melted = maize_ipcc.melt(id_vars=['country', 'region_ar6_dev'], var_name='year', value_name='value')
maize_ipcc_melted['year'] = maize_ipcc_melted['year'].astype(int)  # Convert year to integer
maize_ipcc_melted = maize_ipcc_melted[(maize_ipcc_melted['year'] >= start_year) & (maize_ipcc_melted['year'] <= end_year)]
maize_ipcc_melted["value"] = maize_ipcc_melted["value"] / 1e3
df_maize = maize_ipcc_melted.groupby(['region_ar6_dev', 'year'], as_index=False)['value'].mean()
df_maize.rename(columns={'region_ar6_dev': 'region'}, inplace=True)

#---WHEAT
wheat_ipcc_melted = wheat_ipcc.melt(id_vars=['country', 'region_ar6_dev'], var_name='year', value_name='value')
wheat_ipcc_melted['year'] = wheat_ipcc_melted['year'].astype(int)  # Convert year to integer
wheat_ipcc_melted = wheat_ipcc_melted[(wheat_ipcc_melted['year'] >= start_year) & (wheat_ipcc_melted['year'] <= end_year)]
wheat_ipcc_melted["value"] = wheat_ipcc_melted["value"] / 1e3
df_wheat = wheat_ipcc_melted.groupby(['region_ar6_dev', 'year'], as_index=False)['value'].mean()
df_wheat.rename(columns={'region_ar6_dev': 'region'}, inplace=True)

#---SOY
soy_ipcc_melted = soy_ipcc.melt(id_vars=['country', 'region_ar6_dev'], var_name='year', value_name='value')
soy_ipcc_melted['year'] = soy_ipcc_melted['year'].astype(int)  # Convert year to integer
soy_ipcc_melted = soy_ipcc_melted[(wheat_ipcc_melted['year'] >= start_year) & (soy_ipcc_melted['year'] <= end_year)]
soy_ipcc_melted["value"] = soy_ipcc_melted["value"] / 1e3
df_soy = soy_ipcc_melted.groupby(['region_ar6_dev', 'year'], as_index=False)['value'].mean()
df_soy.rename(columns={'region_ar6_dev': 'region'}, inplace=True)

"""
PLOT
"""
fig = plt.figure(figsize=(10, 6))
title_fontsize = 14
label_fontsize = 12
tick_fontsize = 12

######################## GDP TIMESERIES
ax1 = fig.add_subplot(221)
values = df_gdp.groupby('year')['value'].mean().reset_index()
values = values["value"]

smoothed_values = utils.get_smoothed_values(values, window_size=window_size,
                                      method=smooth_method, np_mode=np_mode)
line1, = ax1.plot(years, smoothed_values[plot_start_idx:plot_end_idx],
                  color="black", linestyle='-', linewidth=0.8, label="Global")

# Add global confidence intervals for 2019
year_2019 = years[-1]
gdp_2019 = gdp_ipcc_melted[gdp_ipcc_melted['year'] == 2019]
gdp_mean = gdp_2019["value"].mean()
ci_low = gdp_2019['value'].quantile(quantile_low)
ci_high = gdp_2019['value'].quantile(quantile_high)
ci_low_interval = abs(gdp_mean - ci_low)
ci_high_interval = abs(gdp_mean - ci_high)
final_val = smoothed_values[plot_end_idx - 1]
ci_low = final_val - ci_low_interval
ci_high = final_val + ci_high_interval
print ("----------------")
print ("GLOBAL 2019 mean GDP:",gdp_mean)
#print("Global GDP CI:", ci_low, ci_high, "final_val:", final_val)

# Plot global CI
offsets = {'global': -0.2, 'ldc': 0.0, 'developing': 0.2, 'developed': 0.4}
ax1.vlines(year_2019 + offsets['global'], ci_low, ci_high,
           color="black", linestyle='-', linewidth=0.8)
ax1.plot(year_2019 + offsets['global'], final_val, 'o',
         color="black", markersize=1)

for region in regions:
    values_ipcc = df_gdp[df_gdp["region"] == region]["value"]
    smoothed_values_ipcc = utils.get_smoothed_values(values_ipcc, window_size=window_size, method=smooth_method,
                                               np_mode=np_mode)
    line1, = ax1.plot(years, smoothed_values_ipcc[plot_start_idx:plot_end_idx],
                      color=ipcc_colors[region], linestyle="solid", linewidth=0.8, label=ipcc_labels[region])

    # Add confidence intervals for 2019
    year_2019 = years[-1]
    gdp_2019 = gdp_ipcc_melted[gdp_ipcc_melted['year'] == 2019]

    # Add CI for each region
    gdp_region = gdp_2019[gdp_2019['region_ar6_dev'].str.lower() == region.lower()]['value']
    if len(gdp_region) > 0:
        print(f"Region {region} GDP value range:", gdp_region.min(), gdp_region.max())
        region_mean = gdp_region.mean()
        ci_low = gdp_region.quantile(quantile_low)
        ci_high = gdp_region.quantile(quantile_high)
        ci_low_interval = abs(region_mean-ci_low)
        ci_high_interval = abs(region_mean-ci_high)
        final_val = smoothed_values_ipcc[plot_end_idx - 1]
        ci_low = final_val - ci_low_interval
        ci_high = final_val + ci_high_interval
        print(f"Region {region} 2019 mean GDP:", region_mean)
        #print(f"Region {region} dy CI:", ci_low, ci_high, "final_val:", final_val)

        #Plot vertical line with CI
        ax1.vlines(year_2019 + offsets[region.lower()], ci_low, ci_high,
                   color=ipcc_colors[region], linestyle='-', linewidth=0.8)
        ax1.plot(year_2019 + offsets[region.lower()], final_val, 'o',
                 color=ipcc_colors[region], markersize=1)

ax1.set_yscale("log")

if ylim_top_gdp != None:
    ax1.set_ylim(top=ylim_top_gdp)
if ylim_bottom_gdp != None:
    ax1.set_ylim(bottom=ylim_bottom_gdp)

ax1.set_xlim(right=end_year+0.6)
ax1.set_title("GDP", pad=10, fontsize=title_fontsize)
ax1.text(0.0, 1.2, "a", transform=ax1.transAxes,fontsize=label_fontsize+1, fontweight='bold', va='top', ha='right')
ax1.axhline(0, color="black", linestyle="dashed", lw=0.9)
ax1.set_xlabel('Year', fontsize=label_fontsize)
ax1.set_ylabel(label_gdp, fontsize=label_fontsize)
ax1.tick_params(axis='both', labelsize=tick_fontsize)
ax1.grid(True)

######################## MAIZE PP TIMESERIES
ax2 = fig.add_subplot(222)
values = df_maize.groupby('year')['value'].mean().reset_index()
values = values["value"]

smoothed_values = utils.get_smoothed_values(values, window_size=window_size,
                                      method=smooth_method, np_mode=np_mode)
line2, = ax2.plot(years, smoothed_values[plot_start_idx:plot_end_idx],
                  color="black", linestyle='-', linewidth=0.8, label="Global")

# Add global confidence intervals for 2019
year_2019 = years[-1]
maize_2019 = maize_ipcc_melted[maize_ipcc_melted['year'] == 2019]
maize_mean = maize_2019["value"].mean()
ci_low = maize_2019['value'].quantile(quantile_low)
ci_high = maize_2019['value'].quantile(quantile_high)
ci_low_interval = abs(maize_mean - ci_low)
ci_high_interval = abs(maize_mean - ci_high)
final_val = smoothed_values[plot_end_idx - 1]
ci_low = final_val - ci_low_interval
ci_high = final_val + ci_high_interval
print ("----------------")
print ("GLOBAL 2019 mean MAIZE PP:",maize_mean)
#print("Global MAIZE CI:", ci_low, ci_high, "final_val:", final_val)

# Plot global CI
offsets = {'global': -0.2, 'ldc': 0.0, 'developing': 0.2, 'developed': 0.4}
ax2.vlines(year_2019 + offsets['global'], ci_low, ci_high,
           color="black", linestyle='-', linewidth=0.8)
ax2.plot(year_2019 + offsets['global'], final_val, 'o',
         color="black", markersize=1)

for region in regions:
    values_ipcc = df_maize[df_maize["region"] == region]["value"]
    smoothed_values_ipcc = utils.get_smoothed_values(values_ipcc, window_size=window_size, method=smooth_method,
                                               np_mode=np_mode)
    line3, = ax2.plot(years, smoothed_values_ipcc[plot_start_idx:plot_end_idx],
                      color=ipcc_colors[region], linestyle="solid", linewidth=0.8, label=ipcc_labels[region])

    # Add confidence intervals for 2019
    year_2019 = years[-1]
    maize_2019 = maize_ipcc_melted[maize_ipcc_melted['year'] == 2019]

    # Add CI for each region
    maize_region = maize_2019[maize_2019['region_ar6_dev'].str.lower() == region.lower()]['value']
    if len(maize_region) > 0:
        print(f"Region {region} MAIZE PP value range:", maize_region.min(), maize_region.max())
        region_mean = maize_region.mean()
        ci_low = maize_region.quantile(quantile_low)
        ci_high = maize_region.quantile(quantile_high)
        ci_low_interval = abs(region_mean-ci_low)
        ci_high_interval = abs(region_mean-ci_high)
        final_val = smoothed_values_ipcc[plot_end_idx - 1]
        ci_low = final_val - ci_low_interval
        ci_high = final_val + ci_high_interval
        print(f"Region {region} 2019 mean MAIZE PP:", region_mean)
        #print(f"Region {region} MAIZE CI:", ci_low, ci_high, "final_val:", final_val)

        #Plot vertical line with CI
        ax2.vlines(year_2019 + offsets[region.lower()], ci_low, ci_high,
                   color=ipcc_colors[region], linestyle='-', linewidth=0.8)
        ax2.plot(year_2019 + offsets[region.lower()], final_val, 'o',
                 color=ipcc_colors[region], markersize=1)

if ylim_top_pp != None:
    ax2.set_ylim(top=ylim_top_pp)
if ylim_bottom_pp != None:
    ax2.set_ylim(bottom=ylim_bottom_pp)

ax2.set_xlim(right=end_year+0.6)
ax2.set_title("Maize", pad=10, fontsize=title_fontsize)
ax2.text(0.0, 1.2, "b", transform=ax2.transAxes,fontsize=label_fontsize+1, fontweight='bold', va='top', ha='right')
ax2.axhline(0, color="black", linestyle="dashed", lw=0.9)
ax2.set_xlabel('Year', fontsize=label_fontsize)
ax2.set_ylabel(label_pp, fontsize=label_fontsize)
ax2.tick_params(axis='both', labelsize=tick_fontsize)
ax2.grid(True)

######################## WHEAT PP TIMESERIES
ax3 = fig.add_subplot(223)
values = df_wheat.groupby('year')['value'].mean().reset_index()
values = values["value"]

smoothed_values = utils.get_smoothed_values(values, window_size=window_size,
                                      method=smooth_method, np_mode=np_mode)
line3, = ax3.plot(years, smoothed_values[plot_start_idx:plot_end_idx],
                  color="black", linestyle='-', linewidth=0.8, label="Global")

# Add global confidence intervals for 2019
year_2019 = years[-1]
wheat_2019 = wheat_ipcc_melted[wheat_ipcc_melted['year'] == 2019]
wheat_mean = wheat_2019["value"].mean()
ci_low = wheat_2019['value'].quantile(quantile_low)
ci_high = wheat_2019['value'].quantile(quantile_high)
ci_low_interval = abs(wheat_mean - ci_low)
ci_high_interval = abs(wheat_mean - ci_high)
final_val = smoothed_values[plot_end_idx - 1]
ci_low = final_val - ci_low_interval
ci_high = final_val + ci_high_interval
print ("----------------")
print ("GLOBAL 2019 mean WHEAT PP:",maize_mean)
#print("Global WHEAT CI:", ci_low, ci_high, "final_val:", final_val)

# Plot global CI
offsets = {'global': -0.2, 'ldc': 0.0, 'developing': 0.2, 'developed': 0.4}
ax3.vlines(year_2019 + offsets['global'], ci_low, ci_high,
           color="black", linestyle='-', linewidth=0.8)
ax3.plot(year_2019 + offsets['global'], final_val, 'o',
         color="black", markersize=1)

for region in regions:
    values_ipcc = df_wheat[df_wheat["region"] == region]["value"]
    smoothed_values_ipcc = utils.get_smoothed_values(values_ipcc, window_size=window_size, method=smooth_method,
                                               np_mode=np_mode)
    line3, = ax3.plot(years, smoothed_values_ipcc[plot_start_idx:plot_end_idx],
                      color=ipcc_colors[region], linestyle="solid", linewidth=0.8, label=ipcc_labels[region])

    # Add confidence intervals for 2019
    year_2019 = years[-1]
    wheat_2019 = wheat_ipcc_melted[wheat_ipcc_melted['year'] == 2019]

    # Add CI for each region
    wheat_region = wheat_2019[wheat_2019['region_ar6_dev'].str.lower() == region.lower()]['value']
    if len(wheat_region) > 0:
        print(f"Region {region} WHEAT PP value range:", wheat_region.min(), wheat_region.max())
        region_mean = wheat_region.mean()
        ci_low = wheat_region.quantile(quantile_low)
        ci_high = wheat_region.quantile(quantile_high)
        ci_low_interval = abs(region_mean-ci_low)
        ci_high_interval = abs(region_mean-ci_high)
        final_val = smoothed_values_ipcc[plot_end_idx - 1]
        ci_low = final_val - ci_low_interval
        ci_high = final_val + ci_high_interval
        print(f"Region {region} 2019 mean WHEAT PP:", region_mean)
        #print(f"Region {region} WHEAT CI:", ci_low, ci_high, "final_val:", final_val)

        #Plot vertical line with CI
        ax3.vlines(year_2019 + offsets[region.lower()], ci_low, ci_high,
                   color=ipcc_colors[region], linestyle='-', linewidth=0.8)
        ax3.plot(year_2019 + offsets[region.lower()], final_val, 'o',
                 color=ipcc_colors[region], markersize=1)

if ylim_top_pp != None:
    ax3.set_ylim(top=ylim_top_pp)
if ylim_bottom_pp != None:
    ax3.set_ylim(bottom=ylim_bottom_pp)

ax3.set_xlim(right=end_year+0.6)
ax3.set_title("Wheat", pad=10, fontsize=title_fontsize)
ax3.text(0.0, 1.2, "c", transform=ax3.transAxes,fontsize=label_fontsize+1, fontweight='bold', va='top', ha='right')
ax3.axhline(0, color="black", linestyle="dashed", lw=0.9)
ax3.set_xlabel('Year', fontsize=label_fontsize)
ax3.set_ylabel(label_pp, fontsize=label_fontsize)
ax3.tick_params(axis='both', labelsize=tick_fontsize)
ax3.grid(True)

######################## SOY PP TIMESERIES
ax4 = fig.add_subplot(224)
values = df_soy.groupby('year')['value'].mean().reset_index()
values = values["value"]

smoothed_values = utils.get_smoothed_values(values, window_size=window_size,
                                      method=smooth_method, np_mode=np_mode)
line4, = ax4.plot(years, smoothed_values[plot_start_idx:plot_end_idx],
                  color="black", linestyle='-', linewidth=0.8, label="Global")

# Add global confidence intervals for 2019
year_2019 = years[-1]
soy_2019 = soy_ipcc_melted[soy_ipcc_melted['year'] == 2019]
soy_mean = soy_2019["value"].mean()
ci_low = soy_2019['value'].quantile(quantile_low)
ci_high = soy_2019['value'].quantile(quantile_high)
ci_low_interval = abs(soy_mean - ci_low)
ci_high_interval = abs(soy_mean - ci_high)
final_val = smoothed_values[plot_end_idx - 1]
ci_low = final_val - ci_low_interval
ci_high = final_val + ci_high_interval
print ("----------------")
print ("GLOBAL 2019 mean SOY PP:",maize_mean)
#print("Global SOY CI:", ci_low, ci_high, "final_val:", final_val)

# Plot global CI
offsets = {'global': -0.2, 'ldc': 0.0, 'developing': 0.2, 'developed': 0.4}
ax4.vlines(year_2019 + offsets['global'], ci_low, ci_high,
           color="black", linestyle='-', linewidth=0.8)
ax4.plot(year_2019 + offsets['global'], final_val, 'o',
         color="black", markersize=1)

for region in regions:
    values_ipcc = df_soy[df_soy["region"] == region]["value"]
    smoothed_values_ipcc = utils.get_smoothed_values(values_ipcc, window_size=window_size, method=smooth_method,
                                               np_mode=np_mode)
    line4, = ax4.plot(years, smoothed_values_ipcc[plot_start_idx:plot_end_idx],
                      color=ipcc_colors[region], linestyle="solid", linewidth=0.8, label=ipcc_labels[region])

    # Add confidence intervals for 2019
    year_2019 = years[-1]
    soy_2019 = soy_ipcc_melted[soy_ipcc_melted['year'] == 2019]

    # Add CI for each region
    soy_region = soy_2019[soy_2019['region_ar6_dev'].str.lower() == region.lower()]['value']
    if len(soy_region) > 0:
        print(f"Region {region} SOY PP value range:", soy_region.min(), soy_region.max())
        region_mean = soy_region.mean()
        ci_low = soy_region.quantile(quantile_low)
        ci_high = soy_region.quantile(quantile_high)
        ci_low_interval = abs(region_mean-ci_low)
        ci_high_interval = abs(region_mean-ci_high)
        final_val = smoothed_values_ipcc[plot_end_idx - 1]
        ci_low = final_val - ci_low_interval
        ci_high = final_val + ci_high_interval
        print(f"Region {region} 2019 mean SOY PP:", region_mean)
        #print(f"Region {region} SOY CI:", ci_low, ci_high, "final_val:", final_val)

        #Plot vertical line with CI
        ax4.vlines(year_2019 + offsets[region.lower()], ci_low, ci_high,
                   color=ipcc_colors[region], linestyle='-', linewidth=0.8)
        ax4.plot(year_2019 + offsets[region.lower()], final_val, 'o',
                 color=ipcc_colors[region], markersize=1)

if ylim_top_pp != None:
    ax4.set_ylim(top=ylim_top_pp)
if ylim_bottom_pp != None:
    ax4.set_ylim(bottom=ylim_bottom_pp)

ax4.set_xlim(right=end_year+0.6)
ax4.set_title("Soy", pad=10, fontsize=title_fontsize)
ax4.text(0.0, 1.2, "d", transform=ax4.transAxes,fontsize=label_fontsize+1, fontweight='bold', va='top', ha='right')
ax4.axhline(0, color="black", linestyle="dashed", lw=0.9)
ax4.set_xlabel('Year', fontsize=label_fontsize)
ax4.set_ylabel(label_pp, fontsize=label_fontsize)
ax4.tick_params(axis='both', labelsize=tick_fontsize)
ax4.grid(True)

######################## LEGEND
ax4.legend(bbox_to_anchor=(-0.2, -0.55), loc='center', fontsize=tick_fontsize, ncol=4)

########################
fig.subplots_adjust(left=0.08,
                    bottom=0.2,
                    right=0.98,  # 0.83 (with legend), 0.9 (no legend)
                    top=0.92,
                    wspace=0.2,
                    hspace=0.65)

plt.show()
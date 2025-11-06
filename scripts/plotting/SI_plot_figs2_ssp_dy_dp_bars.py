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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from scipy.stats import t
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('../')
import utils

"""
PARAMETERS
"""
crops = ["wheat", "soy", "maize"]
ssps = ["ssp126", "ssp245", "ssp370"]
esms = ["GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL"]
ar6_region = "region_ar6_6_short"
regions = ['DEV', 'APC', 'LAM', 'AFR', 'EEA', 'MEA']
pred_str = "t3s3"  # t=tmax; s=spei; ts=[tmax,spei]

######################## PLOTTING PARAMETERS ########################
vmin_dp_row1 = -4000  # First row minimum (DEV, APC, LAM)
vmin_dp_row2 = -1300  # Second row minimum (AFR, EEA, MEA)
vmax_dp = 0.0

vmin_dy_row1 = -62  # First row minimum
vmin_dy_row2 = -62  # Second row minimum
vmax_dy = 0.0

start_year_avg = 2040
end_year_avg = 2070
year_avg_dy = [x for x in range(start_year_avg, end_year_avg + 1)]
year_avg_dp = [x for x in range(start_year_avg, end_year_avg + 1)]
year_str_avg_dy = [str(x) for x in year_avg_dy]
year_str_avg_dp = [str(x) for x in year_avg_dp]

bar_width = 0.15
label_alphabets = utils.generate_alphabet_list(12, option="lower")
label_alphabets = ["(" + x + ")" for x in label_alphabets]
###################### ###################### ###################### ######################

start_year_hist = 1985  # 1985(isimip3b), 2007(corey)
end_year_hist = 2015  # 2015(isimip3b), 2018(corey)
start_year_fut = 2020
end_year_fut = 2070

root_dir = '../../data'

"""
LOAD DATA
"""
dy_all_crops = []
dp_all_crops = []

for crop in crops:

    dy_file = f"{root_dir}/ssp/linregress_outputs/{crop}/dy_isimip3b_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    dp_file = f"{root_dir}/ssp/linregress_outputs/{crop}/dp_isimip3b_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"

    df_dy = pd.read_csv(dy_file)
    df_dp = pd.read_csv(dp_file)

    # Add crop column
    df_dy['crop'] = crop
    df_dp['crop'] = crop

    dy_all_crops.append(df_dy)
    dp_all_crops.append(df_dp)

# Combine all crops
df_dy = pd.concat(dy_all_crops, ignore_index=True)
df_dp = pd.concat(dp_all_crops, ignore_index=True)

num_cntry_dy = df_dy["country"].nunique()
num_cntry_dp = df_dp["country"].nunique()

print("----------------")
print("NUM countries dy:", num_cntry_dy)
print("NUM countries dp:", num_cntry_dp)

# Filter data
cols_dp = ["country", "crop"] + ["ssp"] + ["esm"] + year_str_avg_dp
cols_dy = ["country", "crop"] + ["ssp"] + ["esm"] + year_str_avg_dy
dy_all = df_dy[cols_dy]
dp_all = df_dp[cols_dp]

# COLS = country,crop,ssp,year,value
dy_long = dy_all.melt(id_vars=['country', 'crop', 'ssp', 'esm'], var_name='year', value_name='value')
dp_long = dp_all.melt(id_vars=['country', 'crop', 'ssp', 'esm'], var_name='year', value_name='value')

# Calculate global means/sums for each ESM
dy_global = dy_long.groupby(['ssp', 'year', 'esm'])['value'].mean().reset_index()
dp_global = dp_long.groupby(['ssp', 'year', 'esm'])['value'].sum().reset_index()

print("----------------")
print("Value ranges after combining crops:")
print(f"dy_global min: {dy_global['value'].min():.2f}, max: {dy_global['value'].max():.2f}")
print(f"dp_global min: {dp_global['value'].min() / 1e6:.2f}, max: {dp_global['value'].max() / 1e6:.2f}")

"""
GET IPCC REGIONS
"""
print("----------------")
print("Getting ipcc region:", ar6_region)
df_ipcc = utils.get_ipcc_region_df()
df_ipcc = df_ipcc[["Country", ar6_region]]
df_ipcc = df_ipcc.rename(columns={"Country": "country"})

# Merge with IPCC regions
dy_ipcc = dy_all.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
dp_ipcc = dp_all.merge(df_ipcc[["country", ar6_region]], on="country", how="left")

# Calculate regional means/sums for timeseries
dy_ipcc_melted = dy_ipcc.melt(id_vars=['country', 'ssp', 'esm', 'crop', ar6_region], var_name='year',
                              value_name='value')
dp_ipcc_melted = dp_ipcc.melt(id_vars=['country', 'ssp', 'esm', 'crop', ar6_region], var_name='year',
                              value_name='value')

dy_avg_years = dy_ipcc_melted[
    (dy_ipcc_melted['year'].astype(int).isin(year_avg_dy)) &
    (dy_ipcc_melted['ssp'].isin(ssps))
    ]
dy_avg_years = dy_avg_years.groupby(['country', 'ssp', 'crop', ar6_region, 'esm'])[
    'value'].mean().reset_index()

dp_avg_years = dp_ipcc_melted[
    (dp_ipcc_melted['year'].astype(int).isin(year_avg_dp)) &
    (dp_ipcc_melted['ssp'].isin(ssps))
    ]
dp_avg_years = dp_avg_years.groupby(['country', 'ssp', 'crop', ar6_region, 'esm'])[
    'value'].sum().reset_index()

dp_avg_years['value'] = dp_avg_years['value'] / 1e6 # convert to Million tonnes

"""
GET STATISTICS
"""
# First aggregate by region for each ESM
def aggregate_by_region(df, metric='dp'):
    # For each ESM prediction:
    # 1. First sum over years for each country
    # 2. Then aggregate countries within each region
    if metric == 'dp':
        # For dp (production loss):
        # Then sum across countries in each region
        regional_sums = df.groupby([ar6_region, 'ssp', 'crop', 'esm'])['value'].sum().reset_index()
    else:  # dy (yield loss)
        # Then take mean across countries in each region
        regional_sums = df.groupby([ar6_region, 'ssp', 'crop', 'esm'])['value'].mean().reset_index()
    return regional_sums


# Calculate statistics across ESMs
def calculate_stats(df):
    # For each region-ssp-crop combination:
    # Show median of ESM predictions and their spread
    stats = df.groupby([ar6_region, 'ssp', 'crop']).agg(
        median=('value', 'median'),  # Median across ESM predictions
        q25=('value', lambda x: x.quantile(0.25)),  # 25th percentile of ESM predictions
        q75=('value', lambda x: x.quantile(0.75)),  # 75th percentile of ESM predictions
        min=('value', 'min'),  # Min ESM prediction
        max=('value', 'max')  # Max ESM prediction
    ).reset_index()
    return stats

# Calculate 'all' crops statistics
def calculate_all_crops_stats(df, metric='dp'):
    if metric == 'dp':
        # Sum across crops for each region-ssp-esm combination
        all_crops = df.groupby([ar6_region, 'ssp', 'esm'])['value'].mean().reset_index()
    else:  # degdp
        # Mean across crops for each region-ssp-esm combination
        all_crops = df.groupby([ar6_region, 'ssp', 'esm'])['value'].mean().reset_index()

    all_crops['crop'] = 'all'
    return all_crops

# First aggregate by region for each crop and ESM
dp_regional = aggregate_by_region(dp_avg_years, 'dp')
dy_regional = aggregate_by_region(dy_avg_years, 'dy')

# Add 'all' crops category
dp_all_crops = calculate_all_crops_stats(dp_regional, 'dp')
dp_regional = pd.concat([dp_regional, dp_all_crops])

dy_all_crops = calculate_all_crops_stats(dy_regional, 'dy')
dy_regional = pd.concat([dy_regional, dy_all_crops])

# Calculate statistics across ESMs
dp_stats = calculate_stats(dp_regional)
dy_stats = calculate_stats(dy_regional)
crops = ['maize','wheat','soy','all']

"""
PLOT
"""
fig, axes = plt.subplots(4, 3, figsize=(10, 8))
title_fontsize = 14
label_fontsize = 12
tick_fontsize = 12
legend_fontsize = 12

# Set color scheme for crops
crop_colors = plt.cm.Set2(np.linspace(0, 1, len(crops)))
crop_color_dict = dict(zip(crops, crop_colors))

group_width = bar_width * (len(crops) + 1)  # +1 for spacing between groups

def plot_data(ax, stats, ssps, crops, crop_color_dict, bar_width, group_width, vmin, vmax, is_dy=False):
    # Plot groups
    for ssp_idx, ssp in enumerate(ssps):
        ssp_stats = stats[stats['ssp'] == ssp]
        x_center = ssp_idx * (group_width * 1.5)  # Space between SSP groups

        # Plot each crop
        for crop_idx, crop in enumerate(crops):
            crop_stats = ssp_stats[ssp_stats['crop'] == crop]
            if len(crop_stats) == 0:
                print(f"No data for {ssp}, {crop}")
                continue

            x = x_center + crop_idx * bar_width

            # Get statistics
            median = crop_stats['median'].iloc[0]
            q25 = crop_stats['q25'].iloc[0]
            q75 = crop_stats['q75'].iloc[0]
            min_val = crop_stats['min'].iloc[0]
            max_val = crop_stats['max'].iloc[0]

            # Plot median bar
            ax.bar(x, median, bar_width, color=crop_color_dict[crop], alpha=0.7)

            # Plot error bars (25th-75th percentile)
            ax.vlines(x, q25, q75, color='gray', linewidth=1.0)

            # Plot min/max points
            ax.plot([x], [min_val], '.', color='gray', markersize=2)
            ax.plot([x], [max_val], '.', color='gray', markersize=2)

    # Set x-axis labels
    ax.set_xticks([i * (group_width * 1.5) for i in range(len(ssps))])
    ax.set_xticklabels(['SSP126', 'SSP245', 'SSP370'], fontsize=tick_fontsize)

    # Set y-axis limits and format
    ax.set_ylim(vmin, vmax)

    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid(True, linestyle='--', alpha=0.7)

# Plot each region
n = 0
for region_idx, region in enumerate(regions):
    # Calculate row and column position
    row_offset = region_idx // 3  # 0 for first 3 regions, 1 for last 3 regions
    col = region_idx % 3

    # Get data for this region
    dp_region = dp_stats[dp_stats[ar6_region] == region]
    dy_region = dy_stats[dy_stats[ar6_region] == region]

    lab = f"{label_alphabets[n]} {region}"
    # Plot economic damage (de)
    ax = axes[row_offset, col]
    vmin_dp = vmin_dp_row1 if row_offset == 0 else vmin_dp_row2
    plot_data(ax, dp_region, ssps, crops, crop_color_dict, bar_width, group_width, vmin_dp, vmax_dp)
    ax.set_title(lab, pad=10, fontsize=label_fontsize, weight='bold')

    if col == 0:
        ax.set_ylabel(r'$dp$ [Mt]', fontsize=label_fontsize)
    else:
        ax.set_yticklabels([])

    # Plot dy
    lab = f"{label_alphabets[n + 6]} {region}"
    ax = axes[row_offset + 2, col]
    vmin_dy = vmin_dy_row1 if row_offset == 0 else vmin_dy_row2
    plot_data(ax, dy_region, ssps, crops, crop_color_dict, bar_width, group_width, vmin_dy, vmax_dy, is_dy=True)
    ax.set_title(lab, pad=10, fontsize=label_fontsize, weight="bold")

    print(f"{region} de:")
    print(dp_region)
    print("------------")
    print(f"{region} degdp:")
    print(dy_region)

    if col == 0:
        ax.set_ylabel(r'$\overline{dy}$ [%]', fontsize=label_fontsize)
    else:
        ax.set_yticklabels([])

    n += 1

# Add legend outside the plot
crops_legend = [crop.capitalize() for crop in crops]
legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=crop_color_dict[crop])
                   for crop in crops]
fig.legend(legend_elements, crops_legend, loc='center right',
           bbox_to_anchor=(1.0, 0.5), fontsize=legend_fontsize)

# -----------------------------------------
fig.subplots_adjust(left=0.09,
                    bottom=0.05,
                    right=0.88,
                    top=0.95,
                    wspace=0.08,
                    hspace=0.65)

"""
wspace and hspace specify the space reserved between Matplotlib subplots. They are the fractions of axis width and height, respectively.
left, right, top and bottom parameters specify four sides of the subplots’ positions. They are the fractions of the width and height of the figure.
top and bottom should add up to 1.0
"""

plt.show()

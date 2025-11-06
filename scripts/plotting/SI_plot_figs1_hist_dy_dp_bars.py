# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 300
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path
from scipy.stats import t
sns.set_style('whitegrid')
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('../')
import utils

"""
PARAMETERS
"""
crops = ["maize", "wheat", "soy"]
ar6_region = "region_ar6_6_short"
regions = ['DEV', 'APC', 'LAM', 'AFR', 'EEA', 'MEA']
pred_str = "t3s3"  # t=tmax; s=spei; ts=[tmax,spei]

######################## PLOTTING PARAMETERS ########################
bar_width = 0.15

start_year_avg = 2000  # 2010,2000
end_year_avg = 2019
year_avg_dy = [x for x in range(start_year_avg, end_year_avg + 1)]
year_avg_dp = [x for x in range(start_year_avg, end_year_avg + 1)]
year_str_avg_dy = [str(x) for x in year_avg_dy]
year_str_avg_dp = [str(x) for x in year_avg_dp]

label_alphabets = utils.generate_alphabet_list(6, option="lower")
label_alphabets = ["(" + x + ")" for x in label_alphabets]
###################### ###################### ###################### ######################

# For filename
start_year_hist = 1971
end_year_hist = 1989
start_year_fut = 1990
end_year_fut = 2019

root_dir = '../../data'

"""
LOAD DATA
"""
dy_all_crops = []
dp_all_crops = []

for crop in crops:

    dy_file = f"{root_dir}/historical/linregress_outputs/{crop}/isimip3a/dy_isimip3a_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    dp_file = f"{root_dir}/historical/linregress_outputs/{crop}/isimip3a/dp_isimip3a_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"

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

# Get valid countries
hist_file = f"{root_dir}/historical/linregress_inputs/INPUT_HISTORICAL_DETRENDED_DEMEANED.csv"
df_hist = pd.read_csv(hist_file)
valid_countries = df_hist["country"].unique()

# Filter for valid countries
df_dy = df_dy[df_dy['country'].isin(valid_countries)]
df_dp = df_dp[df_dp['country'].isin(valid_countries)]

# Remove unreasonable data
df_dy = df_dy[df_dy['country'] != 'BGD']
df_dp = df_dp[df_dp['country'] != 'BGD']

num_cntry_dy = df_dy["country"].nunique()
num_cntry_dp = df_dp["country"].nunique()

print("----------------")
print("NUM countries dy:", num_cntry_dy)
print("NUM countries dp:", num_cntry_dp)

# Filter data
cols_dp = ["country", "crop"] + year_str_avg_dp
cols_dy = ["country", "crop"] + year_str_avg_dy
dy_all = df_dy[cols_dy]
dp_all = df_dp[cols_dp]

# COLS = country,crop,ssp,year,value
dy_long = dy_all.melt(id_vars=['country', 'crop'], var_name='year', value_name='value')
dp_long = dp_all.melt(id_vars=['country', 'crop'], var_name='year', value_name='value')

dp_long["value"] = dp_long["value"] / 1e6

# Calculate global means/sums
dy_global = dy_long.groupby(['crop'])['value'].mean().reset_index()
dp_global = dp_long.groupby(['crop'])['value'].sum().reset_index()

print("----------------")
print("Value ranges after combining crops:")
print(f"dy_global min: {dy_global['value'].min():.2f}, max: {dy_global['value'].max():.2f}")
print(f"dp_global min: {dp_global['value'].min():.2f}, max: {dp_global['value'].max():.2f}")

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

dy_ipcc_melted = dy_ipcc.melt(id_vars=['country', 'crop', ar6_region], var_name='year', value_name='value')
dp_ipcc_melted = dp_ipcc.melt(id_vars=['country', 'crop', ar6_region], var_name='year', value_name='value')

# For dy: take mean over years for each country
dy_avg_years = dy_ipcc_melted.groupby(['country', 'crop', ar6_region], as_index=False)['value'].mean()

# For dp:
# First sum over years for each country
dp_country_sum = dp_ipcc_melted.groupby(['country', 'crop', ar6_region], as_index=False)['value'].sum()

# Convert to Million tonnes
dp_country_sum['value'] = dp_country_sum['value'] / 1e6

"""
GET STATISTICS
"""
# For each region and crop, calculate:
# - Total sum across countries
# - 95% confidence interval using bootstrap

def bootstrap_ci(x, n_bootstrap=1000):
    bootstrap_sums = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = np.random.choice(x, size=len(x), replace=True)
        bootstrap_sums.append(np.sum(sample))
    # Get 95% CI
    ci_lower = np.percentile(bootstrap_sums, 2.5)
    ci_upper = np.percentile(bootstrap_sums, 97.5)
    return ci_lower, ci_upper


dp_stats_temp = dp_country_sum.groupby([ar6_region, 'crop']).agg(
    median=('value', 'sum'),  # Total regional sum
    q25=('value', lambda x: bootstrap_ci(x)[0]),  # Lower 95% CI
    q75=('value', lambda x: bootstrap_ci(x)[1]),  # Upper 95% CI
    min=('value', 'min'),
    max=('value', 'max')
).reset_index()

# Use this for dp plotting
dp_avg_years = dp_stats_temp

def calculate_stats(df):
    # For dy: calculate mean and 95% CI across countries in each region
    if df is dy_avg_years:  # Check if this is dy data
        stats = df.groupby([ar6_region, 'crop']).agg(
            median=('value', 'mean'),  # Using mean instead of median
            q25=('value', lambda x: x.mean() - 1.96 * x.std() / np.sqrt(len(x))),  # Lower 95% CI
            q75=('value', lambda x: x.mean() + 1.96 * x.std() / np.sqrt(len(x))),  # Upper 95% CI
            min=('value', 'min'),
            max=('value', 'max')
        ).reset_index()
    else:
        # For dp: we already have the stats calculated
        stats = df.copy()
    return stats

# Calculate statistics across ESMs
dp_stats = calculate_stats(dp_avg_years)
dy_stats = calculate_stats(dy_avg_years)

dp_stats.rename(columns={ar6_region: 'region'}, inplace=True)
dy_stats.rename(columns={ar6_region: 'region'}, inplace=True)

# Rename region column
dp_stats = dp_stats.rename(columns={'region_ar6_6_short': 'region'})
dy_stats = dy_stats.rename(columns={'region_ar6_6_short': 'region'})

print ("------------------")
print("dp_stats:")
print (dp_stats)
print ("------------------")
print("dy_stats:")
print (dy_stats)

"""
PLOT
"""
fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
title_fontsize = 14
label_fontsize = 12
tick_fontsize = 12
legend_fontsize = 12

# Define region colors
region_colors = plt.cm.Set2(np.linspace(0, 1, len(regions)))
region_color_dict = dict(zip(regions, region_colors))

group_width = bar_width * (len(regions) + 1)  # +1 for spacing between groups

def plot_data(ax, stats, crop, title, regions, region_color_dict, bar_width, is_dp=True):
    # Get data for this crop
    crop_stats = stats[stats['crop'] == crop]

    for idx, region in enumerate(regions):
        region_stats = crop_stats[crop_stats['region'] == region]
        if len(region_stats) == 0:
            print(f"No data for {crop}, {region}")
            continue

        x = idx * bar_width * 1.2  # Space between bars

        # Get statistics
        median = region_stats['median'].iloc[0]
        q25 = region_stats['q25'].iloc[0]
        q75 = region_stats['q75'].iloc[0]
        min_val = region_stats['min'].iloc[0]
        max_val = region_stats['max'].iloc[0]

        # Plot median bar
        ax.bar(x, median, bar_width, color=region_color_dict[region], alpha=0.7, label=region if idx == 0 else "")

        lower_err = abs(q25 - median)
        upper_err = abs(q75 - median)
        ax.errorbar(x, median, yerr=[[lower_err], [upper_err]], fmt='none', color='gray', capsize=1,
                    elinewidth=0.8)

    # Set x-axis
    x_positions = np.arange(len(regions)) * bar_width * 1.2
    ax.set_xticks(x_positions)
    ax.set_xticklabels(regions, rotation=90, ha='right')
    ax.set_title(title, pad=10, fontsize=title_fontsize, weight='bold')

    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid(True, linestyle='--', alpha=0.7)

# Plot each crop
n = 0
for col, crop in enumerate(['maize', 'wheat', 'soy']):

    title = f"{label_alphabets[n]} {crop.capitalize()}"
    # Plot dp (top row)
    ax = axes[0, col]
    plot_data(ax, dp_stats, crop, title, regions, region_color_dict, bar_width, is_dp=True)
    if col == 0:
        ax.set_ylabel(r'$dp$ [Mt]', fontsize=label_fontsize)

    # Plot dy (bottom row)
    title = f"{label_alphabets[n + 3]} {crop.capitalize()}"
    ax = axes[1, col]
    plot_data(ax, dy_stats, crop, title, regions, region_color_dict, bar_width, is_dp=False)
    if col == 0:
        ax.set_ylabel(r'$\overline{dy}$ [%]', fontsize=label_fontsize)

    n += 1

# -----------------------------------------
fig.subplots_adjust(left=0.08,
                    bottom=0.12,
                    right=0.98,
                    top=0.92,
                    wspace=0.19,
                    hspace=0.55)

"""
wspace and hspace specify the space reserved between Matplotlib subplots. They are the fractions of axis width and height, respectively.
left, right, top and bottom parameters specify four sides of the subplots’ positions. They are the fractions of the width and height of the figure.
top and bottom should add up to 1.0
"""

plt.show()

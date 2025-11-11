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
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('../')
import utils

"""
PARAMETERS
"""
crops = ["maize", "wheat", "soy"]
ar6_region = "region_ar6_6_short"  # region_ar6_6_short, region_ar6_dev
regions = ['DEV', 'APC', 'LAM', 'AFR', 'EEA', 'MEA']
pred_str = "t3s3"  # t=tmax; s=spei; ts=[tmax,spei]

######################## PLOTTING PARAMETERS ########################
bar_width = 0.15
label_de = "Economic loss [B US$]"
label_degdp = "GDP loss [%]"

label_alphabets = utils.generate_alphabet_list(6, option="lower")
label_alphabets = [x for x in label_alphabets]
#####################################################################

start_year_avg = 2000
end_year_avg = 2019
year_avg_degdp = [x for x in range(start_year_avg, end_year_avg + 1)]
year_avg_de = [x for x in range(start_year_avg, end_year_avg + 1)]
year_str_avg_degdp = [str(x) for x in year_avg_degdp]
year_str_avg_de = [str(x) for x in year_avg_de]

# For input filename
start_year_hist = 1971
end_year_hist = 1989
start_year_fut = 1990
end_year_fut = 2019

root_dir = '../../data'

"""
LOAD DATA
"""
degdp_all_crops = []
de_all_crops = []

for crop in crops:

    degdp_file = f"{root_dir}/historical/linregress_outputs/{crop}/isimip3a/degdp_isimip3a_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    de_file = f"{root_dir}/historical/linregress_outputs/{crop}/isimip3a/de_isimip3a_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"

    df_degdp = pd.read_csv(degdp_file)
    df_de = pd.read_csv(de_file)

    # Add crop column
    df_degdp['crop'] = crop
    df_de['crop'] = crop

    degdp_all_crops.append(df_degdp)
    de_all_crops.append(df_de)

# Combine all crops
df_degdp = pd.concat(degdp_all_crops, ignore_index=True)
df_de = pd.concat(de_all_crops, ignore_index=True)

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

num_cntry_degdp = df_degdp["country"].nunique()
num_cntry_de = df_de["country"].nunique()

print("----------------")
print("NUM countries degdp:", num_cntry_degdp)
print("NUM countries de:", num_cntry_de)

# Filter data
cols_de = ["country", "crop"] + year_str_avg_de
cols_degdp = ["country", "crop"] + year_str_avg_degdp
degdp_all = df_degdp[cols_degdp]
de_all = df_de[cols_de]

# COLS = country,crop,ssp,year,value
degdp_long = degdp_all.melt(id_vars=['country', 'crop'], var_name='year', value_name='value')
de_long = de_all.melt(id_vars=['country', 'crop'], var_name='year', value_name='value')

de_long["value"] = de_long["value"] / 1e9

# Calculate global means/sums
degdp_global = degdp_long.groupby(['crop'])['value'].mean().reset_index()
de_global = de_long.groupby(['crop'])['value'].sum().reset_index()

print("----------------")
print("Value ranges after combining crops:")
print(f"degdp_global min: {degdp_global['value'].min():.2f}, max: {degdp_global['value'].max():.2f}")
print(f"de_global min: {de_global['value'].min():.2f}, max: {de_global['value'].max():.2f}")

"""
GET IPCC REGIONS
"""
print("----------------")
print("Getting ipcc region:", ar6_region)
df_ipcc = utils.get_ipcc_region_df()
df_ipcc = df_ipcc[["Country", ar6_region]]
df_ipcc = df_ipcc.rename(columns={"Country": "country"})

# Merge with IPCC regions
degdp_ipcc = degdp_all.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
de_ipcc = de_all.merge(df_ipcc[["country", ar6_region]], on="country", how="left")

degdp_ipcc_melted = degdp_ipcc.melt(id_vars=['country', 'crop', ar6_region], var_name='year', value_name='value')
de_ipcc_melted = de_ipcc.melt(id_vars=['country', 'crop', ar6_region], var_name='year', value_name='value')

# For degdp: take mean over years for each country
degdp_avg_years = degdp_ipcc_melted.groupby(['country', 'crop', ar6_region], as_index=False)['value'].mean()

# For de:
# First sum over years for each country
de_country_sum = de_ipcc_melted.groupby(['country', 'crop', ar6_region], as_index=False)['value'].sum()

# Convert to Billion US$
de_country_sum['value'] = de_country_sum['value'] / 1e9

"""
GET STATISTICS
"""
# For each region and crop, calculate:
# - Total sum across countries
# - 95% confidence interval using bootstrap

from scipy import stats
import numpy as np

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


de_stats_temp = de_country_sum.groupby([ar6_region, 'crop']).agg(
    median=('value', 'sum'),  # Total regional sum
    q25=('value', lambda x: bootstrap_ci(x)[0]),  # Lower 95% CI
    q75=('value', lambda x: bootstrap_ci(x)[1]),  # Upper 95% CI
    min=('value', 'min'),
    max=('value', 'max')
).reset_index()

de_avg_years = de_stats_temp

def calculate_stats(df):
    # For dy: calculate mean and 95% CI across countries in each region
    if df is degdp_avg_years:  # Check if this is dy data
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
de_stats = calculate_stats(de_avg_years)
degdp_stats = calculate_stats(degdp_avg_years)

de_stats.rename(columns={ar6_region: 'region'}, inplace=True)
degdp_stats.rename(columns={ar6_region: 'region'}, inplace=True)

# Rename region column
de_stats = de_stats.rename(columns={'region_ar6_6_short': 'region'})
degdp_stats = degdp_stats.rename(columns={'region_ar6_6_short': 'region'})

print ("------------------")
print("de_stats:")
print (de_stats)
print ("------------------")
print("degdp_stats:")
print (degdp_stats)

"""
PLOT
"""
fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
title_fontsize = 14
label_fontsize = 12
tick_fontsize = 12
legend_fontsize = 12

# Define regions
region_colors = plt.cm.Set2(np.linspace(0, 1, len(regions)))
region_color_dict = dict(zip(regions, region_colors))

group_width = bar_width * (len(regions) + 1)  # +1 for spacing between groups

def plot_data(ax, stats, crop, title, label, regions, region_color_dict, bar_width, is_de=True):
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
    ax.set_title(title, pad=10, fontsize=title_fontsize)
    ax.text(-0.12, 1.1, label, transform=ax.transAxes, fontweight='bold', fontsize=label_fontsize + 1)

    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid(True, linestyle='--', alpha=0.7)


# Plot each crop
n = 0
for col, crop in enumerate(['maize', 'wheat', 'soy']):

    title = f"{crop.capitalize()}"
    label = label_alphabets[n]

    # Plot dp (top row)
    ax = axes[0, col]
    plot_data(ax, de_stats, crop, title, label, regions, region_color_dict, bar_width, is_de=True)
    if col == 0:
        ax.set_ylabel(label_de, fontsize=label_fontsize)

    # Plot dy (bottom row)
    title = f"{crop.capitalize()}"
    label = label_alphabets[n+3]
    ax = axes[1, col]
    plot_data(ax, degdp_stats, crop, title, label, regions, region_color_dict, bar_width, is_de=False)
    if col == 0:
        ax.set_ylabel(label_degdp, fontsize=label_fontsize)

    n += 1

# -----------------------------------------
fig.subplots_adjust(left=0.09,
                    bottom=0.12,  # Increased to accommodate rotated x-labels
                    right=0.99,
                    top=0.92,
                    wspace=0.19,
                    hspace=0.55)

"""
wspace and hspace specify the space reserved between Matplotlib subplots. They are the fractions of axis width and height, respectively.
left, right, top and bottom parameters specify four sides of the subplots’ positions. They are the fractions of the width and height of the figure.
top and bottom should add up to 1.0
"""

plt.show()

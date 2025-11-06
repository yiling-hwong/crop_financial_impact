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
import cartopy.crs as ccrs
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
pred_str = "t3s3"

root_dir = '../../data'

######################## PLOTTING PARAMETERS ########################
vmin_de_row1 = -1300  # First row minimum (DEV, APC, LAM)
vmin_de_row2 = -250  # Second row minimum (AFR, EEA, MEA)
vmax_de = 0.0

vmin_degdp_row1 = -0.1
vmin_degdp_row2 = -0.1
vmax_degdp = 0.0

start_year_avg = 2040
end_year_avg = 2070
year_avg_degdp = [2040, 2045, 2050, 2055, 2060, 2065, 2070]
year_avg_de = [x for x in range(start_year_avg, end_year_avg + 1)]
year_str_avg_degdp = [str(x) for x in year_avg_degdp]
year_str_avg_de = [str(x) for x in year_avg_de]

label_alphabets = utils.generate_alphabet_list(12, option="lower")
label_alphabets = ["(" + x + ")" for x in label_alphabets]
##################################################################

start_year_hist = 1985
end_year_hist = 2015
start_year_fut = 2020
end_year_fut = 2070

"""
GET de and degdp DATA
"""
degdp_all_crops = []
de_all_crops = []
for crop in crops:
    degdp_file = f"{root_dir}/ssp/linregress_outputs/{crop}/degdp_isimip3b_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    de_file = f"{root_dir}/ssp/linregress_outputs/{crop}/de_isimip3b_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"

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

num_cntry_degdp = df_degdp["country"].nunique()
num_cntry_de = df_de["country"].nunique()

print("----------------")
print("NUM countries degdp:", num_cntry_degdp)
print("NUM countries de:", num_cntry_de)
print("----------------")

# Filter data
cols_de = ["country", "crop"] + ["ssp"] + ["esm"] + year_str_avg_de
cols_degdp = ["country", "crop"] + ["ssp"] + ["esm"] + year_str_avg_degdp
degdp_all = df_degdp[cols_degdp]
de_all = df_de[cols_de]

# COLS = country,crop,ssp,year,value
degdp_long = degdp_all.melt(id_vars=['country', 'crop', 'ssp', 'esm'], var_name='year', value_name='value')
de_long = de_all.melt(id_vars=['country', 'crop', 'ssp', 'esm'], var_name='year', value_name='value')

# Calculate global means/sums for each ESM
degdp_global = degdp_long.groupby(['ssp', 'year', 'esm'])['value'].mean().reset_index()
de_global = de_long.groupby(['ssp', 'year', 'esm'])['value'].sum().reset_index()

print("Value ranges after combining crops:")
print(f"degdp_global min: {degdp_global['value'].min():.2f}, max: {degdp_global['value'].max():.2f}")
print(f"de_global min: {de_global['value'].min() / 1e9:.2f}, max: {de_global['value'].max() / 1e9:.2f}")

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

# Calculate regional means/sums for timeseries
degdp_ipcc_melted = degdp_ipcc.melt(id_vars=['country', 'ssp', 'esm', 'crop', ar6_region], var_name='year',
                                    value_name='value')
de_ipcc_melted = de_ipcc.melt(id_vars=['country', 'ssp', 'esm', 'crop', ar6_region], var_name='year',
                              value_name='value')

degdp_avg_years = degdp_ipcc_melted[
    (degdp_ipcc_melted['year'].astype(int).isin(year_avg_degdp)) &
    (degdp_ipcc_melted['ssp'].isin(ssps))
    ]
degdp_avg_years = degdp_avg_years.groupby(['country', 'ssp', 'crop', ar6_region, 'esm'])['value'].mean().reset_index()

de_avg_years = de_ipcc_melted[
    (de_ipcc_melted['year'].astype(int).isin(year_avg_de)) &
    (de_ipcc_melted['ssp'].isin(ssps))
    ]
de_avg_years = de_avg_years.groupby(['country', 'ssp', 'crop', ar6_region, 'esm'])['value'].sum().reset_index()

"""
GET STATISTICS
"""
# First aggregate by region for each ESM
def aggregate_by_region(df, metric='de'):
    # For each ESM prediction:
    # 1. First sum over years for each country
    # 2. Then aggregate countries within each region
    if metric == 'de':
        # For de (economic damage):
        # Then sum across countries in each region
        regional_sums = df.groupby([ar6_region, 'ssp', 'crop', 'esm'])['value'].sum().reset_index()
        regional_sums['value'] = regional_sums['value'] / 1e9  # Convert to billions
    else:  # degdp (GDP damage)
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
def calculate_all_crops_stats(df, metric='de'):
    if metric == 'de':
        # Sum across crops for each region-ssp-esm combination
        all_crops = df.groupby([ar6_region, 'ssp', 'esm'])['value'].mean().reset_index()
    else:  # degdp
        # Mean across crops for each region-ssp-esm combination
        all_crops = df.groupby([ar6_region, 'ssp', 'esm'])['value'].mean().reset_index()

    all_crops['crop'] = 'all'
    return all_crops

# First aggregate by region for each crop and ESM
de_regional = aggregate_by_region(de_avg_years, 'de')
degdp_regional = aggregate_by_region(degdp_avg_years, 'degdp')

# Add 'all' crops category
de_all_crops = calculate_all_crops_stats(de_regional, 'de')
de_regional = pd.concat([de_regional, de_all_crops])

degdp_all_crops = calculate_all_crops_stats(degdp_regional, 'degdp')
degdp_regional = pd.concat([degdp_regional, degdp_all_crops])

# Calculate statistics across ESMs
de_stats = calculate_stats(de_regional)
degdp_stats = calculate_stats(degdp_regional)
crops = ['maize', 'wheat', 'soy', 'all']

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

# Plot parameters
bar_width = 0.15
group_width = bar_width * (len(crops) + 1)  # +1 for spacing between groups

def plot_data(ax, stats, ssps, crops, crop_color_dict, bar_width, group_width, vmin, vmax, is_gdp=False):
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
    if is_gdp:
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

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
    de_region = de_stats[de_stats[ar6_region] == region]
    degdp_region = degdp_stats[degdp_stats[ar6_region] == region]

    lab = f"{label_alphabets[n]} {region}"
    # Plot economic damage (de)
    ax = axes[row_offset, col]
    vmin_de = vmin_de_row1 if row_offset == 0 else vmin_de_row2
    plot_data(ax, de_region, ssps, crops, crop_color_dict, bar_width, group_width, vmin_de, vmax_de)
    ax.set_title(lab, pad=10, fontsize=label_fontsize, weight='bold')

    if col == 0:
        ax.set_ylabel(r'$de$ [B US\$]', fontsize=label_fontsize)
    else:
        ax.set_yticklabels([])

    # Plot GDP damage (degdp)
    lab = f"{label_alphabets[n + 6]} {region}"
    ax = axes[row_offset + 2, col]
    vmin_degdp = vmin_degdp_row1 if row_offset == 0 else vmin_degdp_row2
    plot_data(ax, degdp_region, ssps, crops, crop_color_dict, bar_width, group_width, vmin_degdp, vmax_degdp,
              is_gdp=True)
    ax.set_title(lab, pad=10, fontsize=label_fontsize, weight="bold")

    print(f"{region} de:")
    print(de_region)
    print("------------")
    print(f"{region} degdp:")
    print(degdp_region)

    if col == 0:
        ax.set_ylabel(r'$\overline{degdp}$ [%GDP]', fontsize=label_fontsize)
    else:
        ax.set_yticklabels([])

    n += 1

# Add legend outside the plot
crops_legend = [crop.capitalize() for crop in crops]
legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=crop_color_dict[crop])
                   for crop in crops]
fig.legend(legend_elements, crops_legend, loc='center right',
           bbox_to_anchor=(1.0, 0.5), fontsize=legend_fontsize)

"""
GET outlying LAM values
"""
degdp_lam_soy = degdp_stats[(degdp_stats[ar6_region] == "LAM") & (degdp_stats['crop'] == "soy")]
print("-----------")
print("STATS for outlying point in LAM soy degdp:")
print(degdp_lam_soy)

# -----------------------------------------
fig.subplots_adjust(left=0.09,
                    bottom=0.05,
                    right=0.88,
                    top=0.95,
                    wspace=0.08,  # 0.1
                    hspace=0.65)  # 0.85

"""
wspace and hspace specify the space reserved between Matplotlib subplots. They are the fractions of axis width and height, respectively.
left, right, top and bottom parameters specify four sides of the subplots’ positions. They are the fractions of the width and height of the figure.
top and bottom should add up to 1.0
"""

plt.show()

#########################################
# PRINT OUT DATA FOR ANALYSIS
#########################################
ssp_target = "ssp370"
country = "ALL"  # ALL, USA
start_year = 2040
end_year = 2070

year_avg_degdp = list(range(start_year, end_year + 1, 5))
year_avg_de = [x for x in range(start_year, end_year + 1)]
year_str_avg_degdp = [str(x) for x in year_avg_degdp]
year_str_avg_de = [str(x) for x in year_avg_de]

degdp_years = degdp_ipcc_melted[
    (degdp_ipcc_melted['year'].astype(int).isin(year_avg_degdp)) &
    (degdp_ipcc_melted['ssp'].isin(ssps))
    ]

de_years = de_ipcc_melted[
    (de_ipcc_melted['year'].astype(int).isin(year_avg_de)) &
    (de_ipcc_melted['ssp'].isin(ssps))
    ]

degdp_years = degdp_years.groupby(['country', 'ssp', 'crop', ar6_region, 'esm'])['value'].mean().reset_index()
de_years = de_years.groupby(['country', 'ssp', 'crop', ar6_region, 'esm'])['value'].sum().reset_index()

# sum/avg across crop
degdp_years = degdp_years.groupby(['country', 'ssp', ar6_region, 'esm'])['value'].mean().reset_index()
de_years = de_years.groupby(['country', 'ssp', ar6_region, 'esm'])['value'].sum().reset_index()

# average across esm
degdp_years = degdp_years.groupby(['country', 'ssp', ar6_region])['value'].mean().reset_index()
de_years = de_years.groupby(['country', 'ssp', ar6_region])['value'].mean().reset_index()

# select ssp
degdp_years = degdp_years[degdp_years["ssp"] == ssp_target]
de_years = de_years[de_years["ssp"] == ssp_target]

if country == "ALL":
    degdp_mean = degdp_years["value"].mean()
    de_sum = de_years["value"].sum()

else:
    degdp_mean = degdp_years[degdp_years["country"] == country]["value"].mean()
    de_sum = de_years[de_years["country"] == country]["value"].sum()

de_sum = de_sum / 1e9

print("--------------------")
print(f"TOTAL DE for {country} for {ssp_target} over {start_year}-{end_year} (B US$): {de_sum}")
print(f"MEAN DEGDP for {country} for {ssp_target} over {start_year}-{end_year} (B US$): {degdp_mean}")
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
crops = ["wheat", "soy", "maize"]
ssps = ["ssp126", "ssp245", "ssp370"]
ssp_diffs = [("ssp245", "ssp126"), ("ssp370", "ssp126")]
esms = ["GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0", "UKESM1-0-LL"]
pred_str = "t3s3"

######################## PLOTTING PARAMETERS ########################
# Maps parameters
cmap_dy = "Reds_r"
cmap_dp = "copper"
vmax_dy = 0.0
vmin_dy = -12
vmax_dp = -1
vmin_dp = -1000
ticks_dp = [0, -0.005, -1, -20]
start_year_avg = 2060
end_year_avg = 2070
year_str_avg = [str(x) for x in range(start_year_avg,end_year_avg+1)]
label_dy = "Yield difference [%]"
label_dp = "Production difference [Mt]"

# Timeseries parameters
ylim_top_dy = 4
ylim_bottom_dy = -16
ylim_top_dp = 60
ylim_bottom_dp = -270
start_year_plot = 2030
end_year_plot = 2070

window_size = 4
smooth_method = "pandas"
np_mode = "full"

ipcc_colors = {"ldc":"crimson","developing":"goldenrod","developed":"dodgerblue"} #ldc,developing,developed
ipcc_labels = {"ldc": "LDC", "developing": "Developing", "developed": "Developed"}

label_alphabets = utils.generate_alphabet_list(12,option="lower")
label_alphabets = [x for x in label_alphabets]
##################################################################

ar6_region = "region_ar6_dev"
regions = ["ldc","developing","developed"]

# Time ranges
start_year = 2020
end_year = 2070

years = [x for x in range(start_year, end_year+1)]
year_str_all = [str(x) for x in range(start_year, end_year+1)]
years_plot = [x for x in range(start_year_plot, end_year_plot+1)]
years_plot = np.array(years_plot)

plot_start_idx = years.index(start_year_plot)
plot_end_idx = years.index(end_year_plot) + 1

# For input file names
start_year_hist = 1985
end_year_hist = 2015
start_year_fut = 2020
end_year_fut = 2070

root_dir = '../../data'
country_shape_file = f"{root_dir}/resources/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

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

# Filter data
cols = ["country", "crop"] + ["ssp"] + ["esm"] + year_str_all
dy_all = df_dy[cols]
dp_all = df_dp[cols]

dy_long = dy_all.melt(id_vars=['country', 'crop', 'ssp', 'esm'], var_name='year', value_name='value')
dp_long = dp_all.melt(id_vars=['country', 'crop', 'ssp', 'esm'], var_name='year', value_name='value')

# Get mean over ESMs
dy_long = dy_long.groupby(['country', 'crop', 'ssp', 'year'])['value'].mean().reset_index()
dp_long = dp_long.groupby(['country', 'crop', 'ssp', 'year'])['value'].mean().reset_index()

# Get mean/sum over crops for
dy_long = dy_long.groupby(['country', 'ssp', 'year'])['value'].mean().reset_index()
dp_long = dp_long.groupby(['country', 'ssp', 'year'])['value'].sum().reset_index()

# Calculate global means/sums
dy_global = dy_long.groupby(['ssp', 'year'])['value'].mean().reset_index()
dp_global = dp_long.groupby(['ssp', 'year'])['value'].sum().reset_index()

print ()
print("\nValue ranges after combining crops:")
print(f"dy_global min: {dy_global['value'].min():.2f}, max: {dy_global['value'].max():.2f}")
print(f"dp_global min: {dp_global['value'].min() / 1e6:.2f}, max: {dp_global['value'].max() / 1e6:.2f}")

"""
GET IPCC REGION DATA
"""
print ("--------------------")
print("Getting IPCC region:", ar6_region)
df_ipcc = utils.get_ipcc_region_df()
df_ipcc = df_ipcc[["Country", ar6_region]]
df_ipcc = df_ipcc.rename(columns={"Country": "country"})

# Merge with IPCC regions
dy_ipcc = dy_all.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
dp_ipcc = dp_all.merge(df_ipcc[["country", ar6_region]], on="country", how="left")

# Calculate regional means/sums for timeseries
dy_ipcc_melted = dy_ipcc.melt(id_vars=['country', 'ssp','esm','crop','region_ar6_dev'], var_name='year', value_name='value')
dp_ipcc_melted = dp_ipcc.melt(id_vars=['country', 'ssp','esm','crop','region_ar6_dev'], var_name='year', value_name='value')

from scipy.stats import t

def calculate_95_ci(data):
    """Calculate 95% confidence interval using t-distribution"""
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN values
    if len(data) < 2:
        return 0, 0
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    sem = std / np.sqrt(n)
    ci = t.interval(confidence=0.95, df=n-1, loc=mean, scale=sem)
    return mean, float((ci[1] - ci[0])/2)  # Return mean and half the CI width

# Calculate statistics for dy and dp
dy_stats = []
dp_stats = []

for year in years:
    year_str = str(year)
    for ssp in ssps:
        # For dy
        dy_data = dy_all[(dy_all['ssp'] == ssp)]
        dy_esm_means = dy_data.groupby('esm')[year_str].mean()
        dy_mean, dy_ci = calculate_95_ci(dy_esm_means)
        dy_stats.append({'year': year, 'ssp': ssp, 'mean': dy_mean, 'ci': dy_ci})

        # For dp
        dp_data = dp_all[(dp_all['ssp'] == ssp)]
        dp_esm_stat = dp_data.groupby('esm')[year_str].sum()

        dp_mean, dp_ci = calculate_95_ci(dp_esm_stat)
        dp_mean = dp_mean / 1e6
        dp_ci = dp_ci / 1e6

        dp_stats.append({'year': year, 'ssp': ssp, 'mean': dp_mean, 'ci': dp_ci})

dy_stats = pd.DataFrame(dy_stats)
dp_stats = pd.DataFrame(dp_stats)

"""
PLOT
"""
fig = plt.figure(figsize=(6.5, 9))
gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 1.5, 1.0, 1.0])

title_fontsize=12
label_fontsize=10
tick_fontsize=9

# Create big axes for column titles
big_axes = []
for i in range(2):
    big_axes.append(fig.add_subplot(gs[0:1, i]))
    if i == 0:
        big_axes[-1].set_title(f"{ssps[1].upper()}–{ssps[0].upper()}",fontsize=title_fontsize, weight="bold", x=0.5, y=1.37, pad=5)
    else:
        big_axes[-1].set_title(f"{ssps[2].upper()}–{ssps[0].upper()}",fontsize=title_fontsize, weight="bold", x=0.65,y=1.37, pad=5)

    big_axes[-1].axis('off')

######################## DY MAPS ########################
for index, (scenario, reference) in enumerate(ssp_diffs):

    ax = fig.add_subplot(gs[0, index], projection=ccrs.Robinson())

    if index == 0:
        ax.set_position([0.12, 0.76, 0.4, 0.2])  # [left, bottom, width, height]
    if index == 1:
        ax.set_position([0.57, 0.76, 0.4, 0.2])  # [left, bottom, width, height]

    # Add alphabet label
    ax.text(-0.03, 1.0, label_alphabets[index], transform=ax.transAxes, fontweight='bold', fontsize=label_fontsize+1)

    # Prepare data
    world = gpd.read_file(country_shape_file)

    # Get data for both scenarios
    dy_scenario = dy_long[(dy_long["ssp"] == scenario) & (dy_long["year"].isin(year_str_avg))]
    dy_reference = dy_long[(dy_long["ssp"] == reference) & (dy_long["year"].isin(year_str_avg))]

    # Calculate mean for each country
    dy_scenario = dy_scenario.groupby('country')['value'].mean().reset_index()
    dy_reference = dy_reference.groupby('country')['value'].mean().reset_index()

    # Merge and calculate difference
    dy_diff = pd.merge(dy_scenario, dy_reference, on='country', suffixes=('_scenario', '_reference'))
    dy_diff['value'] = dy_diff['value_scenario'] - dy_diff['value_reference']

    world = world.merge(dy_diff[['country', 'value']], how='left', left_on='ISO_A3', right_on='country')
    world = world.to_crs(ccrs.Robinson().proj4_init)
    world.boundary.plot(ax=ax, linewidth=0.2, edgecolor='gray')

    # Plot
    world.plot(ax=ax, column='value', cmap=cmap_dy,
              vmin=vmin_dy, vmax=vmax_dy,
              missing_kwds={"color": "white"})

    ax.set_global()
    ax.gridlines(alpha=0.2)


# Add colorbar for dy maps
cax_dy = fig.add_axes([0.27, 0.77, 0.5, 0.005]) #left,bottom,width,height
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin_dy, vmax=vmax_dy), cmap=cmap_dy)
cbar = plt.colorbar(sm, cax=cax_dy, orientation='horizontal', extend='both')
cbar.set_label(label_dy, fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)

######################## DP MAPS ########################
for index, (scenario, reference) in enumerate(ssp_diffs):
    ax = fig.add_subplot(gs[1, index], projection=ccrs.Robinson())

    if index == 0:
        ax.set_position([0.12, 0.52, 0.4, 0.2])  # [left, bottom, width, height]
    if index == 1:
        ax.set_position([0.57, 0.52, 0.4, 0.2])  # [left, bottom, width, height]

    # Add alphabet label
    ax.text(-0.03, 1.0, label_alphabets[index+2], transform=ax.transAxes, fontweight='bold', fontsize=label_fontsize + 1)

    # Prepare data
    world = gpd.read_file(country_shape_file)

    # Get data for both scenarios
    dp_scenario = dp_long[(dp_long["ssp"] == scenario) & (dp_long["year"].isin(year_str_avg))]
    dp_reference = dp_long[(dp_long["ssp"] == reference) & (dp_long["year"].isin(year_str_avg))]

    # Calculate statistics for each country
    dp_scenario = dp_scenario.groupby('country')['value'].sum().reset_index()
    dp_reference = dp_reference.groupby('country')['value'].sum().reset_index()

    # Merge and calculate difference
    dp_diff = pd.merge(dp_scenario, dp_reference, on='country', suffixes=('_scenario', '_reference'))
    dp_diff['value'] = (dp_diff['value_scenario'] - dp_diff['value_reference']) / 1e6  # Convert to Mt

    world = world.merge(dp_diff[['country', 'value']], how='left', left_on='ISO_A3', right_on='country')
    world = world.to_crs(ccrs.Robinson().proj4_init)
    world.boundary.plot(ax=ax, linewidth=0.2, edgecolor='gray')

    # Plot
    norm = colors.SymLogNorm(linthresh=0.1, linscale=0.2, vmin=vmin_dp, vmax=vmax_dp, base=10)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_dp)

    world.plot(ax=ax, column='value', cmap=cmap_dp,
               norm=norm,
              vmin=vmin_dp, vmax=vmax_dp,
              missing_kwds={"color": "white"})

    ax.set_global()
    ax.gridlines(alpha=0.2)

# Add colorbar for dp maps
cax_dp = fig.add_axes([0.27, 0.53, 0.5, 0.005]) #left,bottom,width,height
cbar = plt.colorbar(sm, cax=cax_dp, orientation='horizontal', extend='both'
                    ,format=utils.format_fn
                    )
cbar.set_label(label_dp, fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)

######################## DY TIMESERIES ########################
print ()
for index, (scenario, reference) in enumerate(ssp_diffs):
    ax = fig.add_subplot(gs[2, index])

    if index == 0:
        ax.set_position([0.12, 0.31, 0.4, 0.13])  # [left, bottom, width, height]
    if index == 1:
        ax.set_position([0.57, 0.31, 0.4, 0.13])  # [left, bottom, width, height]

    # Add alphabet label
    ax.text(-0.08, 1.1, label_alphabets[index+4], transform=ax.transAxes, fontsize=label_fontsize+1, fontweight='bold')

    #--------------------Plot GLOBAL
    global_data_scenario = dy_ipcc_melted[
        (dy_ipcc_melted['ssp'] == scenario)
        ]
    global_data_reference = dy_ipcc_melted[
        (dy_ipcc_melted['ssp'] == reference)
        ]

    # Calculate mean/sum and CI for each year
    global_stats = []
    for year in years:
        year_data_ref = global_data_reference[global_data_reference['year'] == str(year)]
        year_data_scen = global_data_scenario[global_data_scenario['year'] == str(year)]

        esm_stats_ref = year_data_ref.groupby('esm')['value'].mean()
        esm_stats_scen = year_data_scen.groupby('esm')['value'].mean()

        # Calculate the difference between esm_stats_scen and esm_stats_ref
        esm_diff = esm_stats_scen - esm_stats_ref

        # Calculate the 95% confidence interval for the differences
        mean_diff, ci_diff = calculate_95_ci(esm_diff)

        global_stats.append({'year': year, 'mean': mean_diff, 'ci': ci_diff})
        # -----------------

    global_stats = pd.DataFrame(global_stats)

    # Smooth the difference
    global_diff_smoothed = utils.get_smoothed_values(global_stats['mean'].values,
                                               window_size=window_size,
                                               method=smooth_method,
                                               np_mode=np_mode)

    ci_upper_smoothed = utils.get_smoothed_values(global_stats['mean'].values + global_stats['ci'].values,
                                            window_size=window_size,
                                            method=smooth_method,
                                            np_mode=np_mode)
    ci_lower_smoothed = utils.get_smoothed_values(global_stats['mean'].values - global_stats['ci'].values,
                                            window_size=window_size,
                                            method=smooth_method,
                                            np_mode=np_mode)

    # Plot line
    line = ax.plot(years_plot, global_diff_smoothed[plot_start_idx:plot_end_idx],
                   color="black", linestyle='solid', linewidth=1.2,
                   label='Global')

    print("------------------")
    print(f"dy diff for GLOBAL, {scenario}-{reference}, {year}: {global_diff_smoothed[plot_end_idx - 1]:.4f}")

    # Add shading for global confidence interval
    ax.fill_between(years_plot,
                    ci_lower_smoothed[plot_start_idx:plot_end_idx],
                    ci_upper_smoothed[plot_start_idx:plot_end_idx],
                    color="black", alpha=0.05)

    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.8)

    #---------------------Plot each region
    for region in regions:
        # Get data for scenario and reference for this region
        region_data_scenario = dy_ipcc_melted[
            (dy_ipcc_melted['region_ar6_dev'] == region) &
            (dy_ipcc_melted['ssp'] == scenario)
        ]
        region_data_reference = dy_ipcc_melted[
            (dy_ipcc_melted['region_ar6_dev'] == region) &
            (dy_ipcc_melted['ssp'] == reference)
        ]

        # Calculate mean/sum and CI for each year
        region_stats = []
        for year in years:
            year_data_ref = region_data_reference[region_data_reference['year'] == str(year)]
            year_data_scen = region_data_scenario[region_data_scenario['year'] == str(year)]

            esm_stats_ref = year_data_ref.groupby('esm')['value'].mean()
            esm_stats_scen = year_data_scen.groupby('esm')['value'].mean()

            # Calculate the difference between esm_stats_scen and esm_stats_ref
            esm_diff = esm_stats_scen - esm_stats_ref

            # Calculate the 95% confidence interval for the differences
            mean_diff, ci_diff = calculate_95_ci(esm_diff)

            region_stats.append({'year': year, 'mean': mean_diff, 'ci': ci_diff})
            #-----------------

        region_stats = pd.DataFrame(region_stats)

        # Smooth the difference
        region_diff_smoothed = utils.get_smoothed_values(region_stats['mean'].values,
                                                 window_size=window_size,
                                                 method=smooth_method,
                                                 np_mode=np_mode)

        ci_upper_smoothed = utils.get_smoothed_values(region_stats['mean'].values + region_stats['ci'].values,
                                                window_size=window_size,
                                                method=smooth_method,
                                                np_mode=np_mode)
        ci_lower_smoothed = utils.get_smoothed_values(region_stats['mean'].values - region_stats['ci'].values,
                                                window_size=window_size,
                                                method=smooth_method,
                                                np_mode=np_mode)


        # Plot line
        line = ax.plot(years_plot, region_diff_smoothed[plot_start_idx:plot_end_idx],
                      color=ipcc_colors[region], linestyle='solid', linewidth=0.8,
                      label=ipcc_labels[region])

        print(f"dy diff for {region}, {scenario}-{reference}, {year}: {region_diff_smoothed[plot_end_idx - 1]:.4f}")

        # Add shading for global confidence interval
        ax.fill_between(years_plot,
                       ci_lower_smoothed[plot_start_idx:plot_end_idx],
                       ci_upper_smoothed[plot_start_idx:plot_end_idx],
                       color=ipcc_colors[region], alpha=0.05)

    # Set limits and labels
    ax.set_xlim(start_year_plot, end_year_plot)
    if ylim_top_dy is not None:
        ax.set_ylim(top=ylim_top_dy)
    if ylim_bottom_dy is not None:
        ax.set_ylim(bottom=ylim_bottom_dy)

    if index == 0:
        ax.set_ylabel(label_dy, fontsize=label_fontsize)
    else:
        ax.set_yticklabels([])

    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=tick_fontsize)

######################## DP TIMESERIES ########################
for index, (scenario, reference) in enumerate(ssp_diffs):
    ax = fig.add_subplot(gs[3, index])

    if index == 0:
        ax.set_position([0.12, 0.11, 0.4, 0.13])  # [left, bottom, width, height]
    if index == 1:
        ax.set_position([0.57, 0.11, 0.4, 0.13])  # [left, bottom, width, height]

    # Add alphabet label
    ax.text(-0.08, 1.1, label_alphabets[index + 6], transform=ax.transAxes, fontsize=label_fontsize + 1,fontweight='bold')

    # --------------------Plot GLOBAL
    global_data_scenario = dp_ipcc_melted[
        (dp_ipcc_melted['ssp'] == scenario)
    ]
    global_data_reference = dp_ipcc_melted[
        (dp_ipcc_melted['ssp'] == reference)
    ]

    # Calculate mean/sum and CI for each year
    global_stats = []
    for year in years:
        year_data_ref = global_data_reference[global_data_reference['year'] == str(year)]
        year_data_scen = global_data_scenario[global_data_scenario['year'] == str(year)]

        esm_stats_ref = year_data_ref.groupby('esm')['value'].sum()
        esm_stats_scen = year_data_scen.groupby('esm')['value'].sum()

        # Calculate the difference between esm_stats_scen and esm_stats_ref
        esm_diff = esm_stats_scen - esm_stats_ref

        # Calculate the 95% confidence interval for the differences
        mean_diff, ci_diff = calculate_95_ci(esm_diff)
        mean_diff = mean_diff / 1e6
        ci_diff = ci_diff / 1e6

        global_stats.append({'year': year, 'mean': mean_diff, 'ci': ci_diff})
        # -----------------

    global_stats = pd.DataFrame(global_stats)

    # Smooth the difference
    global_diff_smoothed = utils.get_smoothed_values(global_stats['mean'].values,
                                               window_size=window_size,
                                               method=smooth_method,
                                               np_mode=np_mode)

    ci_upper_smoothed = utils.get_smoothed_values(global_stats['mean'].values + global_stats['ci'].values,
                                            window_size=window_size,
                                            method=smooth_method,
                                            np_mode=np_mode)
    ci_lower_smoothed = utils.get_smoothed_values(global_stats['mean'].values - global_stats['ci'].values,
                                            window_size=window_size,
                                            method=smooth_method,
                                            np_mode=np_mode)

    # Plot line
    line = ax.plot(years_plot, global_diff_smoothed[plot_start_idx:plot_end_idx],
                   color="black", linestyle='solid', linewidth=1.2,
                   label='Global')

    print("------------------")
    print(f"dp diff for GLOBAL, {scenario}-{reference}, {year}: {global_diff_smoothed[plot_end_idx - 1]:.4f}")

    # Add shading for global confidence interval
    ax.fill_between(years_plot,
                    ci_lower_smoothed[plot_start_idx:plot_end_idx],
                    ci_upper_smoothed[plot_start_idx:plot_end_idx],
                    color="black", alpha=0.05)

    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.8)

    # Plot each region
    for region in regions:
        # Get data for scenario and reference for this region
        region_data_scenario = dp_ipcc_melted[
            (dp_ipcc_melted['region_ar6_dev'] == region) &
            (dp_ipcc_melted['ssp'] == scenario)
        ]
        region_data_reference = dp_ipcc_melted[
            (dp_ipcc_melted['region_ar6_dev'] == region) &
            (dp_ipcc_melted['ssp'] == reference)
        ]

        # Calculate mean/sum and CI for each year
        region_stats = []
        for year in years:
            year_data_ref = region_data_reference[region_data_reference['year'] == str(year)]
            year_data_scen = region_data_scenario[region_data_scenario['year'] == str(year)]

            esm_stats_ref = year_data_ref.groupby('esm')['value'].sum()
            esm_stats_scen = year_data_scen.groupby('esm')['value'].sum()

            # Calculate the difference between esm_stats_scen and esm_stats_ref
            esm_diff = esm_stats_scen - esm_stats_ref

            # Calculate the 95% confidence interval for the differences
            mean_diff, ci_diff = calculate_95_ci(esm_diff)
            mean_diff = mean_diff / 1e6
            ci_diff = ci_diff / 1e6

            region_stats.append({'year': year, 'mean': mean_diff, 'ci': ci_diff})

        region_stats = pd.DataFrame(region_stats)

        # Smooth the difference
        region_diff_smoothed = utils.get_smoothed_values(region_stats['mean'].values,
                                                 window_size=window_size,
                                                 method=smooth_method,
                                                 np_mode=np_mode)

        ci_upper_smoothed = utils.get_smoothed_values(region_stats['mean'].values + region_stats['ci'].values,
                                                window_size=window_size,
                                                method=smooth_method,
                                                np_mode=np_mode)
        ci_lower_smoothed = utils.get_smoothed_values(region_stats['mean'].values - region_stats['ci'].values,
                                                window_size=window_size,
                                                method=smooth_method,
                                                np_mode=np_mode)

        # Plot line
        line = ax.plot(years_plot, region_diff_smoothed[plot_start_idx:plot_end_idx],
                      color=ipcc_colors[region], linestyle='solid', linewidth=0.8,
                      label=ipcc_labels[region])

        print(f"dp diff for {region}, {scenario}-{reference}, {year}: {region_diff_smoothed[plot_end_idx-1]:.4f}")

        # Add shading for global confidence interval
        ax.fill_between(years_plot,
                        ci_lower_smoothed[plot_start_idx:plot_end_idx],
                        ci_upper_smoothed[plot_start_idx:plot_end_idx],
                        color=ipcc_colors[region], alpha=0.05)

    # Set limits and labels
    ax.set_xlim(start_year_plot, end_year_plot)
    if ylim_top_dp is not None:
        ax.set_ylim(top=ylim_top_dp)
    if ylim_bottom_dp is not None:
        ax.set_ylim(bottom=ylim_bottom_dp)

    ax.set_xlabel("Year", fontsize=label_fontsize)
    if index == 0:
        ax.set_ylabel(label_dp, fontsize=label_fontsize)
    else:
        ax.set_yticklabels([])

    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=tick_fontsize)

# LEGEND
ax.legend(bbox_to_anchor=(-0.1, -0.65), loc='center', fontsize=label_fontsize, ncol=4)

plt.show()

"""
PRINT OUT DIAGNOSTICS
"""
# Analyze top 10 countries with largest damage reduction in SSP126 vs SSP370 for 2070
print("\n" + "="*50)
print("Top 10 countries with largest damage reduction in 2070 (SSP126 vs SSP370)")
print("="*50)

# For dy (yield damage)
dy_2070_370 = dy_long[(dy_long['ssp'] == 'ssp370') & (dy_long['year'] == '2070')].set_index('country')['value']
dy_2070_126 = dy_long[(dy_long['ssp'] == 'ssp126') & (dy_long['year'] == '2070')].set_index('country')['value']
dy_diff = dy_2070_126 - dy_2070_370  # Positive values mean less damage in SSP126

print("\nYield Damage Reduction (dy) [%]:")
print("-" * 30)
print(dy_diff.sort_values(ascending=False).head(10).to_frame('dy_diff').round(2))

# For dp (production damage)
dp_2070_370 = dp_long[(dp_long['ssp'] == 'ssp370') & (dp_long['year'] == '2070')].set_index('country')['value']
dp_2070_126 = dp_long[(dp_long['ssp'] == 'ssp126') & (dp_long['year'] == '2070')].set_index('country')['value']
dp_diff = (dp_2070_126 - dp_2070_370) / 1e6  # Convert to Mt, positive values mean less damage in SSP126

print("\nProduction Damage Reduction (dp) [Mt]:")
print("-" * 30)
print(dp_diff.sort_values(ascending=False).head(10).to_frame('dp_diff').round(2))


# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
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
ar6_region = "region_ar6_dev"
regions = ["ldc", "developing", "developed"]
pred_str = "t3s3"

######################## PLOTTING PARAMETERS ########################
# Maps parameters
cmap_de = plt.cm.get_cmap('hot')
cmap_de = colors.ListedColormap(cmap_de(np.linspace(0, 0.85, 256)))  # Stop at 85% to avoid white
#cmap_de = "hot" #Blues_r, YlGnBu_r, PuBuGn_r, YlOrBr_r, YlGnBu_r
cmap_degdp = "YlGnBu_r" #Purples_r, hot
vmin_degdp = -0.04
vmax_degdp = -0.005
vmin_de = -60
vmax_de = -0.01
ticks_de = [0, -0.005, -1, -20]

start_year_map = 2060
end_year_map = 2070
year_map_degdp = list(range(start_year_map, end_year_map+1, 5))
year_map_degdp_str = [str(x) for x in year_map_degdp]
year_map_de_str = [str(x) for x in range(start_year_map, end_year_map+1)]

# Timeseries parameters
ylim_top_degdp = 0.001
ylim_bottom_degdp = -0.05
ylim_top_de = 15
ylim_bottom_de = -250

window_size = 4
smooth_method = "pandas"  # options: numpy, pandas, scipy
np_mode = "full"  # full, same, valid

# Time ranges
start_year = 2020
end_year = 2070
start_year_plot = 2030  # for plotting
end_year_plot = 2070

degdp_color = "black"
de_color = "black"
ipcc_colors = {"ldc": "crimson", "developing": "goldenrod", "developed": "dodgerblue"}  # ldc,developing,developed
ipcc_labels = {"ldc": "LDC", "developing": "Developing", "developed": "Developed"}

years_degdp = list(range(start_year, end_year+1, 5))
years_de = [x for x in range(start_year, end_year + 1)]
year_str_all_de = [str(x) for x in range(start_year, end_year+1)]
year_str_all_degdp = [str(x) for x in years_degdp]

plot_start_idx_de = years_de.index(start_year_plot)
plot_end_idx_de = years_de.index(end_year_plot) + 1
years_plot_de = years_de[plot_start_idx_de:plot_end_idx_de]
years_plot_de = np.array(years_plot_de)

plot_start_idx_degdp = years_degdp.index(start_year_plot)
plot_end_idx_degdp = years_degdp.index(end_year_plot) + 1
years_plot_degdp = years_degdp[plot_start_idx_degdp:plot_end_idx_degdp]
years_plot_degdp = np.array(years_plot_degdp)

label_alphabets = utils.generate_alphabet_list(12, option="lower")
label_alphabets = ["(" + x + ")" for x in label_alphabets]
###################### ###################### ###################### ######################

start_year_hist = 1985  # 1985(isimip3b), 2007(corey)
end_year_hist = 2015  # 2015(isimip3b), 2018(corey)
start_year_fut = 2020
end_year_fut = 2070

root_dir = '../../data'
country_shape_file = f"{root_dir}/resources/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

"""
LOAD DATA
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

# Filter data
cols_de = ["country", "crop"] + ["ssp"] + ["esm"] + year_str_all_de
cols_degdp = ["country", "crop"] + ["ssp"] + ["esm"] + year_str_all_degdp
degdp_all = df_degdp[cols_degdp]
de_all = df_de[cols_de]

# COLS = country,crop,ssp,year,value
degdp_long = degdp_all.melt(id_vars=['country', 'crop', 'ssp', 'esm'], var_name='year', value_name='value')
de_long = de_all.melt(id_vars=['country', 'crop', 'ssp', 'esm'], var_name='year', value_name='value')

# Get mean over ESMs first
degdp_long = degdp_long.groupby(['country', 'crop', 'ssp', 'year'])['value'].mean().reset_index()
de_long = de_long.groupby(['country', 'crop', 'ssp', 'year'])['value'].mean().reset_index()

# Get mean/sum over crops
degdp_long = degdp_long.groupby(['country', 'ssp', 'year'])['value'].mean().reset_index()
de_long = de_long.groupby(['country', 'ssp', 'year'])['value'].sum().reset_index()

# Calculate global means/sums
degdp_global = degdp_long.groupby(['ssp', 'year'])['value'].mean().reset_index()
de_global = de_long.groupby(['ssp', 'year'])['value'].sum().reset_index()

print("----------------")
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
    ci = t.interval(confidence=0.95, df=n - 1, loc=mean, scale=sem)
    return mean, float((ci[1] - ci[0]) / 2)  # Return mean and half the CI width

# Calculate statistics for degdp and de
degdp_stats = []
de_stats = []

for year in years_degdp:
    year_str = str(year)
    for ssp in ssps:
        # For degdp
        degdp_data = degdp_all[(degdp_all['ssp'] == ssp)]
        degdp_esm_means = degdp_data.groupby('esm')[year_str].mean()
        degdp_mean, degdp_ci = calculate_95_ci(degdp_esm_means)
        degdp_stats.append({'year': year, 'ssp': ssp, 'mean': degdp_mean, 'ci': degdp_ci})

for year in years_de:
    year_str = str(year)
    for ssp in ssps:
        # For de
        de_data = de_all[(de_all['ssp'] == ssp)]
        de_esm_stat = de_data.groupby('esm')[year_str].sum()

        de_mean, de_ci = calculate_95_ci(de_esm_stat)
        de_mean = de_mean / 1e9
        de_ci = de_ci / 1e9

        de_stats.append({'year': year, 'ssp': ssp, 'mean': de_mean, 'ci': de_ci})

degdp_stats = pd.DataFrame(degdp_stats)
de_stats = pd.DataFrame(de_stats)

"""
PLOT
"""
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(4, 3, height_ratios=[2.0, 2.0, 1.0, 1.0])

title_fontsize = 14
label_fontsize = 14
tick_fontsize = 12

# Create big axes for column titles
big_axes = []
for i in range(3):
    big_axes.append(fig.add_subplot(gs[0:1, i]))
    if i == 0:
        big_axes[-1].set_title(f"{ssps[0].upper()}", fontsize=title_fontsize, weight="bold", y=0.98, pad=10)
    elif i == 1:
        big_axes[-1].set_title(f"{ssps[1].upper()}", fontsize=title_fontsize, weight="bold", y=0.98, pad=10)
    else:
        big_axes[-1].set_title(f"{ssps[2].upper()}", fontsize=title_fontsize, weight="bold", y=0.98, pad=10)

    big_axes[-1].axis('off')

# Common parameters
title_fontsize = 12
label_fontsize = 12
tick_fontsize = 12

######################## degdp MAPS ########################
for index, ssp in enumerate(ssps):
    ax = fig.add_subplot(gs[0, index], projection=ccrs.Robinson())

    # Add alphabet label
    ax.text(-0.03, 0.9, label_alphabets[index], transform=ax.transAxes, fontsize=label_fontsize)

    # Prepare data
    world = gpd.read_file(country_shape_file)
    degdp_ssp = degdp_long[(degdp_long["ssp"] == ssp) & (degdp_long["year"].isin(year_map_degdp_str))]
    degdp_ssp = degdp_ssp.groupby('country')['value'].mean().reset_index()
    world = world.merge(degdp_ssp, how='left', left_on='ISO_A3', right_on='country')
    world = world.to_crs(ccrs.Robinson().proj4_init)
    world.boundary.plot(ax=ax, linewidth=0.2, edgecolor='gray')

    # Plot
    world.plot(ax=ax, column='value', cmap=cmap_degdp,
               vmin=vmin_degdp, vmax=vmax_degdp,
               missing_kwds={"color": "white"})

    ax.set_global()
    ax.gridlines(alpha=0.2)

# Add colorbar for degdp maps
cax_degdp = fig.add_axes([0.31, 0.75, 0.44, 0.005])  # left,bottom,width,height
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin_degdp, vmax=vmax_degdp), cmap=cmap_degdp)
cbar = plt.colorbar(sm, cax=cax_degdp, orientation='horizontal', extend='both')
cbar.set_label(r'$\overline{degdp}$ [%GDP]', fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)

######################## de MAPS ########################
for index, ssp in enumerate(ssps):
    ax = fig.add_subplot(gs[1, index], projection=ccrs.Robinson())

    # Add alphabet label
    ax.text(-0.03, 0.9, label_alphabets[index + 3], transform=ax.transAxes, fontsize=label_fontsize)

    # Prepare data
    world = gpd.read_file(country_shape_file)
    de_ssp = de_long[(de_long["ssp"] == ssp) & (de_long["year"].isin(year_map_de_str))]
    de_ssp = de_ssp.groupby('country')['value'].sum().reset_index()

    world = world.merge(de_ssp, how='left', left_on='ISO_A3', right_on='country')
    world = world.to_crs(ccrs.Robinson().proj4_init)
    world.boundary.plot(ax=ax, linewidth=0.2, edgecolor='gray')
    world['value'] = world['value'] / 1e9  # Convert to Mt

    # Plot
    norm = plt.Normalize(vmin=vmin_de, vmax=vmax_de)
    #norm = colors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=vmin_de, vmax=vmax_de, base=10)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_de)
    world.plot(ax=ax, column='value', cmap=cmap_de,
               norm=norm,
               vmin=vmin_de, vmax=vmax_de,
               missing_kwds={"color": "white"})

    ax.set_global()
    ax.gridlines(alpha=0.2)

# Add colorbar for de maps
cax_de = fig.add_axes([0.31, 0.46, 0.44, 0.005])  # left,bottom,width,height
cbar = plt.colorbar(sm, cax=cax_de, orientation='horizontal', extend='both'
                    ,format=utils.format_fn
                    #,ticks=ticks_de
                    )
cbar.set_label(r'$de$ [B US\$]', fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)

######################## degdp TIMESERIES ########################
for index, ssp in enumerate(ssps):
    ax = fig.add_subplot(gs[2, index])

    # Add alphabet label
    ax.text(0.02, 0.1, label_alphabets[index + 6], transform=ax.transAxes, fontsize=label_fontsize)

    # Plot global mean first
    df = degdp_stats[degdp_stats['ssp'] == ssp]

    # Get smoothed values for global mean
    mean_smoothed = utils.get_smoothed_values(df['mean'].values, window_size=window_size,
                                        method=smooth_method, np_mode=np_mode)
    ci_upper_smoothed = utils.get_smoothed_values(df['mean'].values + df['ci'].values,
                                            window_size=window_size,
                                            method=smooth_method, np_mode=np_mode)
    ci_lower_smoothed = utils.get_smoothed_values(df['mean'].values - df['ci'].values,
                                            window_size=window_size,
                                            method=smooth_method, np_mode=np_mode)

    # Plot global line
    line = ax.plot(years_plot_degdp, mean_smoothed[plot_start_idx_degdp:plot_end_idx_degdp],
                   color="black", linestyle='solid', linewidth=1.2,
                   label='Global')

    print ("-----------------")
    print (f"GLOBAL degdp {ssp}:",mean_smoothed[plot_end_idx_degdp-1])

    # Add shading for global confidence interval
    ax.fill_between(years_plot_degdp,
                    ci_lower_smoothed[plot_start_idx_degdp:plot_end_idx_degdp],
                    ci_upper_smoothed[plot_start_idx_degdp:plot_end_idx_degdp],
                    color="black", alpha=0.05)

    # Plot each region
    for region in regions:
        # Filter data for this region and SSP
        region_data = degdp_ipcc_melted[
            (degdp_ipcc_melted['region_ar6_dev'] == region) &
            (degdp_ipcc_melted['ssp'] == ssp) &
            (degdp_ipcc_melted['year'].isin(year_str_all_degdp))
            ]

        # Calculate mean and CI for each year
        region_stats = []
        for year in years_degdp:
            year_data = region_data[region_data['year'] == str(year)]
            esm_means = year_data.groupby('esm')['value'].mean()
            mean, ci = calculate_95_ci(esm_means)
            region_stats.append({'year': year, 'mean': mean, 'ci': ci})

        region_stats = pd.DataFrame(region_stats)

        # Get smoothed values
        mean_smoothed = utils.get_smoothed_values(region_stats['mean'].values,
                                            window_size=window_size,
                                            method=smooth_method, np_mode=np_mode)
        ci_upper_smoothed = utils.get_smoothed_values(region_stats['mean'].values + region_stats['ci'].values,
                                                window_size=window_size,
                                                method=smooth_method, np_mode=np_mode)
        ci_lower_smoothed = utils.get_smoothed_values(region_stats['mean'].values - region_stats['ci'].values,
                                                window_size=window_size,
                                                method=smooth_method, np_mode=np_mode)

        # Plot line
        line = ax.plot(years_plot_degdp, mean_smoothed[plot_start_idx_degdp:plot_end_idx_degdp],
                       color=ipcc_colors[region], linestyle='solid', linewidth=0.8,
                       label=ipcc_labels[region])

        print(f"{region} degdp {ssp}:", mean_smoothed[plot_end_idx_degdp - 1])

        # Add shading for confidence interval
        ax.fill_between(years_plot_degdp,
                        ci_lower_smoothed[plot_start_idx_degdp:plot_end_idx_degdp],
                        ci_upper_smoothed[plot_start_idx_degdp:plot_end_idx_degdp],
                        color=ipcc_colors[region], alpha=0.05)

    # Set limits and labels
    ax.set_xlim(start_year_plot, end_year_plot)
    if ylim_top_degdp is not None:
        ax.set_ylim(top=ylim_top_degdp)
    if ylim_bottom_degdp is not None:
        ax.set_ylim(bottom=ylim_bottom_degdp)

    ax.set_xlabel("Year", fontsize=label_fontsize)
    if index == 0:
        ax.set_ylabel(r'$\overline{degdp}$ [%GDP]', fontsize=label_fontsize)
    else:
        ax.set_yticklabels([])

    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=tick_fontsize)

######################## de TIMESERIES ########################
for index, ssp in enumerate(ssps):
    ax = fig.add_subplot(gs[3, index])

    # Add alphabet label
    ax.text(0.02, 0.1, label_alphabets[index + 9], transform=ax.transAxes, fontsize=label_fontsize)

    # Plot global mean first
    df = de_stats[de_stats['ssp'] == ssp]

    # Get smoothed values for global mean
    mean_smoothed = utils.get_smoothed_values(df['mean'].values, window_size=window_size,
                                        method=smooth_method, np_mode=np_mode)
    ci_upper_smoothed = utils.get_smoothed_values(df['mean'].values + df['ci'].values,
                                            window_size=window_size,
                                            method=smooth_method, np_mode=np_mode)
    ci_lower_smoothed = utils.get_smoothed_values(df['mean'].values - df['ci'].values,
                                            window_size=window_size,
                                            method=smooth_method, np_mode=np_mode)

    # Plot global line
    line = ax.plot(years_plot_de, mean_smoothed[plot_start_idx_de:plot_end_idx_de],
                   color="black", linestyle='solid', linewidth=1.2,
                   label='Global')

    print ("-----------------")
    print (f"GLOBAL de {ssp}:",mean_smoothed[plot_end_idx_de-1])

    # Add shading for global confidence interval
    ax.fill_between(years_plot_de,
                    ci_lower_smoothed[plot_start_idx_de:plot_end_idx_de],
                    ci_upper_smoothed[plot_start_idx_de:plot_end_idx_de],
                    color="black", alpha=0.05)

    # Plot each region
    for region in regions:
        # Filter data for this region and SSP
        region_data = de_ipcc_melted[
            (de_ipcc_melted['region_ar6_dev'] == region) &
            (de_ipcc_melted['ssp'] == ssp) &
            (de_ipcc_melted['year'].isin(year_str_all_de))
            ]

        # Calculate mean/sum and CI for each year
        region_stats = []
        for year in years_de:
            year_data = region_data[region_data['year'] == str(year)]
            esm_stats = year_data.groupby('esm')['value'].sum()
            mean, ci = calculate_95_ci(esm_stats)

            # Convert to Billion US$
            mean = mean / 1e9
            ci = ci / 1e9
            region_stats.append({'year': year, 'mean': mean, 'ci': ci})

        region_stats = pd.DataFrame(region_stats)

        # Get smoothed values
        mean_smoothed = utils.get_smoothed_values(region_stats['mean'].values,
                                            window_size=window_size,
                                            method=smooth_method, np_mode=np_mode)
        ci_upper_smoothed = utils.get_smoothed_values(region_stats['mean'].values + region_stats['ci'].values,
                                                window_size=window_size,
                                                method=smooth_method, np_mode=np_mode)
        ci_lower_smoothed = utils.get_smoothed_values(region_stats['mean'].values - region_stats['ci'].values,
                                                window_size=window_size,
                                                method=smooth_method, np_mode=np_mode)

        # Plot line
        line = ax.plot(years_plot_de, mean_smoothed[plot_start_idx_de:plot_end_idx_de],
                       color=ipcc_colors[region], linestyle='solid', linewidth=0.8,
                       label=ipcc_labels[region])

        print (f"{region} de {ssp}:",mean_smoothed[plot_end_idx_de-1])

        # Add shading for confidence interval
        ax.fill_between(years_plot_de,
                        ci_lower_smoothed[plot_start_idx_de:plot_end_idx_de],
                        ci_upper_smoothed[plot_start_idx_de:plot_end_idx_de],
                        color=ipcc_colors[region], alpha=0.05)

    # Set limits and labels
    ax.set_xlim(start_year_plot, end_year_plot)
    if ylim_top_de is not None:
        ax.set_ylim(top=ylim_top_de)
    if ylim_bottom_de is not None:
        ax.set_ylim(bottom=ylim_bottom_de)

    ax.set_xlabel("Year", fontsize=label_fontsize)
    if index == 0:
        ax.set_ylabel(r'$de$ [B US\$]', fontsize=label_fontsize)
    else:
        ax.set_yticklabels([])

    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=tick_fontsize)

ax.legend(bbox_to_anchor=(-0.7, -0.7), loc='center', fontsize=tick_fontsize, ncol=4)

# -----------------------------------------
fig.subplots_adjust(left=0.09,
                    bottom=0.1,
                    right=0.98,
                    top=0.96,
                    wspace=0.14,
                    hspace=0.45)

"""
wspace and hspace specify the space reserved between Matplotlib subplots. They are the fractions of axis width and height, respectively.
left, right, top and bottom parameters specify four sides of the subplots’ positions. They are the fractions of the width and height of the figure.
top and bottom should add up to 1.0
"""

plt.show()

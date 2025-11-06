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
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors
import seaborn as sns
import geopandas as gpd
import cartopy
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('../')
import utils

"""
PARAMETERS
"""
crop = "maize"  # maize, wheat, soy
vars = ["tmax", "spei"]
var_names = {"tmax": "Surface Max. temperature",
             "spei": "SPEI-12"}
var_units = {"tmax": "K",
             "spei": "-"}

ref_start_year = 1990
ref_end_year = 2000
fut_start_year = 2010
fut_end_year = 2019
ts_start_year = 1990
ts_end_year = 2019

root_dir = '../../data'
country_shape_file = f"{root_dir}/resources/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

"""
GET DATA AND PLOT
"""
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])

title_fontsize = 14
label_fontsize = 12
tick_fontsize = 11
legend_fontsize = 11

subplt_labels = utils.generate_alphabet_list(4, "lower")

# Read world map data
world = gpd.read_file(country_shape_file)

# Process and plot each variable
all_changes = {}  # Store all changes to determine overall vmin/vmax for each variable
all_timeseries_data = {}  # Store time series data for each variable

# First pass to get overall vmin/vmax and collect time series data
for var in vars:
    var_file = f"{root_dir}/historical/climate/{var}/{var}_mean_country_{crop}_yearly.csv"

    # Read and process data
    df = pd.read_csv(var_file)

    # Calculate reference period average for anomalies
    df_ref = df[(df['date'] >= ref_start_year) & (df['date'] <= ref_end_year)]
    ref_avg_by_country = df_ref.groupby('country')[var].mean()

    # Calculate yearly anomalies for time series
    yearly_data = []
    for year in range(ts_start_year, ts_end_year + 1):
        df_year = df[df['date'] == year]
        # Calculate anomaly for each country and then take global mean
        df_year = df_year.merge(ref_avg_by_country.reset_index(), on='country', suffixes=('', '_ref'))
        df_year['anomaly'] = df_year[var] - df_year[f'{var}_ref']
        yearly_data.append({
            'year': year,
            'anomaly': df_year['anomaly'].mean()
        })

    all_timeseries_data[var] = pd.DataFrame(yearly_data)

    # Calculate future period average for maps
    df_fut = df[(df['date'] >= fut_start_year) & (df['date'] <= fut_end_year)]
    df_fut_avg = df_fut.groupby('country')[var].mean().reset_index()
    df_fut_avg = df_fut_avg.rename(columns={var: 'fut_value'})

    # Calculate changes for map color scale
    df_ref_avg = df_ref.groupby('country')[var].mean().reset_index()
    df_ref_avg = df_ref_avg.rename(columns={var: 'ref_value'})
    df_merged = pd.merge(df_ref_avg, df_fut_avg, on='country')
    df_merged['abs_change'] = df_merged['fut_value'] - df_merged['ref_value']
    all_changes[var] = df_merged['abs_change'].tolist()

# Create custom colormaps for each variable
cmap_tmax = plt.cm.RdBu_r  # Red for hot, Blue for cold
cmap_spei = plt.cm.BrBG  # Brown for dry, Green for wet
cmap_rootmoist = plt.cm.BrBG  # Brown for dry, Green for wet
var_cmaps = {"tmax": cmap_tmax, "spei": cmap_spei}

# Second pass to create plots
for idx, var in enumerate(vars):
    var_file = f"{root_dir}/historical/climate/{var}/{var}_mean_country_{crop}_yearly.csv"

    # Read and process data for maps
    df = pd.read_csv(var_file)

    df_ref = df[(df['date'] >= ref_start_year) & (df['date'] <= ref_end_year)]
    df_ref_avg = df_ref.groupby('country')[var].mean().reset_index()
    df_ref_avg = df_ref_avg.rename(columns={var: 'ref_value'})

    df_fut = df[(df['date'] >= fut_start_year) & (df['date'] <= fut_end_year)]
    df_fut_avg = df_fut.groupby('country')[var].mean().reset_index()
    df_fut_avg = df_fut_avg.rename(columns={var: 'fut_value'})

    df_merged = pd.merge(df_ref_avg, df_fut_avg, on='country')
    df_merged['abs_change'] = df_merged['fut_value'] - df_merged['ref_value']

    # Calculate vmin/vmax for this variable
    changes = np.array(all_changes[var])
    vmax = max(abs(np.percentile(changes, 2)), abs(np.percentile(changes, 90)))
    vmin = -vmax  # Make it symmetric around zero

    plt_title = f"({subplt_labels[idx]}) {var_names[var]}"

    # Create map subplot
    ax_map = fig.add_subplot(gs[0, idx], projection=ccrs.Robinson())
    ax_map.set_global()

    # Plot map
    df_plot = df_merged[['country', 'abs_change']].rename(columns={'abs_change': 'value'})
    world_plot = world.merge(df_plot, how='left', left_on='ADM0_A3', right_on='country')
    world_plot = world_plot.to_crs(ccrs.Robinson().proj4_init)

    world_plot.boundary.plot(ax=ax_map, linewidth=0.2, edgecolor='gray')
    plot = world_plot.plot(column='value', ax=ax_map, legend=True,
                           cmap=var_cmaps[var],
                           vmin=vmin, vmax=vmax,
                           missing_kwds={'color': 'white'},
                           legend_kwds={
                               'orientation': 'horizontal',
                               # 'orientation': 'vertical',
                               'extend': 'both',
                               'fraction': 0.05,
                               'pad': 0.1,
                               'aspect': 40,
                               'shrink': 0.8})

    # Add colorbar label below
    ax_map.get_figure().axes[-1].set_xlabel(rf'$d${var} [{var_units[var]}]',
                                            fontsize=label_fontsize)
    ax_map.get_figure().axes[-1].tick_params(labelsize=tick_fontsize)

    # Add map title
    ax_map.set_title(plt_title, fontsize=title_fontsize, pad=10)
    ax_map.coastlines(linewidth=0.2)
    ax_map.gridlines(linestyle='--', alpha=0.5)

    # Create time series subplot
    plt_title = f"({subplt_labels[idx + 2]}) {var_names[var]}"
    ax_ts = fig.add_subplot(gs[1, idx])

    # Plot time series
    ts_data = all_timeseries_data[var]
    ax_ts.plot(ts_data['year'], ts_data['anomaly'], '-o', color="dimgray", markersize=2)
    ax_ts.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_ts.grid(True, alpha=0.3)

    # Format time series plot
    ax_ts.set_xlim(ts_start_year, ts_end_year)
    ax_ts.tick_params(axis='both', labelsize=tick_fontsize)

    # Add time series title
    ax_ts.set_title(plt_title, fontsize=title_fontsize, pad=10)

    # Add y-label for all time series plots
    ax_ts.set_ylabel(rf'$d${var} [{var_units[var]}]', fontsize=label_fontsize)

    # Add x-label only to middle plot
    if idx == 1:
        ax_ts.set_xlabel('Year', fontsize=label_fontsize)

# Adjust layout
fig.subplots_adjust(left=0.08,
                    bottom=0.1,
                    right=0.99,
                    top=0.95,
                    wspace=0.25,
                    hspace=0.55)

plt.show()
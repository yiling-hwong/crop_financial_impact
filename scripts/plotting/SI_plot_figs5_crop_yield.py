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
crops = ["maize", "wheat", "soy"]
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
fig = plt.figure(figsize=(12, 5.5))
gs = fig.add_gridspec(2, 3, height_ratios=[2.0, 1.0])

title_fontsize = 14
label_fontsize = 12
tick_fontsize = 11
legend_fontsize = 11

subplt_labels = utils.generate_alphabet_list(6, "lower")

# Read world map data
world = gpd.read_file(country_shape_file)

# Process and plot each crop
all_changes = []  # Store all changes to determine overall vmin/vmax
all_timeseries_data = {}  # Store time series data for each crop

# First pass to get overall vmin/vmax and collect time series data
for crop in crops:
    faostat_file = f"{root_dir}/historical/faostat/FAOSTAT_{crop}.csv"

    # Read and process FAOSTAT data
    df = pd.read_csv(faostat_file)
    df_yield = df[df['Element'] == 'Yield']

    # Calculate reference period average for anomalies
    df_ref = df_yield[(df_yield['Year'] >= ref_start_year) & (df_yield['Year'] <= ref_end_year)]
    ref_avg_by_country = df_ref.groupby('Area')['Value'].mean()

    # Calculate yearly anomalies for time series
    yearly_data = []
    for year in range(ts_start_year, ts_end_year + 1):
        df_year = df_yield[df_yield['Year'] == year]
        # Calculate anomaly for each country and then take global mean
        df_year = df_year.merge(ref_avg_by_country.reset_index(), on='Area', suffixes=('', '_ref'))
        # Convert from 100g/ha to tonnes/ha
        df_year['anomaly'] = (df_year['Value'] - df_year['Value_ref']) / 10000
        yearly_data.append({
            'year': year,
            'anomaly': df_year['anomaly'].mean()
        })

    all_timeseries_data[crop] = pd.DataFrame(yearly_data)

    # Calculate future period average for maps
    df_fut = df_yield[(df_yield['Year'] >= fut_start_year) & (df_yield['Year'] <= fut_end_year)]
    df_fut_avg = df_fut.groupby('Area')['Value'].mean().reset_index()
    df_fut_avg = df_fut_avg.rename(columns={'Value': 'fut_yield', 'Area': 'country'})

    # Calculate changes for map color scale
    df_ref_avg = df_ref.groupby('Area')['Value'].mean().reset_index()
    df_ref_avg = df_ref_avg.rename(columns={'Value': 'ref_yield', 'Area': 'country'})
    df_merged = pd.merge(df_ref_avg, df_fut_avg, on='country')
    # Convert from 100g/ha to tonnes/ha for maps too
    df_merged['abs_change'] = (df_merged['fut_yield'] - df_merged['ref_yield']) / 10000
    all_changes.extend(df_merged['abs_change'].tolist())

# Calculate overall vmin/vmax for consistent scale
vmax = max(abs(np.percentile(all_changes, 2)), abs(np.percentile(all_changes, 80)))
vmin = -vmax  # Make it symmetric around zero

# Calculate overall y-axis limits for time series
ts_ymin = min([df['anomaly'].min() for df in all_timeseries_data.values()])
ts_ymax = max([df['anomaly'].max() for df in all_timeseries_data.values()])
# Add 5% padding to y-axis limits
ts_yrange = ts_ymax - ts_ymin
ts_ymin -= 0.05 * ts_yrange
ts_ymax += 0.05 * ts_yrange

# Create custom colormap (brown to green)
colors_underscore = ['#8C510A', '#DFC27D', '#F5F5F5', '#80CD32', '#003C00']
n_bins = 256
cmap = colors.LinearSegmentedColormap.from_list('BrownGreen', colors_underscore, N=n_bins)

# Second pass to create plots
for idx, crop in enumerate(crops):
    faostat_file = f"{root_dir}/historical/faostat/FAOSTAT_{crop}.csv"

    # Read and process FAOSTAT data for maps
    df = pd.read_csv(faostat_file)
    df_yield = df[df['Element'] == 'Yield']

    df_ref = df_yield[(df_yield['Year'] >= ref_start_year) & (df_yield['Year'] <= ref_end_year)]
    df_ref_avg = df_ref.groupby('Area')['Value'].mean().reset_index()
    df_ref_avg = df_ref_avg.rename(columns={'Value': 'ref_yield', 'Area': 'country'})

    df_fut = df_yield[(df_yield['Year'] >= fut_start_year) & (df_yield['Year'] <= fut_end_year)]
    df_fut_avg = df_fut.groupby('Area')['Value'].mean().reset_index()
    df_fut_avg = df_fut_avg.rename(columns={'Value': 'fut_yield', 'Area': 'country'})

    df_merged = pd.merge(df_ref_avg, df_fut_avg, on='country')
    df_merged['abs_change'] = (df_merged['fut_yield'] - df_merged['ref_yield']) / 10000

    plt_title = f"({subplt_labels[idx]}) {crop.capitalize()}"

    # Create map subplot
    ax_map = fig.add_subplot(gs[0, idx], projection=ccrs.Robinson())
    ax_map.set_global()

    # Plot map
    df_plot = df_merged[['country', 'abs_change']].rename(columns={'abs_change': 'value'})
    world_plot = world.merge(df_plot, how='left', left_on=['NAME'], right_on=['country'])
    world_plot = world_plot.to_crs(ccrs.Robinson().proj4_init)

    world_plot.boundary.plot(ax=ax_map, linewidth=0.2, edgecolor='gray')
    plot = world_plot.plot(column='value', ax=ax_map, legend=False, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           missing_kwds={'color': 'white'})

    # Add map title and label
    ax_map.set_title(plt_title, fontsize=title_fontsize, pad=10)
    ax_map.coastlines(linewidth=0.2)
    ax_map.gridlines(linestyle='--', alpha=0.5)

    # Create time series subplot
    plt_title = f"({subplt_labels[idx + 3]}) {crop.capitalize()}"
    ax_ts = fig.add_subplot(gs[1, idx])

    # Plot time series
    ts_data = all_timeseries_data[crop]
    ax_ts.plot(ts_data['year'], ts_data['anomaly'], '-o', color="dimgray", markersize=2)
    ax_ts.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_ts.grid(True, alpha=0.3)

    # Format time series plot
    ax_ts.set_xlim(ts_start_year, ts_end_year)
    ax_ts.set_ylim(ts_ymin, ts_ymax)
    ax_ts.tick_params(axis='both', labelsize=tick_fontsize)

    # Add time series title and label
    ax_ts.set_title(plt_title, fontsize=title_fontsize, pad=10)

    if idx == 0:
        ax_ts.set_ylabel(r'Yield Anomaly [t/ha]', fontsize=label_fontsize)
    else:
        ax_ts.set_yticklabels([])

    if idx == 1:
        ax_ts.set_xlabel('Year', fontsize=label_fontsize)

        # COLORBAR
        divider = make_axes_locatable(ax_map)
        cax = fig.add_axes([0.275, 0.04 + (0.24 * (3 - idx)), 0.5, 0.01])  # Positioning [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
        cbar.set_label(r'Yield Anomaly [t/ha]', fontsize=label_fontsize)

# -----------------------------------------
fig.subplots_adjust(left=0.06,
                    bottom=0.1,
                    right=0.99,
                    top=0.99,
                    wspace=0.06,  # 0.1
                    hspace=0.5)  # 0.85

"""
wspace and hspace specify the space reserved between Matplotlib subplots. They are the fractions of axis width and height, respectively.
left, right, top and bottom parameters specify four sides of the subplots’ positions. They are the fractions of the width and height of the figure.
top and bottom should add up to 1.0
"""

plt.show()
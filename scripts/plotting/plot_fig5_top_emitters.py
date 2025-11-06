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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import geopandas as gpd
import cartopy
import cartopy.crs as ccrs
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('../')
import utils

"""
PARAMETERS
"""
crops = ["maize", "wheat", "soy"]  # all three crops
estring_top = 'top10'
estring_bot = 'bot50'
pred_str = "t3s3"

###################### PLOTTING PARAMETERS ######################
# Maps parameter
cmap_dy = "Reds_r"
cmap_dp = "copper"
cmap_de = "Blues_r"
cmap_degdp = "Purples_r"
vmin_dy = -3.0
vmax_dy = 0.1
vmin_dp = -10
vmax_dp = -0.001
ticks_dp = [-0.1, -1, -10]

vmin_degdp = -4e-3
vmax_degdp = 0.0
vmin_de = -0.1
vmax_de = -0.001
ticks_de = [-0.001, -0.01, -0.1]

start_year_avg = 2000
end_year_avg = 2019

# Timeseries parameters
ylim_top_dy = 0.2
ylim_bottom_dy = -2.3
ylim_top_dp = 3.0
ylim_bottom_dp = -45
ylim_top_degdp = 0.002
ylim_bottom_degdp = -0.014
ylim_top_de = 0.5
ylim_bottom_de = -10

window_size = 1
smooth_method = "numpy"  # options: numpy, pandas, scipy
np_mode = "full"  # full, same, valid

start_year = 1990
end_year = 2019
start_year_plot = 1990  # for plotting
end_year_plot = 2019

years = [x for x in range(start_year, end_year + 1)]
year_str = [str(x) for x in range(start_year, end_year + 1)]
years_plot = [x for x in range(start_year_plot, end_year_plot + 1)]

plot_start_idx = years.index(start_year_plot)
plot_end_idx = years.index(end_year_plot) + 1

# Colors for different variables
dy_color = "indigo"
dp_color = "teal"
degdp_color = "royalblue"
de_color = "crimson"
color_all = "dimgrey"
color_1 = "darkred"
color_2 = "darkblue"

quantile_low = 0.25
quantile_high = 0.75
##################################################################

start_year_hist = 2007
end_year_hist = 2018
start_year_fut = 1990
end_year_fut = 2019

root_dir = '../../data'
country_shape_file = f"{root_dir}/resources/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

"""
LOAD DATA
"""
df_dy_top_all = []
df_dp_top_all = []
df_dy_bot_all = []
df_dp_bot_all = []
df_degdp_top_all = []
df_de_top_all = []
df_degdp_bot_all = []
df_de_bot_all = []

# Load data for all crops
for crop in crops:
    # Files for top10
    dy_file_top = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dy_{estring_top}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    dp_file_top = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dp_{estring_top}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    degdp_file_top = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/degdp_{estring_top}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    de_file_top = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/de_{estring_top}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"

    # Files for bot50
    dy_file_bot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dy_{estring_bot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    dp_file_bot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/dp_{estring_bot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    degdp_file_bot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/degdp_{estring_bot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"
    de_file_bot = f"{root_dir}/historical/linregress_outputs/{crop}/topbot/de_{estring_bot}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}.csv"

    # Read Data
    df_dy_top = pd.read_csv(dy_file_top)
    df_dp_top = pd.read_csv(dp_file_top)
    df_degdp_top = pd.read_csv(degdp_file_top)
    df_de_top = pd.read_csv(de_file_top)

    df_dy_bot = pd.read_csv(dy_file_bot)
    df_dp_bot = pd.read_csv(dp_file_bot)
    df_degdp_bot = pd.read_csv(degdp_file_bot)
    df_de_bot = pd.read_csv(de_file_bot)

    # Add crop column
    df_dy_top['crop'] = crop
    df_dp_top['crop'] = crop
    df_degdp_top['crop'] = crop
    df_de_top['crop'] = crop

    df_dy_bot['crop'] = crop
    df_dp_bot['crop'] = crop
    df_degdp_bot['crop'] = crop
    df_de_bot['crop'] = crop

    # Append to lists
    df_dy_top_all.append(df_dy_top)
    df_dp_top_all.append(df_dp_top)
    df_dy_bot_all.append(df_dy_bot)
    df_dp_bot_all.append(df_dp_bot)
    df_degdp_top_all.append(df_degdp_top)
    df_de_top_all.append(df_de_top)
    df_degdp_bot_all.append(df_degdp_bot)
    df_de_bot_all.append(df_de_bot)

# Combine all crops data
df_dy_top = pd.concat(df_dy_top_all, ignore_index=True)
df_dp_top = pd.concat(df_dp_top_all, ignore_index=True)
df_dy_bot = pd.concat(df_dy_bot_all, ignore_index=True)
df_dp_bot = pd.concat(df_dp_bot_all, ignore_index=True)
df_degdp_top = pd.concat(df_degdp_top_all, ignore_index=True)
df_de_top = pd.concat(df_de_top_all, ignore_index=True)
df_degdp_bot = pd.concat(df_degdp_bot_all, ignore_index=True)
df_de_bot = pd.concat(df_de_bot_all, ignore_index=True)

# For maps: calculate averages over specified years
year_str_avg = [str(x) for x in range(start_year_avg, end_year_avg + 1)]
cols_avg = ["country", "crop"] + year_str_avg

# Process top10 data
dy_top = df_dy_top[cols_avg]
dp_top = df_dp_top[cols_avg]
degdp_top = df_degdp_top[cols_avg]
de_top = df_de_top[cols_avg]

dy_top["dy_avg"] = dy_top.loc[:, f"{start_year_avg}":f"{end_year_avg}"].mean(axis=1)
dp_top[f"dp_sum"] = dp_top.loc[:, f"{start_year_avg}":f"{end_year_avg}"].sum(axis=1) / 1e6
degdp_top["degdp_avg"] = degdp_top.loc[:, f"{start_year_avg}":f"{end_year_avg}"].mean(axis=1)
de_top[f"de_sum"] = de_top.loc[:, f"{start_year_avg}":f"{end_year_avg}"].sum(axis=1) / 1e9

# Process bot50 data
dy_bot = df_dy_bot[cols_avg]
dp_bot = df_dp_bot[cols_avg]
degdp_bot = df_degdp_bot[cols_avg]
de_bot = df_de_bot[cols_avg]

dy_bot["dy_avg"] = dy_bot.loc[:, f"{start_year_avg}":f"{end_year_avg}"].mean(axis=1)
dp_bot[f"dp_sum"] = dp_bot.loc[:, f"{start_year_avg}":f"{end_year_avg}"].sum(axis=1) / 1e6
degdp_bot["degdp_avg"] = degdp_bot.loc[:, f"{start_year_avg}":f"{end_year_avg}"].mean(axis=1)
de_bot[f"de_sum"] = de_bot.loc[:, f"{start_year_avg}":f"{end_year_avg}"].sum(axis=1) / 1e9

# Calculate differences for mapping
df_dy_diff = pd.merge(dy_top[["country", "crop", "dy_avg"]], dy_bot[["country", "crop", "dy_avg"]],
                      on=["country", "crop"], suffixes=('_top', '_bot'))
df_dy_diff["dy_diff"] = df_dy_diff["dy_avg_top"] - df_dy_diff["dy_avg_bot"]

df_dp_diff = pd.merge(dp_top[["country", "crop", f"dp_sum"]], dp_bot[["country", "crop", f"dp_sum"]],
                      on=["country", "crop"], suffixes=('_top', '_bot'))
df_dp_diff[f"dp_sum_diff"] = df_dp_diff[f"dp_sum_top"] - df_dp_diff[f"dp_sum_bot"]

df_degdp_diff = pd.merge(degdp_top[["country", "crop", "degdp_avg"]], degdp_bot[["country", "crop", "degdp_avg"]],
                         on=["country", "crop"], suffixes=('_top', '_bot'))
df_degdp_diff["degdp_diff"] = df_degdp_diff["degdp_avg_top"] - df_degdp_diff["degdp_avg_bot"]

df_de_diff = pd.merge(de_top[["country", "crop", f"de_sum"]], de_bot[["country", "crop", f"de_sum"]],
                      on=["country", "crop"], suffixes=('_top', '_bot'))
df_de_diff[f"de_sum_diff"] = df_de_diff[f"de_sum_top"] - df_de_diff[f"de_sum_bot"]

"""
PLOT
"""
world = gpd.read_file(country_shape_file)
world = world.rename(columns={'ADM0_A3': 'country'})

# Create figure with GridSpec for different row heights
fig = plt.figure(figsize=(9, 10))
gs = fig.add_gridspec(4, 2, height_ratios=[2.0, 2.0, 1, 1])

# First row: dy and dp maps
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())

# Second row: de and degdp maps
ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson())
ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())

# Third row: dy and dp timeseries
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])

# Fourth row: de and degdp timeseries
ax7 = fig.add_subplot(gs[3, 0])
ax8 = fig.add_subplot(gs[3, 1])

# Plot maps
# TOP 10
map_data = [
    (ax1, df_dy_diff, "dy_avg_top", r'(a) $\overline{dy}$ [%]', cmap_dy, vmin_dy, vmax_dy, "[%]"),
    (ax2, df_dp_diff, f"dp_sum_top", "(b) dp [Mt]", cmap_dp, vmin_dp, vmax_dp, "[Mt]"),
    (ax3, df_degdp_diff, "degdp_avg_top", r'(c) $\overline{degdp}$ [%GDP]', cmap_degdp, vmin_degdp, vmax_degdp,"[%GDP]"),
    (ax4, df_de_diff, f"de_sum_top", "(d) de [Billion US\$]", cmap_de, vmin_de, vmax_de, "[Billion US$]")
]

title_fontsize = 12
label_fontsize = 12
tick_fontsize = 11

for ax, data, column, title, cmap, vmin, vmax, cbar_label in map_data:

    ######################### MAPS #########################

    # Merge data with world geometries
    world_data = world.merge(data, on='country', how='left')

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    # norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    if ax == ax2:
        norm = colors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=vmin, vmax=vmax, base=10)

    if ax == ax4:
        norm = colors.SymLogNorm(linthresh=0.001, linscale=0.001, vmin=vmin, vmax=vmax, base=10)

    # Plot map
    world_data = world_data.to_crs(ccrs.Robinson().proj4_init)
    world_data.plot(ax=ax, color='white', edgecolor='gray', linewidth=0.5)
    world_data.plot(column=column, ax=ax,
                    norm=norm,
                    cmap=cmap, missing_kwds={'color': 'white', 'label': 'No Data'},
                    vmin=vmin, vmax=vmax)

    ax.set_title(title, pad=10, fontsize=title_fontsize)
    ax.set_global()
    ax.gridlines(alpha=0.2)

    ###### COLORBAR
    pos = ax.get_position()

    if ax == ax1:
        cax = fig.add_axes([pos.x0 + 0.02, pos.y0 + 0.055, pos.width - 0.07, 0.005])
    if ax == ax2:
        cax = fig.add_axes([pos.x0 + 0.065, pos.y0 + 0.055, pos.width - 0.07, 0.005])
    if ax == ax3:
        cax = fig.add_axes([pos.x0 + 0.02, pos.y0 + 0.014, pos.width - 0.07, 0.005])
    if ax == ax4:
        cax = fig.add_axes([pos.x0 + 0.065, pos.y0 + 0.014, pos.width - 0.07, 0.005])

    if ax == ax2:
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          format=utils.format_fn,
                          ticks=ticks_dp,
                          cax=cax, orientation='horizontal', extend='both')

    elif ax == ax4:
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          format=utils.format_fn,
                          ticks=ticks_de,
                          cax=cax, orientation='horizontal', extend='both')

    else:
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          format=utils.format_fn,
                          cax=cax, orientation='horizontal', extend='both')

    cb.ax.tick_params(labelsize=tick_fontsize)

######################### TIMESERIES #########################
# DY TIMESERIES
ax5.plot(years_plot, df_dy_top[year_str].mean(), color=color_1, label='Top10')
ax5.plot(years_plot, df_dy_bot[year_str].mean(), color=color_1, linestyle='--', label='Bot50')
ax5.set_ylim(ylim_bottom_dy, ylim_top_dy)
ax5.set_title(r'(e) $\overline{dy}$', fontsize=label_fontsize)
ax5.set_ylabel(r'$\overline{dy}$ [%]', fontsize=label_fontsize)
ax5.tick_params(labelsize=tick_fontsize)

ax5.legend(fontsize=tick_fontsize)
ax5.grid(True)

# ----------------- ADD CONFIDENCE INTERVAL (DY)-----------------
year_2019 = years_plot[-1]
# ---TOP10
dy_2019_top = df_dy_top["2019"]
ci_low_top = dy_2019_top.quantile(quantile_low)
ci_high_top = dy_2019_top.quantile(quantile_high)
final_val_top = df_dy_top[year_str].mean()[-1]
# ---BOT50
dy_2019_bot = df_dy_bot["2019"]
ci_low_bot = dy_2019_bot.quantile(quantile_low)
ci_high_bot = dy_2019_bot.quantile(quantile_high)
final_val_bot = df_dy_bot[year_str].mean()[-1]

# Plot CI
ax5.vlines(year_2019 + 0.3, ci_low_top, ci_high_top,
           color=color_1, linestyle='-', linewidth=0.8)
ax5.plot(year_2019 + 0.3, final_val_top, 'o',
         color=color_1, markersize=1)
ax5.vlines(year_2019 + 0.3, ci_low_bot, ci_high_bot,
           color=color_1, linestyle='-', linewidth=0.8)
ax5.plot(year_2019 + 0.3, final_val_bot, 'o',
         color=color_1, markersize=1)

# ----------------------------------------------------------


# DP TIMESERIES
ax6.plot(years_plot, df_dp_top[year_str].sum() / 1e6, color=color_1, label='Top10')
ax6.plot(years_plot, df_dp_bot[year_str].sum() / 1e6, color=color_1, linestyle='--', label='Bot50')
ax6.set_ylim(ylim_bottom_dp, ylim_top_dp)
ax6.set_title('(f) dp', fontsize=label_fontsize)
ax6.set_ylabel('dp [Mt]', fontsize=label_fontsize)
ax6.tick_params(labelsize=tick_fontsize)
ax6.grid(True)

degdp_top_val = df_degdp_top[year_str].mean()
degdp_top_val = np.nan_to_num(degdp_top_val, nan=0)
degdp_bot_val = df_degdp_bot[year_str].mean()
degdp_bot_val = np.nan_to_num(degdp_bot_val, nan=0)

# DEGDP TIMESERIES
ax7.plot(years_plot, degdp_top_val, color=color_2, label='Top10')
ax7.plot(years_plot, degdp_bot_val, color=color_2, linestyle='--', label='Bot50')
ax7.set_ylim(ylim_bottom_degdp, ylim_top_degdp)
ax7.set_title(r'(g) $\overline{degdp}$', fontsize=label_fontsize)
ax7.set_ylabel(r'$\overline{degdp}$ [%GDP]', fontsize=label_fontsize)
ax7.tick_params(labelsize=tick_fontsize)
ax7.legend(fontsize=tick_fontsize)
ax7.grid(True)

# ----------------- ADD CONFIDENCE INTERVAL (DEGDP)-----------------
year_2019 = years_plot[-1]
# ---TOP10
degdp_2019_top = df_degdp_top["2019"]
ci_low_top = degdp_2019_top.quantile(quantile_low)
ci_high_top = degdp_2019_top.quantile(quantile_high)
final_val_top = df_degdp_top[year_str].mean()[-1]
# ---BOT50
degdp_2019_bot = df_degdp_bot["2019"]
ci_low_bot = degdp_2019_bot.quantile(quantile_low)
ci_high_bot = degdp_2019_bot.quantile(quantile_high)
final_val_bot = df_degdp_bot[year_str].mean()[-1]

# Plot CI
ax7.vlines(year_2019 + 0.55, ci_low_top, ci_high_top,
           color=color_2, linestyle='-', linewidth=0.8)
ax7.plot(year_2019 + 0.55, final_val_top, 'o',
         color=color_2, markersize=1)
ax7.vlines(year_2019 + 0.1, ci_low_bot, ci_high_bot,
           color=color_2, linestyle='-', linewidth=0.8)
ax7.plot(year_2019 + 0.1, final_val_bot, 'o',
         color=color_2, markersize=1)

# ----------------------------------------------------------

# DE TIMESERIES
ax8.plot(years_plot, df_de_top[year_str].sum() / 1e9, color=color_2, label='Top10')
ax8.plot(years_plot, df_de_bot[year_str].sum() / 1e9, color=color_2, linestyle='--', label='Bot50')
ax8.set_ylim(ylim_bottom_de, ylim_top_de)
ax8.set_title('(h) de', fontsize=label_fontsize)
ax8.set_ylabel('de [Billion US$]', fontsize=label_fontsize)
ax8.tick_params(labelsize=tick_fontsize)
ax8.grid(True)

print("TOP10 dy:", df_dy_top[year_str].mean().tolist()[-1])
print("BOT50 dy:", df_dy_bot[year_str].mean().tolist()[-1])
print("TOP10 dp:", df_dp_top[year_str].sum().tolist()[-1] / 1e6)
print("BOT50 dp:", df_dp_bot[year_str].sum().tolist()[-1] / 1e6)
print("---------")
print("TOP10 degdp:", df_degdp_top[year_str].mean().tolist()[-1])
print("BOT50 degdp:", df_degdp_bot[year_str].mean().tolist()[-1])
print("TOP10 de:", df_de_top[year_str].sum().tolist()[-1] / 1e9)
print("BOT50 de:", df_de_bot[year_str].sum().tolist()[-1] / 1e9)

print("-------------")
sum_dp_top10 = sum(df_dp_top[year_str].sum().tolist()) / 1e6
sum_dp_bot50 = sum(df_dp_bot[year_str].sum().tolist()) / 1e6
sum_de_top10 = sum(df_de_top[year_str].sum().tolist()) / 1e9
sum_de_bot50 = sum(df_de_bot[year_str].sum().tolist()) / 1e9

print("TOP10 dp SUM:", sum_dp_top10)
print("BOT50 dp SUM:", sum_dp_bot50)
print("TOP10 de SUM:", sum_de_top10)
print("BOT50 de SUM:", sum_de_bot50)

# Add x-label to bottom plots
ax7.set_xlabel('Year', fontsize=label_fontsize)
ax8.set_xlabel('Year', fontsize=label_fontsize)

# Adjust individual subplot positions
# First row (maps)
ax1.set_position([0.03, 0.76, 0.5, 0.2])  # [left, bottom, width, height]
ax2.set_position([0.5, 0.76, 0.5, 0.2])

# Second row (maps)
ax3.set_position([0.03, 0.46, 0.5, 0.2])  # [left, bottom, width, height]
ax4.set_position([0.5, 0.46, 0.5, 0.2])

# Third row (timeseries)
ax5.set_position([0.12, 0.24, 0.38, 0.13])  # [left, bottom, width, height]
ax6.set_position([0.6, 0.24, 0.38, 0.13])

# Fourth row (timeseries)
ax7.set_position([0.12, 0.05, 0.38, 0.13])  # [left, bottom, width, height]
ax8.set_position([0.6, 0.05, 0.38, 0.13])

plt.show()
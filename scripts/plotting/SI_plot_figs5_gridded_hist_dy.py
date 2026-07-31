# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
import os
import sys
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('../')
import utils


# PARAMETERS
crops = ["maize","wheat", "soy"]
pred_str = "t3s3"
estring = "isimip3a"
spei_month = "03"
start_year_avg = 2007
end_year_avg = 2019

start_year_hist = 1974
end_year_hist = 2004
start_year_fut = 1990
end_year_fut = 2019

######################## PLOTTING PARAMETERS ########################
vmin = -10
vmax = 0
cmap = "YlOrRd_r"

title_fontsize = 15
label_fontsize = 14
tick_fontsize = 13

subplt_labels = utils.generate_alphabet_list(3, "lower")
#####################################################################

root_dir = f"../../data"

"""
PLOT MAPS
"""

proj = ccrs.Robinson()

fig, axes = plt.subplots(
    3, 1,
    figsize=(7, 10),
    subplot_kw={"projection": proj}
)
axes = axes.flatten()

datasets = {}

for crop in crops:
    nc_file = f"{root_dir}/historical/gridded_dy/{crop}/dy_{estring}_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.nc"
    ds = xr.open_dataset(nc_file)
    dy_avg = ds["dy"].sel(year=slice(start_year_avg, end_year_avg)).mean(dim="year")
    datasets[crop] = dy_avg


for ax, crop, label in zip(axes, crops, subplt_labels):
    dy_avg = datasets[crop]

    lon = dy_avg["lon"].values
    lat = dy_avg["lat"].values
    data = dy_avg.values

    title = crop.capitalize()

    if crop == "soy":
        title = "Soybeans"

    im = ax.pcolormesh(
        lon, lat, data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto"
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.set_global()
    ax.set_title(title, fontsize=title_fontsize)
    ax.text(-0.01, 1.04, label, transform=ax.transAxes, fontweight='bold', fontsize=title_fontsize)

#COLORBAR
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cax = fig.add_axes([0.86, 0.22, 0.01, 0.5])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cax, orientation="vertical", extend="both")
cbar.set_label("Yield change [%]", fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)


fig.subplots_adjust(left=0.02,
                    bottom=0.02,
                    right=0.85,
                    top=0.95,
                    wspace=0.05,
                    hspace=0.2)

plt.show()
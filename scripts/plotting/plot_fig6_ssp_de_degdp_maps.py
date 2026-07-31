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
pred_str_irr = f"{pred_str}_irr"
spei_month = "03"

######################## PLOTTING PARAMETERS ########################
# Maps parameters
cmap_de = utils.get_truncated_cmap("RdPu_r")
cmap_degdp = utils.get_truncated_cmap("YlGnBu_r")
vmin_degdp = -0.04
vmax_degdp = 0.0
vmin_de = -100
vmax_de = -0.1
ticks_de = [-0.1, -1, -10, -100]
start_year_map = 2080
end_year_map = 2100
year_map_degdp = [2080,2085,2090,2095,2100]
year_map_degdp_str = [str(x) for x in year_map_degdp]
year_map_de_str = [str(x) for x in range(start_year_map, end_year_map+1)]
label_de = "Economic loss [B US$]"
label_degdp = "GDP loss [%]"

# Timeseries parameters
ylim_top_degdp = 0.01
ylim_bottom_degdp = -0.04
ylim_top_de = 20
ylim_bottom_de = -120
start_year_plot = 2030
end_year_plot = 2100

window_size = 4
smooth_method = "pandas"
np_mode = "full"

ipcc_colors = {"ldc": "crimson", "developing": "goldenrod", "developed": "dodgerblue"}  # ldc,developing,developed
ipcc_labels = {"ldc": "LDC", "developing": "Developing", "developed": "Developed"}

label_alphabets = utils.generate_alphabet_list(12, option="lower")
label_alphabets = [x for x in label_alphabets]
##################################################################

ar6_region = "region_ar6_dev"
regions = ["ldc", "developing", "developed"]

# Time ranges
start_year = 2020
end_year = 2100

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

start_year_hist = 1974
end_year_hist = 2004
start_year_fut = 2020
end_year_fut = 2100

root_dir = '../../data'
country_shape_file = f"{root_dir}/resources/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

ssp_gdp_weights = utils.load_ssp_gdp_weights(root_dir)

degdp_all_crops = []
de_all_crops = []
degdp_all_crops_irr = []
de_all_crops_irr = []

for crop in crops:
    degdp_file = f"{root_dir}/ssp/linregress_outputs/{crop}/degdp_isimip3b_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    de_file = f"{root_dir}/ssp/linregress_outputs/{crop}/de_isimip3b_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    degdp_file_irr = f"{root_dir}/ssp/linregress_outputs/{crop}/degdp_isimip3b_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"
    de_file_irr = f"{root_dir}/ssp/linregress_outputs/{crop}/de_isimip3b_{pred_str_irr}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"

    df_degdp = pd.read_csv(degdp_file)
    df_de = pd.read_csv(de_file)
    df_degdp_irr = pd.read_csv(degdp_file_irr)
    df_de_irr = pd.read_csv(de_file_irr)

    # Add crop column
    df_degdp['crop'] = crop
    df_de['crop'] = crop
    df_degdp_irr['crop'] = crop
    df_de_irr['crop'] = crop

    degdp_all_crops.append(df_degdp)
    de_all_crops.append(df_de)
    degdp_all_crops_irr.append(df_degdp_irr)
    de_all_crops_irr.append(df_de_irr)

# Combine all crops
df_degdp = pd.concat(degdp_all_crops, ignore_index=True)
df_de = pd.concat(de_all_crops, ignore_index=True)
df_degdp_irr = pd.concat(degdp_all_crops_irr, ignore_index=True)
df_de_irr = pd.concat(de_all_crops_irr, ignore_index=True)

num_cntry_degdp = df_degdp["country"].nunique()
num_cntry_de = df_de["country"].nunique()

cols_de = ["country", "crop"] + ["ssp"] + ["esm"] + year_str_all_de
cols_degdp = ["country", "crop"] + ["ssp"] + ["esm"] + year_str_all_degdp
degdp_all = df_degdp[cols_degdp]
de_all = df_de[cols_de]
degdp_all_irr = df_degdp_irr[cols_degdp]
de_all_irr = df_de_irr[cols_de]

degdp_long = degdp_all.melt(id_vars=['country', 'crop', 'ssp', 'esm'], var_name='year', value_name='value')
de_long = de_all.melt(id_vars=['country', 'crop', 'ssp', 'esm'], var_name='year', value_name='value')
degdp_long_irr = degdp_all_irr.melt(id_vars=['country', 'crop', 'ssp', 'esm'], var_name='year', value_name='value')
de_long_irr = de_all_irr.melt(id_vars=['country', 'crop', 'ssp', 'esm'], var_name='year', value_name='value')

# Get mean over ESMs first
degdp_long = degdp_long.groupby(['country', 'crop', 'ssp', 'year'])['value'].mean().reset_index()
de_long = de_long.groupby(['country', 'crop', 'ssp', 'year'])['value'].mean().reset_index()
degdp_long_irr = degdp_long_irr.groupby(['country', 'crop', 'ssp', 'year'])['value'].mean().reset_index()
de_long_irr = de_long_irr.groupby(['country', 'crop', 'ssp', 'year'])['value'].mean().reset_index()

# Get mean/sum over crops
degdp_long = degdp_long.groupby(['country', 'ssp', 'year'])['value'].mean().reset_index()
de_long = de_long.groupby(['country', 'ssp', 'year'])['value'].sum().reset_index()
degdp_long_irr = degdp_long_irr.groupby(['country', 'ssp', 'year'])['value'].mean().reset_index()
de_long_irr = de_long_irr.groupby(['country', 'ssp', 'year'])['value'].sum().reset_index()

de_global = de_long.groupby(['ssp', 'year'])['value'].sum().reset_index()

print("\nValue ranges after combining crops:")
print(f"de_global min: {de_global['value'].min() / 1e9:.2f}, max: {de_global['value'].max() / 1e9:.2f}")

"""
GET IPCC REGION DATA
"""
print ("--------------------")
print("Getting IPCC region:", ar6_region)
df_ipcc = utils.get_ipcc_region_df()
df_ipcc = df_ipcc[["Country", ar6_region]]
df_ipcc = df_ipcc.rename(columns={"Country": "country"})

# Merge with IPCC regions
degdp_ipcc = degdp_all.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
de_ipcc = de_all.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
degdp_ipcc_irr = degdp_all_irr.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
de_ipcc_irr = de_all_irr.merge(df_ipcc[["country", ar6_region]], on="country", how="left")

# Calculate regional means/sums for timeseries
degdp_ipcc_melted = degdp_ipcc.melt(id_vars=['country', 'ssp', 'esm', 'crop', 'region_ar6_dev'], var_name='year',
                                    value_name='value')
de_ipcc_melted = de_ipcc.melt(id_vars=['country', 'ssp', 'esm', 'crop', 'region_ar6_dev'], var_name='year',
                              value_name='value')
degdp_ipcc_melted_irr = degdp_ipcc_irr.melt(id_vars=['country', 'ssp', 'esm', 'crop', 'region_ar6_dev'], var_name='year',
                                             value_name='value')
de_ipcc_melted_irr = de_ipcc_irr.melt(id_vars=['country', 'ssp', 'esm', 'crop', 'region_ar6_dev'], var_name='year',
                                      value_name='value')

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
    ci = t.interval(confidence=0.95, df=n - 1, loc=mean, scale=sem)
    return mean, float((ci[1] - ci[0]) / 2)  # Return mean and half the CI width

# Calculate statistics for degdp and de
degdp_stats = []
de_stats = []

for year in years_degdp:
    year_str = str(year)
    for ssp in ssps:
        # For degdp — SSP-projected GDP-weighted mean per ESM
        degdp_data_ssp = degdp_all[degdp_all['ssp'] == ssp]
        try:
            gdp_this = ssp_gdp_weights.xs(ssp, level='ssp')[year_str]
        except KeyError:
            gdp_this = pd.Series(dtype=float)
        degdp_esm_means = {}
        for esm in degdp_data_ssp['esm'].unique():
            degdp_esm = degdp_data_ssp[degdp_data_ssp['esm'] == esm].groupby('country')[year_str].mean()
            common = degdp_esm.index.intersection(gdp_this.index)
            if len(common) == 0:
                degdp_esm_means[esm] = 0.0
                continue
            w = gdp_this.loc[common]
            w = w / w.sum()
            degdp_esm_means[esm] = float((w * degdp_esm.loc[common]).sum())
        degdp_esm_means = pd.Series(degdp_esm_means)
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


def compute_gdp_weighted_per_esm(degdp_sub, ssp, year_str, ssp_gdp_weights):
    """
    SSP-projected GDP-weighted degdp for each ESM.
    """
    try:
        gdp_this = ssp_gdp_weights.xs(ssp, level='ssp')[year_str]
    except KeyError:
        return pd.Series({esm: 0.0 for esm in degdp_sub['esm'].unique()})
    result = {}
    for esm in degdp_sub['esm'].unique():
        degdp = degdp_sub[degdp_sub['esm'] == esm].groupby('country')['value'].mean()
        common = degdp.index.intersection(gdp_this.index)
        if len(common) == 0:
            result[esm] = 0.0
            continue
        w = gdp_this.loc[common]
        w = w / w.sum()
        result[esm] = float((w * degdp.loc[common]).sum())
    return pd.Series(result)


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

# Load and project world shapefile once (reused for all 4 map iterations)
world_proj = gpd.read_file(country_shape_file)
world_proj = world_proj.to_crs(ccrs.Robinson().proj4_init)

######################## DEGDP MAPS ########################
for index, (scenario, reference) in enumerate(ssp_diffs):

    ax = fig.add_subplot(gs[0, index], projection=ccrs.Robinson())

    if index == 0:
        ax.set_position([0.12, 0.76, 0.4, 0.2])  # [left, bottom, width, height]
    if index == 1:
        ax.set_position([0.57, 0.76, 0.4, 0.2])  # [left, bottom, width, height]

    # Add alphabet label
    ax.text(-0.03, 1.0, label_alphabets[index], transform=ax.transAxes, fontweight='bold', fontsize=label_fontsize+1)

    # Prepare data
    world = world_proj.copy()

    # Get data for both scenarios
    degdp_scenario = degdp_long[(degdp_long["ssp"] == scenario) & (degdp_long["year"].isin(year_map_degdp_str))]
    degdp_reference = degdp_long[(degdp_long["ssp"] == reference) & (degdp_long["year"].isin(year_map_degdp_str))]

    # Calculate mean for each country
    degdp_scenario = degdp_scenario.groupby('country')['value'].mean().reset_index()
    degdp_reference = degdp_reference.groupby('country')['value'].mean().reset_index()

    # Merge and calculate difference
    degdp_diff = pd.merge(degdp_scenario, degdp_reference, on='country', suffixes=('_scenario', '_reference'))
    degdp_diff['value'] = degdp_diff['value_scenario'] - degdp_diff['value_reference']

    world = world.merge(degdp_diff[['country', 'value']], how='left', left_on='ISO_A3', right_on='country')
    world.boundary.plot(ax=ax, linewidth=0.2, edgecolor='gray')

    # Plot
    world.plot(ax=ax, column='value', cmap=cmap_degdp,
               vmin=vmin_degdp, vmax=vmax_degdp,
               missing_kwds={"color": "white"})

    ax.set_global()
    ax.gridlines(alpha=0.2)

# Add colorbar for degdp maps
cax_degdp = fig.add_axes([0.27, 0.77, 0.5, 0.005]) #left,bottom,width,height
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin_degdp, vmax=vmax_degdp), cmap=cmap_degdp)
cbar = plt.colorbar(sm, cax=cax_degdp, orientation='horizontal', extend='both')
cbar.set_label(label_degdp, fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)

######################## DE MAPS ########################
for index, (scenario, reference) in enumerate(ssp_diffs):
    ax = fig.add_subplot(gs[1, index], projection=ccrs.Robinson())

    if index == 0:
        ax.set_position([0.12, 0.52, 0.4, 0.2])  # [left, bottom, width, height]
    if index == 1:
        ax.set_position([0.57, 0.52, 0.4, 0.2])  # [left, bottom, width, height]

    # Add alphabet label
    ax.text(-0.03, 1.0, label_alphabets[index+2], transform=ax.transAxes, fontweight='bold', fontsize=label_fontsize + 1)

    # Prepare data
    world = world_proj.copy()

    # Get data for both scenarios
    de_scenario = de_long[(de_long["ssp"] == scenario) & (de_long["year"].isin(year_map_de_str))]
    de_reference = de_long[(de_long["ssp"] == reference) & (de_long["year"].isin(year_map_de_str))]

    # Calculate statistics for each country
    de_scenario = de_scenario.groupby('country')['value'].sum().reset_index()
    de_reference = de_reference.groupby('country')['value'].sum().reset_index()

    # Merge and calculate difference
    de_diff = pd.merge(de_scenario, de_reference, on='country', suffixes=('_scenario', '_reference'))
    de_diff['value'] = (de_diff['value_scenario'] - de_diff['value_reference']) / 1e9  # Convert to Mt

    world = world.merge(de_diff[['country', 'value']], how='left', left_on='ISO_A3', right_on='country')
    world.boundary.plot(ax=ax, linewidth=0.2, edgecolor='gray')

    # Plot
    norm = colors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=vmin_de, vmax=vmax_de, base=10)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_de)
    world.plot(ax=ax, column='value', cmap=cmap_de,
               norm=norm,
               vmin=vmin_de, vmax=vmax_de,
               missing_kwds={"color": "white"})

    ax.set_global()
    ax.gridlines(alpha=0.2)

# Add colorbar for de maps
cax_de = fig.add_axes([0.27, 0.53, 0.5, 0.005])  # left,bottom,width,height
cbar = plt.colorbar(sm, cax=cax_de, orientation='horizontal', extend='both'
                    , format=utils.format_fn
                    ,ticks=ticks_de
                    )
cbar.set_label(label_de, fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)

degdp_global_grp     = {k: v for k, v in degdp_ipcc_melted.groupby(['ssp', 'year'])}
de_global_grp        = {k: v for k, v in de_ipcc_melted.groupby(['ssp', 'year'])}
degdp_global_grp_irr = {k: v for k, v in degdp_ipcc_melted_irr.groupby(['ssp', 'year'])}
de_global_grp_irr    = {k: v for k, v in de_ipcc_melted_irr.groupby(['ssp', 'year'])}

degdp_reg_grp     = {k: v for k, v in degdp_ipcc_melted.groupby(['region_ar6_dev', 'ssp', 'year'])}
de_reg_grp        = {k: v for k, v in de_ipcc_melted.groupby(['region_ar6_dev', 'ssp', 'year'])}
degdp_reg_grp_irr = {k: v for k, v in degdp_ipcc_melted_irr.groupby(['region_ar6_dev', 'ssp', 'year'])}
de_reg_grp_irr    = {k: v for k, v in de_ipcc_melted_irr.groupby(['region_ar6_dev', 'ssp', 'year'])}

######################## DEGDP TIMESERIES ########################
print()
for index, (scenario, reference) in enumerate(ssp_diffs):
    ax = fig.add_subplot(gs[2, index])

    if index == 0:
        ax.set_position([0.12, 0.31, 0.4, 0.13])  # [left, bottom, width, height]
    if index == 1:
        ax.set_position([0.57, 0.31, 0.4, 0.13])  # [left, bottom, width, height]

    # Add alphabet label
    ax.text(-0.08, 1.1, label_alphabets[index+4], transform=ax.transAxes, fontsize=label_fontsize+1, fontweight='bold')

    # --------------------Plot GLOBAL
    # Calculate GDP-weighted degdp and CI for each year
    global_stats = []
    for year in years_degdp:
        year_str = str(year)
        degdp_year_ref  = degdp_global_grp[(reference, year_str)]
        degdp_year_scen = degdp_global_grp[(scenario,  year_str)]

        esm_stats_ref  = compute_gdp_weighted_per_esm(degdp_year_ref,  reference, year_str, ssp_gdp_weights)
        esm_stats_scen = compute_gdp_weighted_per_esm(degdp_year_scen, scenario,  year_str, ssp_gdp_weights)

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
    line = ax.plot(years_plot_degdp, global_diff_smoothed[plot_start_idx_degdp:plot_end_idx_degdp],
                   color="black", linestyle='solid', linewidth=1.2,
                   label='Global')

    # Plot irr dashed line (global)
    global_stats_irr = []
    for year in years_degdp:
        year_str = str(year)
        degdp_year_ref_irr  = degdp_global_grp_irr[(reference, year_str)]
        degdp_year_scen_irr = degdp_global_grp_irr[(scenario,  year_str)]
        esm_diff_irr = compute_gdp_weighted_per_esm(degdp_year_scen_irr, scenario,  year_str, ssp_gdp_weights) - compute_gdp_weighted_per_esm(degdp_year_ref_irr,  reference, year_str, ssp_gdp_weights)
        mean_diff_irr, _ = calculate_95_ci(esm_diff_irr)
        global_stats_irr.append({'year': year, 'mean': mean_diff_irr})
    global_stats_irr = pd.DataFrame(global_stats_irr)
    global_diff_smoothed_irr = utils.get_smoothed_values(global_stats_irr['mean'].values,
                                                         window_size=window_size, method=smooth_method, np_mode=np_mode)
    ax.plot(years_plot_degdp, global_diff_smoothed_irr[plot_start_idx_degdp:plot_end_idx_degdp],
            color="black", linestyle='--', linewidth=1.2, alpha=0.6)

    print("------------------")
    print(f"degdp diff for GLOBAL, {scenario}-{reference}, {year}: {global_diff_smoothed[plot_end_idx_degdp - 1]:.4f}")

    # Add shading for global confidence interval
    ax.fill_between(years_plot_degdp,
                    ci_lower_smoothed[plot_start_idx_degdp:plot_end_idx_degdp],
                    ci_upper_smoothed[plot_start_idx_degdp:plot_end_idx_degdp],
                    color="black", alpha=0.05)

    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.8)

    # ---------------------Plot each region
    for region in regions:
        # Calculate GDP-weighted degdp and CI for each year
        region_stats = []
        for year in years_degdp:
            year_str = str(year)
            degdp_year_ref  = degdp_reg_grp[(region, reference, year_str)]
            degdp_year_scen = degdp_reg_grp[(region, scenario,  year_str)]

            esm_stats_ref  = compute_gdp_weighted_per_esm(degdp_year_ref,  reference, year_str, ssp_gdp_weights)
            esm_stats_scen = compute_gdp_weighted_per_esm(degdp_year_scen, scenario,  year_str, ssp_gdp_weights)

            # Calculate the difference between esm_stats_scen and esm_stats_ref
            esm_diff = esm_stats_scen - esm_stats_ref

            # Calculate the 95% confidence interval for the differences
            mean_diff, ci_diff = calculate_95_ci(esm_diff)

            region_stats.append({'year': year, 'mean': mean_diff, 'ci': ci_diff})
            # -----------------

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
        line = ax.plot(years_plot_degdp, region_diff_smoothed[plot_start_idx_degdp:plot_end_idx_degdp],
                       color=ipcc_colors[region], linestyle='solid', linewidth=0.8,
                       label=ipcc_labels[region])

        # Plot irr dashed line (region)
        region_stats_irr = []
        for year in years_degdp:
            year_str = str(year)
            degdp_year_ref_irr  = degdp_reg_grp_irr[(region, reference, year_str)]
            degdp_year_scen_irr = degdp_reg_grp_irr[(region, scenario,  year_str)]
            esm_diff_irr = compute_gdp_weighted_per_esm(degdp_year_scen_irr, scenario,  year_str, ssp_gdp_weights) - compute_gdp_weighted_per_esm(degdp_year_ref_irr,  reference, year_str, ssp_gdp_weights)
            mean_diff_irr, _ = calculate_95_ci(esm_diff_irr)
            region_stats_irr.append({'year': year, 'mean': mean_diff_irr})
        region_stats_irr = pd.DataFrame(region_stats_irr)
        region_diff_smoothed_irr = utils.get_smoothed_values(region_stats_irr['mean'].values,
                                                             window_size=window_size, method=smooth_method, np_mode=np_mode)
        ax.plot(years_plot_degdp, region_diff_smoothed_irr[plot_start_idx_degdp:plot_end_idx_degdp],
                color=ipcc_colors[region], linestyle='--', linewidth=0.8, alpha=0.6)

        print(f"degdp diff for {region}, {scenario}-{reference}, {year}: {region_diff_smoothed[plot_end_idx_degdp - 1]:.4f}")

        # Add shading for global confidence interval
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

    if index == 0:
        ax.set_ylabel(label_degdp, fontsize=label_fontsize)
    else:
        ax.set_yticklabels([])

    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=tick_fontsize)

######################## DE TIMESERIES ########################
for index, (scenario, reference) in enumerate(ssp_diffs):
    ax = fig.add_subplot(gs[3, index])

    if index == 0:
        ax.set_position([0.12, 0.11, 0.4, 0.13])  # [left, bottom, width, height]
    if index == 1:
        ax.set_position([0.57, 0.11, 0.4, 0.13])  # [left, bottom, width, height]

    # Add alphabet label
    ax.text(-0.08, 1.1, label_alphabets[index + 6], transform=ax.transAxes, fontsize=label_fontsize + 1,fontweight='bold')

    # --------------------Plot GLOBAL
    # Calculate mean/sum and CI for each year
    global_stats = []
    for year in years_de:
        year_str = str(year)
        year_data_ref  = de_global_grp[(reference, year_str)]
        year_data_scen = de_global_grp[(scenario,  year_str)]

        esm_stats_ref = year_data_ref.groupby('esm')['value'].sum()
        esm_stats_scen = year_data_scen.groupby('esm')['value'].sum()

        # Calculate the difference between esm_stats_scen and esm_stats_ref
        esm_diff = esm_stats_scen - esm_stats_ref

        # Calculate the 95% confidence interval for the differences
        mean_diff, ci_diff = calculate_95_ci(esm_diff)
        mean_diff = mean_diff / 1e9
        ci_diff = ci_diff / 1e9

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
    line = ax.plot(years_plot_de, global_diff_smoothed[plot_start_idx_de:plot_end_idx_de],
                   color="black", linestyle='solid', linewidth=1.2,
                   label='Global')

    # Plot irr dashed line (global)
    global_stats_irr = []
    for year in years_de:
        year_str = str(year)
        year_data_ref_irr  = de_global_grp_irr[(reference, year_str)]
        year_data_scen_irr = de_global_grp_irr[(scenario,  year_str)]
        esm_diff_irr = year_data_scen_irr.groupby('esm')['value'].sum() - year_data_ref_irr.groupby('esm')['value'].sum()
        mean_diff_irr, _ = calculate_95_ci(esm_diff_irr)
        mean_diff_irr = mean_diff_irr / 1e9
        global_stats_irr.append({'year': year, 'mean': mean_diff_irr})
    global_stats_irr = pd.DataFrame(global_stats_irr)
    global_diff_smoothed_irr = utils.get_smoothed_values(global_stats_irr['mean'].values,
                                                         window_size=window_size, method=smooth_method, np_mode=np_mode)
    ax.plot(years_plot_de, global_diff_smoothed_irr[plot_start_idx_de:plot_end_idx_de],
            color="black", linestyle='--', linewidth=1.2, alpha=0.6)

    print("------------------")
    print(f"de diff for GLOBAL, {scenario}-{reference}, {year}: {global_diff_smoothed[plot_end_idx_de - 1]:.4f}")

    # Add shading for global confidence interval
    ax.fill_between(years_plot_de,
                    ci_lower_smoothed[plot_start_idx_de:plot_end_idx_de],
                    ci_upper_smoothed[plot_start_idx_de:plot_end_idx_de],
                    color="black", alpha=0.05)

    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.8)

    # Plot each region
    for region in regions:
        # Calculate mean/sum and CI for each year
        region_stats = []
        for year in years_de:
            year_str = str(year)
            year_data_ref  = de_reg_grp[(region, reference, year_str)]
            year_data_scen = de_reg_grp[(region, scenario,  year_str)]

            esm_stats_ref = year_data_ref.groupby('esm')['value'].sum()
            esm_stats_scen = year_data_scen.groupby('esm')['value'].sum()

            # Calculate the difference between esm_stats_scen and esm_stats_ref
            esm_diff = esm_stats_scen - esm_stats_ref

            # Calculate the 95% confidence interval for the differences
            mean_diff, ci_diff = calculate_95_ci(esm_diff)
            mean_diff = mean_diff / 1e9
            ci_diff = ci_diff / 1e9

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
        line = ax.plot(years_plot_de, region_diff_smoothed[plot_start_idx_de:plot_end_idx_de],
                       color=ipcc_colors[region], linestyle='solid', linewidth=0.8,
                       label=ipcc_labels[region])

        # Plot irr dashed line (region)
        region_stats_irr = []
        for year in years_de:
            year_str = str(year)
            yd_ref_irr  = de_reg_grp_irr[(region, reference, year_str)]
            yd_scen_irr = de_reg_grp_irr[(region, scenario,  year_str)]
            esm_diff_irr = yd_scen_irr.groupby('esm')['value'].sum() - yd_ref_irr.groupby('esm')['value'].sum()
            mean_diff_irr, _ = calculate_95_ci(esm_diff_irr)
            mean_diff_irr = mean_diff_irr / 1e9
            region_stats_irr.append({'year': year, 'mean': mean_diff_irr})
        region_stats_irr = pd.DataFrame(region_stats_irr)
        region_diff_smoothed_irr = utils.get_smoothed_values(region_stats_irr['mean'].values,
                                                             window_size=window_size, method=smooth_method, np_mode=np_mode)
        ax.plot(years_plot_de, region_diff_smoothed_irr[plot_start_idx_de:plot_end_idx_de],
                color=ipcc_colors[region], linestyle='--', linewidth=0.8, alpha=0.6)

        print(f"de diff for {region}, {scenario}-{reference}, {year}: {region_diff_smoothed[plot_end_idx_de - 1]:.4f}")

        # Add shading for global confidence interval
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
        ax.set_ylabel(label_de, fontsize=label_fontsize)
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
print(f"Top 10 countries with largest damage reduction in {end_year_plot} (SSP126 vs SSP370)")
print("="*50)

# For dy (yield damage)
degdp_370 = degdp_long[(degdp_long['ssp'] == 'ssp370') & (degdp_long['year'] == f'{end_year_plot}')].set_index('country')['value']
degdp_126 = degdp_long[(degdp_long['ssp'] == 'ssp126') & (degdp_long['year'] == f'{end_year_plot}')].set_index('country')['value']
degdp_diff = degdp_126 - degdp_370  # Positive values mean less damage in SSP126

print("\nDEGDP Damage Reduction (degdp) [%]:")
print("-" * 30)
print(degdp_diff.sort_values(ascending=False).head(10).to_frame('degdp_diff').round(2))

# For dp (production damage)
de_370 = de_long[(de_long['ssp'] == 'ssp370') & (de_long['year'] == f'{end_year_plot}')].set_index('country')['value']
de_126 = de_long[(de_long['ssp'] == 'ssp126') & (de_long['year'] == f'{end_year_plot}')].set_index('country')['value']
de_diff = (de_126 - de_370) / 1e9 # Convert to Mt, positive values mean less damage in SSP126

print("\nDE Damage Reduction (de) [Billion US$]:")
print("-" * 30)
print(de_diff.sort_values(ascending=False).head(10).to_frame('de_diff').round(2))

"""
COMPARISON TABLE: RAINFED vs IRR
"""
# Merge with IPCC regions for regional breakdown
degdp_long_ipcc = degdp_long.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
de_long_ipcc = de_long.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
degdp_long_irr_ipcc = degdp_long_irr.merge(df_ipcc[["country", ar6_region]], on="country", how="left")
de_long_irr_ipcc = de_long_irr.merge(df_ipcc[["country", ar6_region]], on="country", how="left")

# Filter for the averaging period (2080-2100)
degdp_period = degdp_long_ipcc[degdp_long_ipcc['year'].isin(year_map_degdp_str)]
de_period = de_long_ipcc[de_long_ipcc['year'].isin(year_map_de_str)]
degdp_period_irr = degdp_long_irr_ipcc[degdp_long_irr_ipcc['year'].isin(year_map_degdp_str)]
de_period_irr = de_long_irr_ipcc[de_long_irr_ipcc['year'].isin(year_map_de_str)]

de_long_ipcc_dg = de_long_ipcc[de_long_ipcc['year'].isin(year_map_degdp_str)]
de_long_irr_ipcc_dg = de_long_irr_ipcc[de_long_irr_ipcc['year'].isin(year_map_degdp_str)]
degdp_long_ipcc_dg = degdp_long_ipcc[degdp_long_ipcc['year'].isin(year_map_degdp_str)]
degdp_long_irr_ipcc_dg = degdp_long_irr_ipcc[degdp_long_irr_ipcc['year'].isin(year_map_degdp_str)]

for scenario, reference in ssp_diffs:
    print()
    print("=" * 80)
    print(f"COMPARISON TABLE: RAINFED vs IRR — {scenario.upper()}–{reference.upper()} ({start_year_map}-{end_year_map})")
    print("=" * 80)

    def ssp_diff_stats(df, ssp_scen, ssp_ref, groupby_col=None, agg='mean'):
        scen = df[df['ssp'] == ssp_scen]
        ref  = df[df['ssp'] == ssp_ref]
        if groupby_col:
            return scen.groupby(groupby_col)['value'].agg(agg) - ref.groupby(groupby_col)['value'].agg(agg)
        else:
            return scen['value'].agg(agg) - ref['value'].agg(agg)

    def gdp_weighted_ssp_diff(degdp_df, ssp_scen, ssp_ref, region=None):
        """SSP-projected GDP-weighted degdp difference (scenario - reference) averaged over
        w_i = GDP_SSP_i(t) / Σ_j GDP_SSP_j(t); result = Σ_i (w_i × degdp_i)
        """
        def _wt(ssp):
            degdp = degdp_df[degdp_df['ssp'] == ssp]
            if region:
                degdp = degdp[degdp[ar6_region] == region]
            try:
                gdp_ssp = ssp_gdp_weights.xs(ssp, level='ssp')
            except KeyError:
                return float('nan')
            vals = []
            for yr in year_map_degdp_str:
                degdp_yr = degdp[degdp['year'] == yr].groupby('country')['value'].mean()
                common = degdp_yr.index.intersection(gdp_ssp.index)
                if len(common) == 0:
                    continue
                w = gdp_ssp.loc[common, yr]
                w = w / w.sum()
                vals.append(float((w * degdp_yr.loc[common]).sum()))
            return np.mean(vals) if vals else float('nan')
        return _wt(ssp_scen) - _wt(ssp_ref)

    # DEGDP: SSP-GDP-weighted; DE: sum then /1e9
    degdp_region_diff     = pd.Series({r: gdp_weighted_ssp_diff(degdp_long_ipcc_dg,     scenario, reference, r) for r in regions})
    degdp_region_diff_irr = pd.Series({r: gdp_weighted_ssp_diff(degdp_long_irr_ipcc_dg, scenario, reference, r) for r in regions})
    de_region_diff        = ssp_diff_stats(de_period,        scenario, reference, ar6_region, 'sum') / 1e9
    de_region_diff_irr    = ssp_diff_stats(de_period_irr,    scenario, reference, ar6_region, 'sum') / 1e9

    degdp_global_diff     = gdp_weighted_ssp_diff(degdp_long_ipcc_dg,     scenario, reference)
    degdp_global_diff_irr = gdp_weighted_ssp_diff(degdp_long_irr_ipcc_dg, scenario, reference)
    de_global_diff        = ssp_diff_stats(de_period,        scenario, reference, agg='sum') / 1e9
    de_global_diff_irr    = ssp_diff_stats(de_period_irr,    scenario, reference, agg='sum') / 1e9

    comparison_data = {
        "Metric": [
            "DEGDP MEAN LDC [%]", "DEGDP MEAN Developing [%]", "DEGDP MEAN Developed [%]", "DEGDP MEAN Global [%]",
            "DE SUM LDC [B US$]", "DE SUM Developing [B US$]", "DE SUM Developed [B US$]", "DE SUM Global [B US$]",
        ],
        "Rainfed": [
            degdp_region_diff.get('ldc', float('nan')), degdp_region_diff.get('developing', float('nan')), degdp_region_diff.get('developed', float('nan')), degdp_global_diff,
            de_region_diff.get('ldc', float('nan')), de_region_diff.get('developing', float('nan')), de_region_diff.get('developed', float('nan')), de_global_diff,
        ],
        "Irr": [
            degdp_region_diff_irr.get('ldc', float('nan')), degdp_region_diff_irr.get('developing', float('nan')), degdp_region_diff_irr.get('developed', float('nan')), degdp_global_diff_irr,
            de_region_diff_irr.get('ldc', float('nan')), de_region_diff_irr.get('developing', float('nan')), de_region_diff_irr.get('developed', float('nan')), de_global_diff_irr,
        ],
    }
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison["Diff"] = df_comparison["Irr"] - df_comparison["Rainfed"]
    df_comparison["Diff (%)"] = ((df_comparison["Irr"] - df_comparison["Rainfed"]) / df_comparison["Rainfed"].abs()) * 100
    print(df_comparison.to_string(index=False))

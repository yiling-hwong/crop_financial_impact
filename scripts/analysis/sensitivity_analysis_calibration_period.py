# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""

import sys
import numpy as np
import pandas as pd
from tabulate import tabulate

import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('..')
import utils

"""
PARAMETERS
"""
crops = ["maize", "wheat", "soy"]
pred_str = "t3s3"
spei_month = "03"
ssps = ["ssp126", "ssp245", "ssp370"]
regions = ["ldc", "developing", "developed"]
calib_periods = {
    "2007-2019": "",
    "2000-2019": "2000_2019/",
}

start_year_avg, end_year_avg = 2080, 2100
window_size = 4

root_dir = f'../../data'
file_suffix = f"isimip3b_{pred_str}_hist1974-2004_fut2020-2100_spei{spei_month}.csv"
years = [str(y) for y in range(2020, 2101)]
years_5 = [str(y) for y in range(2020, 2101, 5)]
years_avg_5 = [y for y in years_5 if int(y) >= start_year_avg]
len_avg = end_year_avg - start_year_avg + 1
metrics = ["dy [%]", "dp [Mt]", "de [B USD]", "degdp [%]"]

prod_weights = utils.load_fao_prod_weights(root_dir, spei_month)
ssp_gdp = utils.load_ssp_gdp_weights(root_dir)
df_ipcc = utils.get_ipcc_region_df()[["Country", "region_ar6_dev"]].rename(columns={"Country": "country"})

"""
HELPERS
"""
def load(var, subdir):
    return pd.concat([
        pd.read_csv(f"{root_dir}/ssp/linregress_outputs/{crop}/{subdir}{var}_{file_suffix}").assign(crop=crop)
        for crop in crops
    ]).merge(df_ipcc, on="country", how="left")


def smooth(values):
    return utils.movingaverage_pandas(values, window_size).values


def dy_series(dy):
    """
    ESM-mean of FAO production-weighted dy, per year
    """
    esm_vals = []
    for _, d in dy.groupby("esm"):
        d = d.set_index(["country", "crop"])[years]
        common = d.index.intersection(prod_weights.index)
        w = prod_weights[common] / prod_weights[common].sum()
        esm_vals.append(d.loc[common].mul(w, axis=0).sum())
    return pd.concat(esm_vals, axis=1).mean(axis=1).values


def degdp_series(degdp, ssp):
    """
    ESM-mean of SSP GDP-weighted degdp (country value = mean over crops), per 5-year step
    """
    gdp = ssp_gdp.xs(ssp, level="ssp")
    esm_vals = []
    for _, d in degdp.groupby("esm"):
        d = d.groupby("country")[years_5].mean()
        common = d.index.intersection(gdp.index)
        esm_vals.append([(gdp.loc[common, y] / gdp.loc[common, y].sum() * d.loc[common, y]).sum() for y in years_5])
    return np.mean(esm_vals, axis=0)


def sum_series(df, scale):
    return df.groupby("esm")[years].sum().mean().values / scale


"""
GLOBAL IMPACTS (ESM mean, 2080-2100)
"""
table = pd.DataFrame(index=ssps, columns=pd.MultiIndex.from_product([metrics, list(calib_periods)]), dtype=float)

for label, subdir in calib_periods.items():
    dy, dp, de, degdp = load("dy", subdir), load("dp", subdir), load("de", subdir), load("degdp", subdir)
    for ssp in ssps:
        by_region = lambda df: [df[(df["ssp"] == ssp) & (df["region_ar6_dev"] == r)] for r in regions]
        dy_reg = [smooth(dy_series(d))[-len_avg:].mean() for d in by_region(dy)]
        degdp_reg = [pd.Series(degdp_series(d, ssp), index=years_5)[years_avg_5].mean() for d in by_region(degdp)]

        table.loc[ssp, ("dy [%]", label)] = np.mean(dy_reg)
        table.loc[ssp, ("dp [Mt]", label)] = smooth(sum_series(dp[dp["ssp"] == ssp], 1e6))[-len_avg:].sum()
        table.loc[ssp, ("de [B USD]", label)] = smooth(sum_series(de[de["ssp"] == ssp], 1e9))[-len_avg:].sum()
        table.loc[ssp, ("degdp [%]", label)] = np.mean(degdp_reg)

rows = []
for m in metrics:
    nd = 4 if m == "degdp [%]" else 1
    for ssp in ssps:
        rows.append([m if ssp == ssps[0] else "", ssp] + [f"{table.loc[ssp, (m, p)]:.{nd}f}" for p in calib_periods])

print(f"\nGlobal impacts by calibration period, {start_year_avg}-{end_year_avg} (dy, degdp: mean; dp, de: sum)\n")
print(tabulate(rows, headers=["", "SSP"] + list(calib_periods), colalign=("left", "left", "right", "right"), disable_numparse=True))

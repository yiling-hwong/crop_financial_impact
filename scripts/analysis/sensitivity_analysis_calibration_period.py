# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""

import sys
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
years = [str(y) for y in range(2081, 2101)]

calib_periods = {
    "2007-2019": "",
    "2000-2019": "2000_2019/",
}

root_dir = f'../../data'
file_suffix = f"isimip3b_{pred_str}_hist1974-2004_fut2020-2100_spei{spei_month}.csv"

years_5 = [y for y in years if int(y) % 5 == 0]
metrics = ["dy [%]", "dp [Mt]", "de [B USD]", "degdp [%]"]

prod_weights = utils.load_fao_prod_weights(root_dir, spei_month)
ssp_gdp = utils.load_ssp_gdp_weights(root_dir)

"""
GLOBAL IMPACTS (ESM mean, 2081-2100 mean)
"""
def load(var, subdir):
    return pd.concat([
        pd.read_csv(f"{root_dir}/ssp/linregress_outputs/{crop}/{subdir}{var}_{file_suffix}").assign(crop=crop)
        for crop in crops
    ])


table = pd.DataFrame(index=ssps, columns=pd.MultiIndex.from_product([metrics, list(calib_periods)]), dtype=float)
for label, subdir in calib_periods.items():
    dy, dp, de = load("dy", subdir), load("dp", subdir), load("de", subdir)
    for ssp in ssps:
        dy_ssp = dy[dy["ssp"] == ssp].groupby(["country", "crop"])[years].mean().reset_index()
        table.loc[ssp, ("dy [%]", label)] = utils.compute_dy_weighted(dy_ssp, None, years, prod_weights=prod_weights).mean()

        dp_ssp = dp[dp["ssp"] == ssp].groupby("esm")[years].sum().mean()
        table.loc[ssp, ("dp [Mt]", label)] = dp_ssp.sum() / 1e6

        de_ssp = de[de["ssp"] == ssp]
        table.loc[ssp, ("de [B USD]", label)] = de_ssp.groupby("esm")[years].sum().mean().sum() / 1e9

        de_c = de_ssp.groupby(["country", "esm"])[years_5].sum().groupby("country").mean()
        gdp = ssp_gdp.xs(ssp, level="ssp")
        common = de_c.index.intersection(gdp.index)
        table.loc[ssp, ("degdp [%]", label)] = (de_c.loc[common].sum() / gdp.loc[common, years_5].sum()).mean() * 100

rows = []
for m in metrics:
    nd = 4 if m == "degdp [%]" else 1
    for ssp in ssps:
        rows.append([m if ssp == ssps[0] else "", ssp] + [f"{table.loc[ssp, (m, p)]:.{nd}f}" for p in calib_periods])
print("\nGlobal impacts by calibration period, 2081-2100 (dy, degdp: mean; dp, de: sum)\n")
print(tabulate(rows, headers=["", "SSP"] + list(calib_periods), colalign=("left", "left", "right", "right")))

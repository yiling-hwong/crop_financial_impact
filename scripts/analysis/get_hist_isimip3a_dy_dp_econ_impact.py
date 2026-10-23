# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import xarray as xr

import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('..')
import utils

"""
PARAMETERS
"""
crops = ["maize", "wheat", "soy"]
pred_str = "t3s3"
spei_month = "03"  # 03, 12 (06, 09 have no 1971-2006 files)

start_year_hist, end_year_hist = 1974, 2004  # 1974-2004, 1970-1990, 1980-2000, 1984-2014
start_year_fut, end_year_fut = 1990, 2019
tmax_unit = "K"  # K (main analysis), degC

root_dir = f'../../data'
tmax_hist_dir = "path/to/historical/tmax"  # growing-season gridded tmax netCDFs, ISIMIP3a
spei_hist_dir = "path/to/historical/spei"  # growing-season gridded SPEI netCDFs, ISIMIP3a
rdf_dir = "path/to/pixel_data"  # pixel-to-country mapping with harvested area (rdf_{crop}.csv), ISO3 country names (iso3.csv)

predictors = ["tmax", "tmax2", "tmax3", "spei", "spei2", "spei3"]
yrs = [str(x) for x in range(start_year_fut, end_year_fut + 1)]
file_suffix = f"isimip3a_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"

drop_countries = {
    "maize": ['LBR', 'MNE', 'PSE', 'SSD', 'XXX', 'Z01', 'Z03', 'Z06', 'Z07', 'XKO'],
    "soy": ['XKO', 'XXX', 'COG', 'AFG', 'MKD', 'MNE', 'MOZ'],
    "wheat": ['XCA', 'XKO', 'XXX', 'Z01', 'Z06', 'Z07'],
}
fao_drop = ['VNM', 'SSD', 'MNE']
hg_to_t = 1e-4
tmax_offset = 273.15 if tmax_unit == "degC" else 0

cnames = pd.read_csv(f"{rdf_dir}/iso3.csv", index_col="iso3")["name"]
gdp = pd.read_csv(f"{root_dir}/resources/gdp_world_bank_2015_usd.csv", index_col="Country Code")

"""
HELPERS
"""
def get_yhat(tmax, spei, coefs):
    """
    Linear predictor of log yield from growing-season means of the t3s3 terms
    """
    terms = [tmax, tmax**2, tmax**3, spei, spei**2, spei**3]
    return sum(c * t.mean("time") for c, t in zip(coefs, terms))


def open_gs(d, var, fname, crop):
    """
    Growing-season gridded data, 1971-2006 and 2007-2019 files concatenated
    """
    files = [f"{d}/{fname}_gs_{crop}_hist.nc", f"{d}/{fname}_gs_{crop}.nc"]
    return xr.concat([xr.open_dataset(f)[var] for f in files], dim="time").load()


for crop in crops:
    print(f"Crop: {crop}")
    out_dir = f"{root_dir}/historical/linregress_outputs/{crop}/isimip3a"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{root_dir}/models/model_{crop}_{pred_str}_spei{spei_month}.pickle", "rb") as f:
        coefs = pickle.load(f).params[predictors].values

    rdf = pd.read_csv(f"{rdf_dir}/rdf_{crop}.csv", index_col="pixID")
    harv_area_frac = rdf["harv_area"] / rdf.groupby("country")["harv_area"].transform("sum")

    """
    YIELD CHANGE (dy, %)
    """
    tmax = open_gs(tmax_hist_dir, "tmax", "pixdat_tmax", crop) - tmax_offset
    spei = open_gs(spei_hist_dir, "spei", f"pixdat_spei{spei_month}", crop)

    hist_slice = slice(f"{start_year_hist}-01-01", f"{end_year_hist}-12-31")
    ybase = get_yhat(tmax.sel(time=hist_slice), spei.sel(time=hist_slice), coefs)

    dy_pix = pd.DataFrame(index=rdf.index)
    for yr in yrs:
        yfut = get_yhat(tmax.sel(time=yr), spei.sel(time=yr), coefs)
        dy_pix[yr] = ((np.exp(yfut - ybase) - 1) * 100).values

    dy = dy_pix.mul(harv_area_frac, axis=0).groupby(rdf["country"]).sum()
    dy.index.name = "country"
    dy.to_csv(f"{out_dir}/dy_{file_suffix}")

    """
    PRODUCTION CHANGE (dp, tonnes)
    """
    fao = pd.read_csv(f"{root_dir}/historical/faostat/FAOSTAT_{crop}.csv")
    faoha = fao[fao["Element"] == "Area harvested"].pivot_table(index="Area", columns="Year", values="Value", aggfunc="first")
    faoy = fao[fao["Element"] == "Yield"].pivot_table(index="Area", columns="Year", values="Value", aggfunc="first")

    dp = dy[~dy.index.isin(drop_countries[crop] + fao_drop)].copy()
    prod = (faoha * faoy * hg_to_t).reindex(cnames.loc[dp.index].values)[[int(y) for y in yrs]]
    dp[yrs] = dp[yrs].values * prod.values / 100
    dp["countryname"] = cnames.loc[dp.index].values
    dp.to_csv(f"{out_dir}/dp_{file_suffix}")

    """
    ECONOMIC LOSS (de, USD 2015)
    """
    fao_pp = utils.get_fao_producer_prices(crop, start_year_fut, end_year_fut)
    fao_pp = fao_pp[np.isfinite(fao_pp["pp"])].drop_duplicates(["iso", "year"])
    pp = fao_pp.pivot(index="iso", columns="year", values="pp").reindex(columns=[int(y) for y in yrs])

    de = dp.loc[dp.index.isin(pp.index), yrs].copy()
    de[yrs] = de.values * pp.loc[de.index].values
    de.to_csv(f"{out_dir}/de_{file_suffix}")

    """
    ECONOMIC LOSS AS SHARE OF GDP (degdp, %)
    """
    degdp = de[de.index.isin(gdp.index)].copy()
    degdp[yrs] = degdp.values / gdp.loc[degdp.index, yrs].values * 100
    degdp.to_csv(f"{out_dir}/degdp_{file_suffix}")

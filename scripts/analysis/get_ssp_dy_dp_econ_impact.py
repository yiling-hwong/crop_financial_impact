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
spei_month = "03"
ssp_spei_month = "12"  # ISIMIP3b SSP growing-season SPEI files are SPEI12
calib_start_year = 2000  # 2007 (main analysis); 2000 (sensitivity check)
calib_end_year = 2019
tmax_unit = "degC"  # K (main analysis), degC

start_year_hist, end_year_hist = 1974, 2004
start_year_fut, end_year_fut = 2020, 2100
ssps = ["ssp126", "ssp245", "ssp370"]
esms = ["gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll"]

root_dir = f'../../data'
ext_root_dir = "/Users/yhwong/Documents/IIASA/NGFS_crop_impact"
gs_dir = f"{ext_root_dir}/outputs/growing_season"
period_dir = "" if calib_start_year == 2007 else f"{calib_start_year}_{calib_end_year}/"
tmax_offset = 273.15 if tmax_unit == "degC" else 0

predictors = ["tmax", "tmax2", "tmax3", "spei", "spei2", "spei3"]
yrs_de = [str(x) for x in range(start_year_fut, end_year_fut + 1)]
yrs_degdp = [str(x) for x in range(start_year_fut, end_year_fut + 1, 5)]
file_suffix = f"isimip3b_{pred_str}_hist{start_year_hist}-{end_year_hist}_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv"

drop_countries = {
    "maize": ['LBR', 'MNE', 'PSE', 'SSD', 'XXX', 'Z01', 'Z03', 'Z06', 'Z07', 'XKO'],
    "soy": ['XKO', 'XXX', 'COG', 'AFG', 'MKD', 'MNE', 'MOZ'],
    "wheat": ['XCA', 'XKO', 'XXX', 'Z01', 'Z06', 'Z07'],
}
fao_drop = ['VNM', 'SSD', 'MNE']
fao_last_year = 2022
hg_to_t = 1e-4

cnames = pd.read_csv(f"{ext_root_dir}/corey-yield-impacts/data/iso3.csv", index_col="iso3")["name"]
gdp = utils.load_ssp_gdp_weights(root_dir)

"""
HELPERS
"""
def get_yhat(tmax, spei, coefs):
    """
    Linear predictor of log yield from growing-season means of the t3s3 terms
    """
    terms = [tmax, tmax**2, tmax**3, spei, spei**2, spei**3]
    return sum(c * t.mean("time") for c, t in zip(coefs, terms))


for crop in crops:
    print(f"Crop: {crop}")
    out_dir = f"{root_dir}/ssp/linregress_outputs/{crop}/{period_dir}"
    os.makedirs(out_dir, exist_ok=True)

    model_file = f"{root_dir}/models/{period_dir}model_{crop}_{pred_str}_spei{spei_month}.pickle"
    with open(model_file, "rb") as f:
        coefs = pickle.load(f).params[predictors].values

    rdf = pd.read_csv(f"{ext_root_dir}/corey-yield-impacts/data/rdf_{crop}.csv", index_col="pixID")
    harv_area_frac = rdf["harv_area"] / rdf.groupby("country")["harv_area"].transform("sum")

    """
    YIELD CHANGE (dy, %)
    """
    hist_slice = slice(f"{start_year_hist}-01-01", f"{end_year_hist}-12-31")
    tmax_hist = xr.open_dataset(f"{gs_dir}/tmax/isimip3a/corey/pixdat_tmax_gs_{crop}_hist.nc")["tmax"].sel(time=hist_slice) - tmax_offset
    spei_hist = xr.open_dataset(f"{gs_dir}/spei/isimip3a/corey/pixdat_spei{ssp_spei_month}_gs_{crop}_hist.nc")["spei"].sel(time=hist_slice)
    ybase = get_yhat(tmax_hist, spei_hist, coefs)

    fut_slice = slice(f"{start_year_fut}-01-01", f"{end_year_fut}-12-31")
    dy_list = []
    for ssp in ssps:
        for esm in esms:
            print(f"  {ssp} {esm}")
            tmax_file = f"{gs_dir}/tmax/isimip3b/corey/2015_2100/{esm}_pixdat_tmax_{ssp}_gs_{crop}.nc"
            spei_file = f"{gs_dir}/spei/isimip3b/corey/2015_2100/{esm}_pixdat_spei_{ssp}_gs_{crop}.nc"
            tmax = xr.open_dataset(tmax_file)["tmax"].sel(time=fut_slice).load() - tmax_offset
            spei = xr.open_dataset(spei_file)["spei"].sel(time=fut_slice).load()

            dy_pix = pd.DataFrame(index=rdf.index)
            for yr in yrs_de:
                yfut = get_yhat(tmax.sel(time=yr), spei.sel(time=yr), coefs)
                dy_pix[yr] = ((np.exp(yfut - ybase) - 1) * 100).values
            dy_ssp = dy_pix.mul(harv_area_frac, axis=0).groupby(rdf["country"]).sum()
            dy_ssp.insert(0, "ssp", ssp)
            dy_ssp.insert(1, "esm", esm)
            dy_list.append(dy_ssp)

    dy = pd.concat(dy_list)
    dy = dy[~dy.index.isin(drop_countries[crop])]
    dy.index.name = "country"
    dy.to_csv(f"{out_dir}/dy_{file_suffix}")

    """
    PRODUCTION CHANGE (dp, tonnes)
    """
    fao = pd.read_csv(f"{root_dir}/historical/faostat/FAOSTAT_{crop}.csv")
    faoha = fao[fao["Element"] == "Area harvested"].pivot_table(index="Area", columns="Year", values="Value", aggfunc="first")
    faoy = fao[fao["Element"] == "Yield"].pivot_table(index="Area", columns="Year", values="Value", aggfunc="first")

    countries = dy.index.unique().difference(fao_drop)
    prod = (faoha * faoy * hg_to_t).reindex(cnames.loc[countries].values)
    prod.index = countries
    prod = pd.DataFrame({yr: prod[min(int(yr), fao_last_year)] for yr in yrs_de})

    dp = dy[dy.index.isin(countries)].copy()
    dp[yrs_de] = dp[yrs_de].values * prod.loc[dp.index].values / 100
    dp.to_csv(f"{out_dir}/dp_{file_suffix}")

    """
    ECONOMIC LOSS (de, USD 2015)
    """
    fao_pp = utils.get_fao_producer_prices(crop, 1990, 2024)
    fao_pp = fao_pp[np.isfinite(fao_pp["pp"])]
    pp = {}
    for iso, g in fao_pp.groupby("iso"):
        s = g.sort_values("year").set_index("year")["pp"]
        pp[iso] = s.reindex(range(s.index.min(), end_year_fut + 1)).ffill()
    pp = pd.DataFrame(pp).T.reindex(columns=range(start_year_fut, end_year_fut + 1))
    pp.columns = yrs_de

    de = dp[dp.index.isin(pp.index)].copy()
    de[yrs_de] = de[yrs_de].values * pp.loc[de.index].values
    de.to_csv(f"{out_dir}/de_{file_suffix}")

    """
    ECONOMIC LOSS AS SHARE OF GDP (degdp, %)
    """
    degdp = de[de.index.isin(gdp.index.get_level_values("country"))][["ssp", "esm"] + yrs_degdp]
    gdp_rows = gdp.reindex(pd.MultiIndex.from_arrays([degdp.index, degdp["ssp"]]))[yrs_degdp].values
    degdp[yrs_degdp] = degdp[yrs_degdp].values / gdp_rows * 100
    degdp.to_csv(f"{out_dir}/degdp_{file_suffix}")

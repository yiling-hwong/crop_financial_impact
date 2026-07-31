# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

root_dir = f'../../data'
ipcc_region_file = f'{root_dir}/resources/region_classification.xlsx'

def format_fn(x,p):
    """
    Format function for colorbar ticks
    """
    if x == 0:
        return '0'
    if x.is_integer():
        return f'{int(x)}'
    # Remove trailing zeros after decimal point
    return f'{x:f}'.rstrip('0').rstrip('.')


def get_ipcc_region_df():
    df = pd.read_excel(ipcc_region_file)

    # Rename the country column
    df = df.rename(columns={'ISO': 'Country'})

    return df

def generate_alphabet_list(n,option):

    import string

    if option == "lower":
        alphabets = list(string.ascii_lowercase)

    if option == "upper":
        alphabets = list(string.ascii_uppercase)

    alphabet_list = alphabets[:n]

    return alphabet_list


def movingaverage(values, window_size, mode):
    '''
    Get moving average of list of values with specified window size
    '''

    import numpy as np

    window = np.ones(int(window_size)) / float(window_size)

    return np.convolve(values, window, mode)  # 'valid' 'full' 'same'

def movingaverage_pandas(values, window_size):
    """
    Calculate moving average using pandas rolling
    - Center window on each point
    """
    series = pd.Series(values)
    return series.rolling(window=window_size, center=True, min_periods=1).mean()


def movingaverage_scipy(values, window_size):
    """
    Calculate moving average using scipy's savgol_filter
    - Preserves features better
    """
    from scipy.signal import savgol_filter
    # window_size must be odd for savgol
    if window_size % 2 == 0:
        window_size += 1
    return savgol_filter(values, window_size, polyorder=1)

def get_smoothed_values(values, window_size, method, np_mode):
    """
    Apply smoothing to values based on specified method

    Args:
        values: numpy array of values to smooth
        window_size: size of moving window
        method: smoothing method ('numpy', 'pandas', or 'scipy')
        np_mode: mode for numpy moving average ('valid', 'same', or 'full')

    Returns:
        smoothed values
    """
    if method == 'numpy':
        return movingaverage(values, window_size, np_mode)
    elif method == 'pandas':
        return movingaverage_pandas(values, window_size)
    elif method == 'scipy':
        return movingaverage_scipy(values, window_size)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

def compute_degdp_weighted(df_degdp, year_cols, gdp_weights=None, weight_opt="weighted"):
    """
    GDP-weighted degdp timeseries.
    """
    if weight_opt == "simple":
        return df_degdp.groupby("country")[year_cols].mean().mean(axis=0).values

    degdp_c = df_degdp.groupby("country")[year_cols].mean()
    common = degdp_c.index.intersection(gdp_weights.index)
    w = gdp_weights[common]
    w = w / w.sum()
    result = []
    for yr in year_cols:
        result.append(float((w * degdp_c.loc[common, yr]).sum()))
    return np.nan_to_num(np.array(result), nan=0)


def compute_dy_weighted(df_dy, df_dp, year_cols, prod_weights=None, weight_opt="weighted"):
    """
    Production-weighted dy timeseries.
    """
    if weight_opt == "simple":
        return df_dy.groupby("country")[year_cols].mean().mean(axis=0).values

    if prod_weights is not None:
        # --- FAO production weights ---
        dy_indexed = df_dy.set_index(["country", "crop"])[year_cols]
        common = dy_indexed.index.intersection(prod_weights.index)
        w = prod_weights[common]
        w = w / w.sum()  # normalize within common set
        result = []
        for yr in year_cols:
            result.append(float((w * dy_indexed.loc[common, yr]).sum()))
        return np.nan_to_num(np.array(result), nan=0)

    dp_c = df_dp.groupby("country")[year_cols].sum()
    dy_c = df_dy.groupby("country")[year_cols].sum()

    dp_mean = dp_c.mean(axis=1)
    dy_mean = dy_c.mean(axis=1)

    prod_w = (dp_mean / dy_mean).replace([np.inf, -np.inf], np.nan).abs()
    prod_w = prod_w.dropna()
    prod_w = prod_w[prod_w > 0]
    common = dy_c.index.intersection(prod_w.index)
    den = prod_w[common].sum()
    result = []
    for yr in year_cols:
        result.append(float((prod_w[common] * dy_c.loc[common, yr]).sum() / den) if den != 0 else 0.0)
    return np.nan_to_num(np.array(result), nan=0)


def load_fao_prod_weights(root_dir, spei_month="03"):
    """
    Load FAO production weights
    """
    fpath = f"{root_dir}/historical/linregress_inputs/INPUT_HISTORICAL_spei{spei_month}.csv"
    df = pd.read_csv(fpath)
    df = df[df["year"].between(2007, 2019)]
    prod = df.groupby(["country", "crop"])["production"].mean()
    prod = prod[prod > 0].dropna()
    return prod


def load_gdp_weights(root_dir, baseline_years=None):
    """
    Load World Bank GDP weights (constant 2015 USD).
    """
    fpath = f"{root_dir}/resources/gdp_world_bank_2015_usd.csv"
    df = pd.read_csv(fpath)
    if baseline_years is None:
        baseline_years = list(range(2007, 2020))
    yr_cols = [str(y) for y in baseline_years if str(y) in df.columns]
    gdp = (df.set_index("Country Code")[yr_cols]
             .replace('', np.nan)
             .astype(float)
             .mean(axis=1)
             .dropna())
    gdp = gdp[gdp > 0]
    return gdp


def load_ssp_gdp_weights(root_dir):
    """
    Load SSP-projected GDP
    """
    ssp_mapping = {'ssp1': 'ssp126', 'ssp2': 'ssp245', 'ssp3': 'ssp370', 'ssp5': 'ssp585'}
    yrs_degdp = [str(x) for x in range(2020, 2101, 5)]
    yrs_ppp   = [str(x) for x in range(1998, 2021)]

    gdp_file      = f"{root_dir}/resources/gdp_update_2021.xlsx"
    gdp_hist_file = f"{root_dir}/resources/gdp_world_bank_2015_usd.csv"

    df_gdp = pd.read_excel(gdp_file, sheet_name="country_level")

    gdppc = df_gdp[df_gdp['variable'] == 'gdppc']
    gdppc = gdppc[gdppc['scenario'] != 'WDI']
    pop   = df_gdp[df_gdp['variable'] == 'pop']
    pop   = pop[pop['scenario'] != 'WDI']

    yrs_gdp = [str(int(yrs_degdp[0]) + 1)] + yrs_degdp[1:]
    cols_to_extract = ['countryCode', 'scenario'] + yrs_gdp

    gdppc = gdppc[cols_to_extract]
    gdppc['scenario'] = gdppc['scenario'].str.lower()
    gdppc = gdppc.rename(columns={'2021': '2020', 'countryCode': 'country', 'scenario': 'ssp'})
    gdppc = gdppc.set_index('country')

    pop = pop[cols_to_extract]
    pop['scenario'] = pop['scenario'].str.lower()
    pop[yrs_gdp] = pop[yrs_gdp] * 1e6
    pop = pop.rename(columns={'2021': '2020', 'countryCode': 'country', 'scenario': 'ssp'})
    pop = pop.set_index('country')

    gdp = gdppc.copy()
    gdp[yrs_degdp] = gdppc[yrs_degdp] * pop[yrs_degdp]
    gdp['ssp'] = gdp['ssp'].replace(ssp_mapping)
    gdp = gdp[~gdp['ssp'].isin(['ssp4', 'ssp5'])]

    # GDP correction: 2017 PPP → constant 2015 USD
    gdp_hist = pd.read_csv(gdp_hist_file)
    gdp_hist = gdp_hist.drop(columns=["Country Name", "Indicator Name", "Indicator Code", "Unnamed: 68"],
                             errors='ignore')
    gdp_hist = gdp_hist.rename(columns={"Country Code": "country"}).set_index("country")
    yr_hist_avail = [y for y in yrs_ppp if y in gdp_hist.columns]
    gdp_hist = gdp_hist[yr_hist_avail].replace('', np.nan).astype(float)

    gdppc_ppp = df_gdp[df_gdp['variable'] == 'gdppc']
    gdppc_ppp = gdppc_ppp[gdppc_ppp['scenario'] == 'WDI']
    pop_ppp   = df_gdp[df_gdp['variable'] == 'pop']
    pop_ppp   = pop_ppp[pop_ppp['scenario'] == 'WDI']
    gdppc_ppp = gdppc_ppp.set_index('countryCode')
    pop_ppp   = pop_ppp.set_index('countryCode')

    yr_ppp_avail = [y for y in yrs_ppp if y in gdppc_ppp.columns]
    gdp_ppp = gdppc_ppp[yr_ppp_avail] * (pop_ppp[yr_ppp_avail] * 1e6)
    gdp_ppp.index.name = 'country'

    common_yrs = [y for y in yr_hist_avail if y in gdp_ppp.columns]
    corr_ratio = (gdp_hist[common_yrs] / gdp_ppp[common_yrs]).mean(axis=1)

    year_cols = [col for col in gdp.columns if str(col).isdigit()]
    gdp_corr = gdp.copy()
    gdp_corr = gdp_corr.set_index('ssp', append=True)
    gdp_corr[year_cols] = gdp_corr[year_cols].mul(corr_ratio, axis=0)
    gdp_corr = gdp_corr.reset_index(level='ssp')

    # Return as MultiIndex(country, ssp) with year columns only
    gdp_final = gdp_corr.set_index('ssp', append=True)[yrs_degdp]
    gdp_final.index.names = ['country', 'ssp']
    return gdp_final

def get_fao_producer_prices(crop,start_year,end_year):
    """
    GET FAOSTAT producer prices
    Adjust for inflation (2015) using FAOSTAT producer price index
    """
    root_dir = f"../../data"
    fao_producer_price_file = f"{root_dir}/historical/faostat/FAOSTAT_PP_data_en_10-14-2025.csv"
    country_iso_code_file = f"{root_dir}/resources/country_iso_m49.csv"

    faopp_master = pd.read_csv(fao_producer_price_file)
    iso_code = pd.read_csv(country_iso_code_file)

    if crop == "maize":
        crop_pp = "Maize (corn)"
    if crop == "wheat":
        crop_pp = "Wheat"
    if crop == "soy":
        crop_pp = "Soya beans"

    years = [int(x) for x in range(start_year, end_year + 1)]
    faopp = faopp_master[(faopp_master["Element"] == "Producer Price (USD/tonne)") & (faopp_master["Item"] == crop_pp) & (faopp_master["Year"].isin(years))]
    faoppi = faopp_master[(faopp_master["Element"] == "Producer Price Index (2014-2016 = 100)") & (faopp_master["Item"] == crop_pp) & (faopp_master["Year"].isin(years))]
    faopp = faopp[["Area Code (M49)", "Area", "Year", "Value"]]
    faoppi = faoppi[["Area Code (M49)", "Area", "Year", "Value"]]

    fao_pp = faopp.merge(iso_code[['Area Code (M49)', 'ISO-alpha3']], on='Area Code (M49)', how='left')
    fao_pp.columns = ["M49", "cname", "year", "pp", "iso"]
    fao_pp = fao_pp[["cname", "year", "pp", "iso"]]
    fao_pp = fao_pp[fao_pp['iso'].notna()]

    fao_ppi = faoppi.merge(iso_code[['Area Code (M49)', 'ISO-alpha3']], on='Area Code (M49)', how='left')
    fao_ppi.columns = ["M49","cname","year","ppi","iso"]
    fao_ppi = fao_ppi [["cname","year","ppi","iso"]]
    fao_ppi = fao_ppi[fao_ppi['iso'].notna()]

    fao_pp = fao_pp.merge(fao_ppi, on=['cname', 'year', 'iso'], how='left')

    # Get PPI for base year
    ppi_base = fao_pp[fao_pp['year'] == 2015][['iso', 'ppi']].rename(columns={'ppi': 'ppi_base'})
    fao_pp = pd.merge(fao_pp, ppi_base[['iso', 'ppi_base']], on='iso', how='left')

    # ADJUST PP for INFLATION TO 2015 (2014-2016=100, so PPI is already adjusted to 2015 is ~100)
    fao_pp["year"] = fao_pp["year"].astype(int)
    fao_pp_adj = fao_pp.copy()

    fao_pp_adj["pp_deflated"] = (fao_pp["pp"] / fao_pp["ppi"]) * fao_pp["ppi_base"]

    fao_pp_adj = fao_pp_adj.drop(columns=["pp"])
    fao_pp_adj = fao_pp_adj.rename(columns={'pp_deflated': 'pp'})
    fao_pp_adj = fao_pp_adj[fao_pp_adj['pp'].notna()] # drop rows when no values for pp

    fao_pp = fao_pp_adj

    return fao_pp


def get_truncated_cmap(cmap_name, max_frac=0.8):
    """
    Return a ListedColormap truncated to [0, max_frac] of the named colormap,
    useful for skipping the lightest or darkest end of a colormap.

    Parameters
    ----------
    cmap_name : str
        Name of a matplotlib colormap (e.g. 'RdPu_r').
    max_frac : float
        Upper bound of the colormap range to keep (default 0.8, skipping top 20%).

    Returns
    -------
    ListedColormap
    """
    cmap = cm.get_cmap(cmap_name, 256)
    return ListedColormap(cmap(np.linspace(0, max_frac, 256)))


# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import sys
sys.path.append('../')
import pandas as pd
import numpy as np

ipcc_region_file = '../../data/resources/region_classification.xlsx'

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
    - Better edge handling using min_periods
    - Center window on each point
    """
    series = pd.Series(values)
    return series.rolling(window=window_size, center=True, min_periods=1).mean()


def movingaverage_scipy(values, window_size):
    """
    Calculate moving average using scipy's savgol_filter
    - Excellent edge handling
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
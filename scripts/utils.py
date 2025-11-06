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
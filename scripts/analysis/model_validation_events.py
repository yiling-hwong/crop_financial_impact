# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""

import sys
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('../')

"""
PARAMETERS
"""
pred_str = "t3s3"
spei_month = "03"
spei_str = f"_spei{spei_month}"

start_year = 2007
end_year = 2019

plot_flag = False

root_dir = f'../../data'
input_hist_file = f"{root_dir}/historical/linregress_inputs/INPUT_HISTORICAL{spei_str}.csv"

predictors = ["tmax", "tmax2", "tmax3", "spei", "spei2", "spei3"]

EVENTS = [
    {"event": "Russia 2010 heatwave/drought",           "country": "RUS", "crop": "wheat", "year": 2010},
    {"event": "Ukraine 2010 heatwave/drought",           "country": "UKR", "crop": "wheat", "year": 2010},
    {"event": "USA 2012 Midwest drought (maize)",        "country": "USA", "crop": "maize", "year": 2012},
    {"event": "USA 2012 Midwest drought (soy)",          "country": "USA", "crop": "soy",   "year": 2012},
    {"event": "Argentina 2018 drought",                  "country": "ARG", "crop": "soy",   "year": 2018},
    {"event": "Australia 2018 drought",                  "country": "AUS", "crop": "wheat", "year": 2018},
    {"event": "Australia 2019 drought",                  "country": "AUS", "crop": "wheat", "year": 2019},
    {"event": "Horn of Africa 2011 drought (Ethiopia)",  "country": "ETH", "crop": "maize", "year": 2011},
    {"event": "Horn of Africa 2011 drought (Kenya)",     "country": "KEN", "crop": "maize", "year": 2011},
]


def detrend_demean_within_fold(df_in, vars_to_transform, train_years, test_years,
                                entity_col='country'):
    """
    For each country, fit a linear trend + mean on TRAINING years only,
    then detrend/demean both train and test rows using that fit.
    Ported from run_linregress.py's LOYO blocked CV (same logic).
    """
    train_rows = []
    test_rows = []

    for entity, grp in df_in.groupby(entity_col):
        grp_train = grp[grp['year'].isin(train_years)].copy()
        grp_test = grp[grp['year'].isin(test_years)].copy()

        if len(grp_train) < 2:
            continue
        if len(grp_test) == 0:
            continue

        for var in vars_to_transform:
            trend_fit = smf.ols(f'{var} ~ year_numeric', data=grp_train).fit()

            grp_train[f'{var}_dtr'] = grp_train[var] - trend_fit.predict(grp_train)
            grp_test[f'{var}_dtr'] = grp_test[var] - trend_fit.predict(grp_test)

            train_mean = grp_train[f'{var}_dtr'].mean()
            grp_train[f'{var}_dtr_dmn'] = grp_train[f'{var}_dtr'] - train_mean
            grp_test[f'{var}_dtr_dmn'] = grp_test[f'{var}_dtr'] - train_mean

        train_rows.append(grp_train)
        test_rows.append(grp_test)

    df_train_out = pd.concat(train_rows, ignore_index=True)
    df_test_out = pd.concat(test_rows, ignore_index=True)
    return df_train_out, df_test_out


def load_crop_panel(df_all, crop):
    df_crop = df_all[(df_all["crop"] == crop) & (df_all["year"].between(start_year, end_year))].copy()
    df_crop['log_yield'] = np.log(df_crop["yield"])
    df_crop['year_numeric'] = df_crop['year'].astype(int)
    return df_crop


def actual_pct_rank(df_crop, country, year):
    """
    Percentile rank (1 = worst) of `year`'s actual yield anomaly within
    country's own full-period (start_year-end_year) detrended distribution.
    Full-period trend used here since this is purely a diagnostic of how
    extreme the observed year was, not part of the OOS prediction itself.
    """
    grp = df_crop[df_crop["country"] == country].copy()
    if len(grp) < 3 or year not in grp["year"].values:
        return None, None
    trend_fit = smf.ols('log_yield ~ year_numeric', data=grp).fit()
    grp['resid'] = grp['log_yield'] - trend_fit.predict(grp)
    grp_sorted = grp.sort_values('resid')
    rank = int(np.where(grp_sorted['year'].values == year)[0][0]) + 1
    return rank, len(grp)


"""
RUN LOYO REFIT PER (crop, event_year) GROUP
"""
df_all = pd.read_csv(input_hist_file)

results = []

groups = sorted(set((e["crop"], e["year"]) for e in EVENTS))

for crop, event_year in groups:

    df_crop = load_crop_panel(df_all, crop)
    all_years = sorted(df_crop['year'].unique())

    if event_year not in all_years:
        print(f"SKIP {crop} {event_year}: year not in panel")
        continue

    train_years = [y for y in all_years if y != event_year]
    test_years = [event_year]

    df_train_fold, df_test_fold = detrend_demean_within_fold(
        df_crop,
        vars_to_transform=['log_yield'] + predictors,
        train_years=train_years,
        test_years=test_years,
        entity_col='country'
    )

    y_fold = 'log_yield_dtr_dmn'
    preds_fold = [f'{p}_dtr_dmn' for p in predictors]
    formula_fold = f"{y_fold} ~ 1 + {' + '.join(preds_fold)}"

    df_train_idx = df_train_fold.set_index(['country', 'year'])
    df_test_idx = df_test_fold.set_index(['country', 'year'])

    model_fold = PanelOLS.from_formula(formula_fold, data=df_train_idx).fit()

    X_test = df_test_idx[preds_fold].copy()
    X_test.insert(0, 'Intercept', 1.0)
    y_test = df_test_idx[y_fold]

    y_pred = model_fold.predict(X_test)
    common_idx = y_test.index.intersection(y_pred.index)
    fold_r2 = r2_score(y_test.loc[common_idx], y_pred.loc[common_idx, 'predictions'])

    for e in [e for e in EVENTS if e["crop"] == crop and e["year"] == event_year]:
        country = e["country"]
        idx = (country, event_year)

        if idx not in y_test.index:
            print(f"SKIP {e['event']} ({country}): not in test fold")
            continue

        actual_pct = float(y_test.loc[idx]) * 100
        predicted_pct = float(y_pred.loc[idx, 'predictions']) * 100

        sign_match = np.sign(actual_pct) == np.sign(predicted_pct)
        magnitude_ratio = predicted_pct / actual_pct if actual_pct != 0 else np.nan

        if sign_match:
            agreement = f"Sign match; model captures {magnitude_ratio * 100:.0f}% of observed magnitude"
        else:
            agreement = "Sign mismatch"

        rank, n_years = actual_pct_rank(df_crop, country, event_year)
        rank_str = f"{rank} of {n_years} years (1=worst)" if rank is not None else "n/a"

        # Performance score for sorting: sign matches ranked by how close the
        # magnitude ratio is to 1 (best); sign mismatches always rank worst.
        log_ratio_dev = abs(np.log(magnitude_ratio)) if (sign_match and magnitude_ratio > 0) else np.inf

        results.append({
            "Event": e["event"],
            "Country": country,
            "Crop": crop,
            "Year": event_year,
            "Actual FAO yield anomaly": f"{actual_pct:.1f}%",
            "Model-predicted dy": f"{predicted_pct:.1f}%",
            "Agreement (sign/magnitude)": agreement,
            "_actual_rank": rank_str,
            "_fold_OOS_R2": round(fold_r2, 4),
            "_sign_mismatch": not sign_match,
            "_log_ratio_dev": log_ratio_dev,
        })

"""
OUTPUT
"""
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by=["_sign_mismatch", "_log_ratio_dev"]).reset_index(drop=True)

table_cols = ["Event", "Country", "Crop", "Year", "Actual FAO yield anomaly",
              "Model-predicted dy", "Agreement (sign/magnitude)"]

print()
print("=" * 100)

print("DISASTER-YEAR VALIDATION: leave-one-year-out (LOYO) out-of-sample predictions")
print(f"Model: {pred_str}, spei{spei_month}. Held-out year excluded GLOBALLY from training for every country.")
print("Rows ordered best -> worst performance (sign match first, then magnitude ratio closest to 1).")
print("=" * 100)
print(df_results[table_cols].to_string(index=False))

"""
OPTIONAL PLOT
"""
if plot_flag:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = df_results["Event"] + " (" + df_results["Country"] + ")"
    x = np.arange(len(df_results))
    width = 0.35

    actual_vals = df_results["Actual FAO yield anomaly"].str.rstrip('%').astype(float)
    predicted_vals = df_results["Model-predicted dy"].str.rstrip('%').astype(float)

    ax.bar(x - width / 2, actual_vals, width, label="Actual FAO yield anomaly")
    ax.bar(x + width / 2, predicted_vals, width, label="Model-predicted dy (OOS)")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Yield deviation [%]")
    ax.legend()
    fig.tight_layout()
    plt.show()

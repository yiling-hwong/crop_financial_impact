# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""

import sys
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import pickle
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('../')
import utils

"""
PARAMETERS
"""
crop = "maize"  # maize, wheat, soy, all
start_year = 2007
end_year = 2019
pred_str = "t3s3"  # t3s3, t2s2_int, t3s3_irr (incl.irrigation interaction)
spei_month = "03" #03,06,09,12
spei_str = f"_spei{spei_month}"

do_cv_flag = True
print_model_summary_flag = False
save_model_flag = False

if pred_str == "t3s3_irr":
    irr_str = "IRR_"
else:
    irr_str = ""

root_dir = f'../../data'
input_hist_file = f"{root_dir}/historical/linregress_inputs/{irr_str}INPUT_HISTORICAL{spei_str}.csv"
output_model_file = f"{root_dir}/models/model_{crop}_{pred_str}{spei_str}.pickle"

"""
X AND Y VARIABLES
"""
y_var = "log_yield"

if pred_str == "t3s3":
    predictors = ["tmax", "tmax2", "tmax3", "spei", "spei2", "spei3"]
if pred_str == "t3s3_irr":
    predictors = ["tmax", "tmax2", "tmax3", "spei", "spei2", "spei3"] + ['tmax_xix_irr', 'tmax2_xix_irr', 'tmax3_xix_irr', 'spei_xix_irr', 'spei2_xix_irr', 'spei3_xix_irr']
if pred_str == "t2s2_int":
    predictors = ["tmax", "tmax2", "spei", "spei2", "tmax_xix_spei", "tmax2_xix_spei",
                  "tmax_xix_spei2", "tmax2_xix_spei2"]


print ()
print (f"Crop: {crop}")
print("Predictors:", predictors)

"""
READ AND PROCESS INPUT DATA
"""
df = pd.read_csv(input_hist_file)

dfs = []

if crop == "all":
    crops = ["maize", "wheat", "soy"]
    crop_str = "all"
else:
    crops = [crop]
    crop_str = crop

for crop in crops:
    df_crop = df[df["crop"] == crop]

    years = [x for x in range(start_year, end_year + 1)]
    df_crop = df_crop[df_crop["year"].isin(years)]

    df_crop['log_yield'] = np.log(df_crop["yield"])

    dfs.append(df_crop)

df_crops = pd.concat(dfs, ignore_index=True)
df_crops.rename(columns={'yield': 'y_ield'}, inplace=True)
print("NUM country-year observations:", df_crops.shape[0])

"""
FIT PANEL REGRESSION MODEL
"""
df_panel = df_crops.copy()
x_vars = predictors

print()
print("################## Fitting PanelOLS model ##################")
print("---------------FITTING MODEL---------------")
print()

df_panel['year_numeric'] = df_panel['year'].astype('int')
df_panel['country_id'] = df_panel['country'].astype('category')
df_panel['country_crop'] = df_panel['country'] + '_' + df_panel['crop']
df_panel['country_crop_id'] = df_panel['country_crop'].astype('category')

x_vars_panel = x_vars
x_vars_str = ' + '.join([f"{var}" for var in x_vars_panel])

if crop_str == "all":
    df_panel = df_panel.set_index(['country_crop', 'year'])
    x_vars_all = x_vars_panel + ['year_numeric'] + ['country_crop_id']
    formula = f"{y_var} ~ 1 + {x_vars_str} + C(country_crop_id):year_numeric + EntityEffects"
else:
    df_panel = df_panel.set_index(['country', 'year'])
    x_vars_all = x_vars_panel + ['year_numeric'] + ['country_id']
    formula = f"{y_var} ~ 1 + {x_vars_str} + C(country_id):year_numeric + EntityEffects"

tmp1 = df_panel[x_vars_all]
tmp2 = df_panel[y_var]
df_panel = pd.concat([tmp1, tmp2], axis=1)

print("Formula:", formula)
model_panel = PanelOLS.from_formula(formula, data=df_panel, check_rank=False,
                                    drop_absorbed=True).fit()

if print_model_summary_flag == True:
    print ()
    print ("Model summary:")
    summary_lines = str(model_panel.summary).split("\n")
    start = [i for i, l in enumerate(summary_lines) if "Parameter  Std. Err." in l][0]
    keep_until = start + 2 + len(predictors) + 1  # header line + separator + 7 parameters
    summary_short = "\n".join(summary_lines[:keep_until])
    print(summary_short)
    print ()

"""
SAVE MODEL
"""
if save_model_flag:
    print ()
    print("Saving trained model...")
    with open(output_model_file, "wb") as f:
        pickle.dump(model_panel, f)
    print ()

#sys.exit()

"""
PRINT RESULTS
"""
len_predictors = len(x_vars_panel) + 1
params = model_panel.params[:len_predictors]
pvals = model_panel.pvalues[:len_predictors]
stderrs = model_panel.std_errors[:len_predictors]
predictor_params = pd.DataFrame({
    'Coefficient': params,
    'p-value': pvals,
    'std_err': stderrs
})

model_stats = pd.DataFrame({
    'Statistic': [
        'R-squared',
        'R-squared inclusive (Overall)',
        'Number of Observations',
        'Degrees of Freedom (Model)',
        'Degrees of Freedom (Residuals)',
        'Included effects',
    ],
    'Value': [
        model_panel.rsquared,
        model_panel.rsquared_inclusive,
        model_panel.nobs,
        model_panel.df_model,
        model_panel.df_resid,
        model_panel.included_effects,
    ]
})

print("-----------")
print(model_stats)
print()
print(predictor_params)
print()

"""
FIT OLS MODEL TO COMPARE 
"""
print()
print("################## Fitting OLS model ##################")
df_ols = df_crops.copy()
df_ols['country'] = df_ols['country'].astype('category')
df_ols['year'] = df_ols['year'].astype('category')
df_ols['year_numeric'] = df_ols['year'].astype('int')

x_vars_str_ols = ' + '.join([f"{var}" for var in predictors])
formula_ols = f"{y_var} ~ 1 + {x_vars_str_ols} + C(country) + C(country):year_numeric"
print("Formula OLS:", formula_ols)
model_ols = smf.ols(formula=formula_ols, data=df_ols, missing='drop').fit()

print("--------------")
print("R2 and adjusted R-squared:", round(model_ols.rsquared, 4),
      round(model_ols.rsquared_adj, 4))
print("--------------")

"""
DO OOS CROSS VALIDATION
=============================================================================
Two CV approaches:
  A) Random 10-fold CV
  B) Leave-One-Year-Out (LOYO) blocked CV
=============================================================================
"""

def detrend_demean_within_fold(df_in, vars_to_transform, train_years, test_years,
                               entity_col='country'):
    """
    For each country, fit a linear trend + mean on TRAINING years only,
    """
    train_rows = []
    test_rows  = []

    for entity, grp in df_in.groupby(entity_col):
        grp_train = grp[grp['year'].isin(train_years)].copy()
        grp_test  = grp[grp['year'].isin(test_years)].copy()

        if len(grp_train) < 2:   # minimum to fit intercept + slope
            continue
        if len(grp_test) == 0:
            continue

        for var in vars_to_transform:
            if var not in grp_train.columns:
                continue

            # --- fit linear trend on training data ---
            trend_fit = smf.ols(f'{var} ~ year_numeric', data=grp_train).fit()

            # Detrend: subtract trend fitted on TRAINING years
            grp_train[f'{var}_dtr'] = (grp_train[var]
                                       - trend_fit.predict(grp_train))
            grp_test[f'{var}_dtr']  = (grp_test[var]
                                       - trend_fit.predict(grp_test))

            # Demean: subtract TRAINING mean of the detrended series
            train_mean = grp_train[f'{var}_dtr'].mean()
            grp_train[f'{var}_dtr_dmn'] = grp_train[f'{var}_dtr'] - train_mean
            grp_test[f'{var}_dtr_dmn']  = grp_test[f'{var}_dtr']  - train_mean

        train_rows.append(grp_train)
        test_rows.append(grp_test)

    df_train_out = pd.concat(train_rows, ignore_index=True)
    df_test_out  = pd.concat(test_rows,  ignore_index=True)
    return df_train_out, df_test_out


def detrend_demean_from_split(df_train_raw, df_test_raw, vars_to_transform,
                               entity_col='country'):
    """
    Detrend and demean per entity using training rows only.
    Works with arbitrary train/test row splits (not year-based).
    """
    train_rows = []
    test_rows  = []

    for entity, grp_train in df_train_raw.groupby(entity_col):
        grp_test  = df_test_raw[df_test_raw[entity_col] == entity].copy()
        grp_train = grp_train.copy()

        if len(grp_train) < 2:   # minimum to fit intercept + slope
            continue
        if len(grp_test) == 0:
            continue

        for var in vars_to_transform:
            if var not in grp_train.columns:
                continue

            trend_fit = smf.ols(f'{var} ~ year_numeric', data=grp_train).fit()

            grp_train[f'{var}_dtr'] = grp_train[var] - trend_fit.predict(grp_train)
            grp_test[f'{var}_dtr']  = grp_test[var]  - trend_fit.predict(grp_test)

            train_mean = grp_train[f'{var}_dtr'].mean()
            grp_train[f'{var}_dtr_dmn'] = grp_train[f'{var}_dtr'] - train_mean
            grp_test[f'{var}_dtr_dmn']  = grp_test[f'{var}_dtr']  - train_mean

        train_rows.append(grp_train)
        test_rows.append(grp_test)

    df_train_out = pd.concat(train_rows, ignore_index=True)
    df_test_out  = pd.concat(test_rows,  ignore_index=True)
    return df_train_out, df_test_out


print()
if do_cv_flag:

    entity_col = 'country' if crop_str != "all" else 'country_crop'

    # ------------------------------------------------------------------
    # Shared raw-data setup (used by both A and B)
    # ------------------------------------------------------------------
    df_cv_raw = pd.read_csv(input_hist_file)
    if crop_str != "all":
        df_cv_raw = df_cv_raw[df_cv_raw["crop"] == crop_str]
    df_cv_raw = df_cv_raw[
        df_cv_raw["year"].isin(range(start_year, end_year + 1))
    ].copy()
    df_cv_raw['log_yield']    = np.log(df_cv_raw['yield'])
    df_cv_raw['year_numeric'] = df_cv_raw['year'].astype(int)
    if crop_str == "all":
        df_cv_raw['country_crop'] = df_cv_raw['country'] + '_' + df_cv_raw['crop']

    # Variance of full-period per-entity detrended log_yield (used as MSE baseline)
    _dtr_resids = []
    for _entity, _grp in df_cv_raw.groupby(entity_col):
        if len(_grp) < 2:
            continue
        _fit = smf.ols('log_yield ~ year_numeric', data=_grp).fit()
        _dtr_resids.append(_grp['log_yield'] - _fit.predict(_grp))
    y_variance = pd.concat(_dtr_resids).var()
    del _dtr_resids

    vars_to_transform = ['log_yield'] + predictors
    all_years         = sorted(df_cv_raw['year'].unique())

    # ==================================================================
    # A) RANDOM 10-FOLD CV
    # ==================================================================
    print("=" * 70)
    print("A) Random 10-fold CV ")
    print("=" * 70)
    print()

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    r2_rnd_all  = []
    mse_rnd_all = []

    df_cv_rnd_raw = df_cv_raw.reset_index(drop=True)

    for i, (train_idx, test_idx) in enumerate(kfold.split(df_cv_rnd_raw)):
        df_train_raw_fold = df_cv_rnd_raw.iloc[train_idx].copy()
        df_test_raw_fold  = df_cv_rnd_raw.iloc[test_idx].copy()

        df_train_fold_r, df_test_fold_r = detrend_demean_from_split(
            df_train_raw_fold, df_test_raw_fold,
            vars_to_transform=vars_to_transform,
            entity_col=entity_col
        )

        if len(df_train_fold_r) == 0 or len(df_test_fold_r) == 0:
            print(f"  Fold {i+1}: skipped (empty fold after detrending)")
            continue

        y_fold_r       = 'log_yield_dtr_dmn'
        preds_fold_r   = [f'{p}_dtr_dmn' for p in predictors]
        formula_fold_r = f"{y_fold_r} ~ 1 + {' + '.join(preds_fold_r)}"

        df_train_fold_r = df_train_fold_r.set_index([entity_col, 'year'])
        df_test_fold_r  = df_test_fold_r.set_index([entity_col, 'year'])

        try:
            model_rnd = PanelOLS.from_formula(formula_fold_r, data=df_train_fold_r).fit()
        except Exception as e:
            print(f"  Fold {i+1}: model fit failed — {e}")
            continue

        X_test_r = df_test_fold_r[preds_fold_r].copy()
        X_test_r.insert(0, 'Intercept', 1.0)
        y_test_r = df_test_fold_r[y_fold_r]

        y_pred_r = model_rnd.predict(X_test_r)
        common_r = y_test_r.index.intersection(y_pred_r.index)
        r2_r     = r2_score(y_test_r.loc[common_r], y_pred_r.loc[common_r])
        mse_r    = mean_squared_error(y_test_r.loc[common_r], y_pred_r.loc[common_r])

        r2_rnd_all.append(r2_r)
        mse_rnd_all.append(mse_r)

    r2_rnd_mean   = np.mean(r2_rnd_all)
    r2_rnd_median = np.median(r2_rnd_all)
    r2_rnd_std    = np.std(r2_rnd_all)
    mse_rnd_mean  = np.mean(mse_rnd_all)

    worst_rnd_idx = np.argmin(r2_rnd_all)
    r2_rand_excl  = np.mean([r for j, r in enumerate(r2_rnd_all) if j != worst_rnd_idx])

    print(f"  N folds              : {len(r2_rnd_all)}")
    print(f"  OOS R² mean ± std    : {r2_rnd_mean:.4f} ± {r2_rnd_std:.4f}")
    print()

    # ==================================================================
    # B) LOYO BLOCKED CV
    # ==================================================================
    print("=" * 70)
    print("B) Leave-One-Year-Out (LOYO) blocked CV")
    print("=" * 70)
    print()

    r2_loyo_all  = []
    mse_loyo_all = []
    fold_results = []

    for fold_idx, test_year in enumerate(all_years):
        train_years = [y for y in all_years if y != test_year]

        df_train_fold, df_test_fold = detrend_demean_within_fold(
            df_cv_raw,
            vars_to_transform=vars_to_transform,
            train_years=train_years,
            test_years=[test_year],
            entity_col=entity_col  # 'country' for single crop, 'country_crop' for pooled
        )

        if len(df_train_fold) == 0 or len(df_test_fold) == 0:
            print(f"  Fold {fold_idx+1} (test year={test_year}): skipped (empty fold)")
            continue

        y_fold       = 'log_yield_dtr_dmn'
        preds_fold   = [f'{p}_dtr_dmn' for p in predictors]
        formula_fold = f"{y_fold} ~ 1 + {' + '.join(preds_fold)}"

        df_train_fold = df_train_fold.set_index([entity_col, 'year'])
        df_test_fold  = df_test_fold.set_index([entity_col, 'year'])

        try:
            model_fold = PanelOLS.from_formula(
                formula_fold, data=df_train_fold
            ).fit()
        except Exception as e:
            print(f"  Fold {fold_idx+1} (test year={test_year}): model fit failed — {e}")
            continue

        X_test_fold = df_test_fold[preds_fold].copy()
        X_test_fold.insert(0, 'Intercept', 1.0)
        y_test_fold = df_test_fold[y_fold]

        y_pred     = model_fold.predict(X_test_fold)
        common_idx = y_test_fold.index.intersection(y_pred.index)

        if len(common_idx) == 0:
            print(f"  Fold {fold_idx+1} (test year={test_year}): no common index after predict")
            continue

        r2_oos  = r2_score(y_test_fold.loc[common_idx], y_pred.loc[common_idx])
        mse_val = mean_squared_error(y_test_fold.loc[common_idx], y_pred.loc[common_idx])

        r2_loyo_all.append(r2_oos)
        mse_loyo_all.append(mse_val)
        fold_results.append({'test_year': test_year,
                             'n_test':    len(common_idx),
                             'OOS_R2':    round(r2_oos, 4),
                             'MSE':       round(mse_val, 6)})

    r2_loyo_std    = np.std(r2_loyo_all)
    out_year = fold_results[np.argmin([f['OOS_R2'] for f in fold_results])]['test_year']
    r2_loyo_mean     = np.mean([f['OOS_R2'] for f in fold_results
                                if f['test_year'] != out_year])
    n_pos            = sum(r > 0 for r in r2_loyo_all)

    r2_loyo_top = sum(sorted(r2_loyo_all, reverse=True)[:10]) / 10

    print(f"  N folds                          : {len(r2_loyo_all)}")
    print(f"  OOS R² mean ± std                : {r2_loyo_mean:.4f} ± {r2_loyo_std:.4f}")
    print()

    # ==================================================================
    # COMBINED SUMMARY
    # ==================================================================
    print("=" * 70)
    print("COMBINED CV SUMMARY")
    print("=" * 70)
    print()
    print(f"  Crop & model    : {crop_str}  {pred_str}")
    print(f"  Overall adj. R² : {round(model_ols.rsquared_adj, 4)}")
    print(f"  Variance of detrended log_yield : {round(y_variance, 4)}")
    print()

    header = (f"  {'Method':<45} {'Mean R²':>8}"
              f"{'Std R²':>8}")
    print(header)
    print("  " + "-" * 83)
    print(f"  {'A) Random 10-fold CV':<45} "
          f"{r2_rnd_mean:>8.4f} "
          f"{r2_rnd_std:>8.4f} ")
    print(f"  {'B) LOYO blocked CV':<45} "
          f"{r2_loyo_mean:>8.4f} "
          f"{r2_loyo_std:>8.4f} ")
    print()


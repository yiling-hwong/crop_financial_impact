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
crop = "maize" # maize, wheat, soy, all
start_year = 2007
end_year = 2019
pred_str = "t3s3" #t3s3, t2s2_int

do_cv_flag = True  # 10-fold cross-validation (OOS within-R2)
save_model_flag = False

root_dir = f"../../data"
input_hist_file = f"{root_dir}/historical/linregress_inputs/INPUT_HISTORICAL.csv"
input_hist_file_dtr = f"{root_dir}/historical/linregress_inputs/INPUT_HISTORICAL_DETRENDED_DEMEANED.csv"
output_model_file = f"{root_dir}/models/model_{crop}_{pred_str}.pickle"

"""
X AND Y VARIABLES
"""
y_var = "log_yield"

if pred_str == "t2s2_int":
    predictors = ["tmax", "tmax2", "spei", "spei2", "tmax_xix_spei", "tmax2_xix_spei","tmax_xix_spei2", "tmax2_xix_spei2"]
if pred_str == "t3s3":
    predictors = ["tmax", "tmax2", "tmax3", "spei", "spei2", "spei3"]

y_var_dtr = f"{y_var}_dtr_dmn"
predictors_dtr = [f"{x}_dtr_dmn" for x in predictors]

print ()
print("Crops:", crop)
print("Predictors:", predictors)

"""
READ AND PROCESS INPUT DATA
"""
df = pd.read_csv(input_hist_file)
df_dtr = pd.read_csv(input_hist_file_dtr)

dfs = []
dfs_dtr = []

if crop == "all":
    crops = ["maize", "wheat", "soy"]
    crop_str = "all"
else:
    crops = [crop]
    crop_str = crop

for crop in crops:

    df_crop = df[df["crop"] == crop]
    df_crop_dtr = df_dtr[df_dtr["crop"] == crop]

    # Get historical (baseline) years
    years = [x for x in range(start_year, end_year + 1)]
    df_crop = df_crop[df_crop["year"].isin(years)]
    df_crop_dtr = df_crop_dtr[df_crop_dtr["year"].isin(years)]

    # Log-transform the yield variable
    df_crop['log_yield'] = np.log(df_crop["yield"])
    df_crop_dtr['log_yield'] = np.log(df_crop_dtr["yield"])

    dfs.append(df_crop)
    dfs_dtr.append(df_crop_dtr)

# Combine all crop data
df_crops = pd.concat(dfs, ignore_index=True)
df_crops_dtr = pd.concat(dfs_dtr, ignore_index=True)
df_crops.rename(columns={'yield': 'y_ield'}, inplace=True)  # yield is a reserved Python keyword
df_crops_dtr.rename(columns={'yield': 'y_ield'}, inplace=True)
print("NUM country-year observations:", df_crops.shape[0], df_crops_dtr.shape[0])

"""
FIT PANEL REGRESSION MODEL
"""
df_panel = pd.DataFrame()
df_panel = df_crops.copy()
x_vars = predictors

df_panel_dtr = pd.DataFrame()
df_panel_dtr = df_crops_dtr.copy()
x_vars_dtr = predictors_dtr

print()
print("################## Fitting PanelOLS model ##################")

print("---------------FITTING MODEL---------------")
print()

# Create an interaction term between country and year
df_panel['year_numeric'] = df_panel['year'].astype('int')
df_panel['country_id'] = df_panel['country'].astype('category')
df_panel['country_crop'] = df_panel['country'] + '_' + df_panel['crop']
df_panel['country_crop_id'] = df_panel['country_crop'].astype('category')

x_vars_panel = x_vars
x_vars_str = ' + '.join([f"{var}" for var in x_vars_panel])

# Set the multi-index for panel data with 'country' (first term) as the entity and 'year' (second term) as the time
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
model_panel = PanelOLS.from_formula(formula, data=df_panel, check_rank=False, drop_absorbed=True).fit()

################
# USING DETRENDED VARS (FOR OOS R2 CALCULATION)
################
if crop_str == "all":
    df_panel_dtr['country_crop'] = df_panel_dtr['country'] + '_' + df_panel_dtr['crop']
    df_panel_dtr = df_panel_dtr.set_index(['country_crop', 'year'])
else:
    df_panel_dtr = df_panel_dtr.set_index(['country', 'year'])

x_vars_panel_dtr = x_vars_dtr
x_vars_str_dtr = ' + '.join([f"{var}" for var in x_vars_panel_dtr])
tmp1 = df_panel_dtr[x_vars_panel_dtr]
tmp2 = df_panel_dtr[y_var_dtr]
df_panel_dtr = pd.concat([tmp1, tmp2], axis=1)
formula_dtr = f"{y_var_dtr} ~ 1 + {x_vars_str_dtr}"
print("Formula:", formula_dtr)

model_panel_dtr = PanelOLS.from_formula(formula_dtr, data=df_panel_dtr).fit()

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

len_predictors_dtr = len(x_vars_panel_dtr) + 1
params_dtr = model_panel_dtr.params[:len_predictors_dtr]
params_dtr = model_panel_dtr.params.apply(lambda x: f'{x:.6f}')
pvals_dtr = model_panel_dtr.pvalues[:len_predictors_dtr]
stderrs_dtr = model_panel_dtr.std_errors[:len_predictors_dtr]
predictor_params_dtr = pd.DataFrame({
    'Coefficient': params_dtr,
    'p-value': pvals_dtr,
    'std_err': stderrs_dtr
})


# Create a DataFrame with model statistics
model_stats = pd.DataFrame({
    'Statistic': [
        'R-squared',
        'R-squared inclusive (Overall)',
        'Within-sample R-squared',
        'R-squared of detrended model',
        'Number of Observations',
        'Degrees of Freedom (Model)',
        'Degrees of Freedom (Residuals)',
        'Included effects',
    ],
    'Value': [
        model_panel.rsquared,
        model_panel.rsquared_inclusive,
        model_panel_dtr.rsquared_within,
        model_panel_dtr.rsquared_inclusive,
        model_panel.nobs,
        model_panel.df_model,
        model_panel.df_resid,
        model_panel.included_effects,
    ]
})

print("-----------")
print(model_stats)
print()
#print("-----------original vars:")
print(predictor_params)
print()
# print("-----------detrended vars:")
# print(predictor_params_dtr)

"""
FIT OLS MODEL TO COMPARE
"""

print()
print("################## Fitting OLS model ##################")
df_ols = pd.DataFrame()
df_ols = df_crops.copy()
df_ols['country'] = df_ols['country'].astype('category')
df_ols['year'] = df_ols['year'].astype('category')
df_ols['year_numeric'] = df_ols['year'].astype('int')

x_vars_str_ols = ' + '.join([f"{var}" for var in predictors])
formula_ols = f"{y_var} ~ 1 + {x_vars_str_ols} + C(country) + C(country):year_numeric"  # IMPORTANT: this returns same result as option 3
print("Formula OLS:", formula_ols)
model_ols = smf.ols(formula=formula_ols, data=df_ols, missing='drop').fit()

# Print the regression results
print("--------------")
print("R2 and adjusted R-squared:", round(model_ols.rsquared, 4), round(model_ols.rsquared_adj, 4))
print("--------------")
# print(model_ols.summary())

# Initialize a DataFrame to hold the results
coefficients_df = pd.DataFrame({
    'Coefficient': model_ols.params,
    'P-value': model_ols.pvalues,
    'std_err': model_ols.bse
})

# Filter for selected predictors (and their categorical equivalents)
predictors_ols = ["Intercept"] + predictors
coefficients_selected = coefficients_df.loc[coefficients_df.index.isin(predictors_ols)]

# Print the coefficients and p-values for selected predictors
# print("\nCoefficients and p-values for predictors:\n")
# print(coefficients_selected)

"""
DO OOS CROSS VALIDATION
"""
print ()
if do_cv_flag == True:

    print("################################")
    print("Do 10-fold cross validation ...")

    df_cv_dtr = pd.DataFrame()
    df_cv_dtr = df_crops_dtr.copy()
    df_cv_dtr['country_id'] = df_cv_dtr['country'].astype('category')
    df_cv_dtr = df_cv_dtr.set_index(['country', 'year'])

    x_vars_dtr = predictors_dtr
    x_vars_str_dtr = ' + '.join([f"{var}" for var in x_vars_dtr])
    formula_dtr = f"{y_var_dtr} ~ 1 + {x_vars_str_dtr}"

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    X = df_cv_dtr[x_vars_dtr]
    y = df_cv_dtr[y_var_dtr]

    # Initialize an empty list to store scores
    r2_scores_train = []
    r2_scores_test = []
    r2_oos_all = []  # out-of-sample CV
    mse_scores = []

    # Perform k-fold cross-validation
    for i, (train_index, test_index) in enumerate(kfold.split(df_cv_dtr)):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_test = pd.concat([pd.Series(1, index=X_test.index, name="Intercept"), X_test], axis=1)

        # Combine training data back into a panel structure
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        # Fit the PanelOLS model
        model_cv_train = PanelOLS.from_formula(formula_dtr, data=train_data).fit()
        model_cv_test = PanelOLS.from_formula(formula_dtr, data=test_data).fit()

        y_pred = model_cv_train.predict(X_test)

        # Get scores
        r2_train = model_cv_train.rsquared
        r2_test = model_cv_test.rsquared
        r2_oos = r2_score(y_test, y_pred)
        mse_score = mean_squared_error(y_test, y_pred)

        r2_scores_train.append(r2_train)
        r2_scores_test.append(r2_test)
        r2_oos_all.append(r2_oos)
        mse_scores.append(mse_score)

    # MEAN AND AVG
    r2_train_mean = np.mean(r2_scores_train)
    r2_test_mean = np.mean(r2_scores_test)
    r2_oos_mean = np.mean(r2_oos_all)
    mse_mean = np.mean(mse_scores)

    r2_train_std = np.std(r2_scores_train)
    r2_test_std = np.std(r2_scores_test)
    r2_oos_std = np.std(r2_oos_all)
    mse_std = np.std(mse_scores)

    # A good MSE should be lower the the variance of outcome var, if it is close to or higher than bad
    y_variance = df_cv_dtr[y_var_dtr].var()

    print()
    print("In-sample R-squared:", round(model_panel_dtr.rsquared_inclusive, 4))
    print("MEAN R-squared score for OOS, TRAIN & TEST, MSE:", round(r2_oos_mean, 4), round(r2_train_mean, 4),
          round(r2_test_mean, 4), round(mse_mean, 4))
    print("STD R-squared score for OOS, TRAIN and TEST, MSE:", round(r2_oos_std, 4), round(r2_train_std, 4),
          round(r2_test_std, 4), round(mse_std, 4))

    print ()
    print("################################")
    print ("Crop & model:",crop_str, pred_str)
    print("Number of observations:", model_panel.nobs)
    print ("Overall adj. R2 and 10-fold CV R2:", round(model_ols.rsquared_adj, 4), round(r2_oos_mean, 4))
    print("Variance of y_var (MSE should be lower):", round(y_variance, 4))
    if mse_mean < y_variance:
        print("MSE smaller than y_variance, GOOD!")
    else:
        print("MSE larger than y_variance, not good.")

    print("################################")
    print ()

"""
SAVE MODEL
"""
if save_model_flag == True:

    print("Saving trained model...")
    with open(output_model_file, "wb") as f:
        pickle.dump(model_panel, f)
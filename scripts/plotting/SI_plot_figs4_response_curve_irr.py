# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
import os
import sys
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
import pickle
warnings.filterwarnings(action='ignore')

sys.path.append('..')
import utils

"""
PARAMETERS
"""
irr_factors = [0,0.3] #unirrigated, irrigated

#-----------PLOTTING PARAMETERS------------------------------------------
color_tmax = "purple"
color_spei = "teal"
xmin_t, xmax_t = 10, 35
ymin_t, ymax_t = -75, 40
xmin_s, xmax_s = -3, 3
ymin_s, ymax_s = -30, 10
#------------------------------------------------------------------------

root_dir = f'../../data'
input_hist_file = f"{root_dir}/historical/linregress_inputs/IRR_INPUT_HISTORICAL_spei03.csv"
model_dir = f"{root_dir}/models"

predictors = ["tmax", "tmax2", "tmax3", "spei", "spei2", "spei3",
              'tmax_xix_irr', 'tmax2_xix_irr', 'tmax3_xix_irr',
              'spei_xix_irr', 'spei2_xix_irr', 'spei3_xix_irr']

# ── Response functions (anchored to reference, conditioned on I) ──
def tmax_response(T, T_ref, I=0.0):
    yfut  = (b1t*T      + b2t*T**2      + b3t*T**3
           + b1ti*(T*I) + b2ti*(T**2*I) + b3ti*(T**3*I))
    ybase = (b1t*T_ref      + b2t*T_ref**2      + b3t*T_ref**3
           + b1ti*(T_ref*I) + b2ti*(T_ref**2*I) + b3ti*(T_ref**3*I))
    return (np.exp(yfut - ybase) - 1) * 100

def spei_response(S, S_ref, I=0.0):
    yfut  = (b1s*S      + b2s*S**2      + b3s*S**3
           + b1si*(S*I) + b2si*(S**2*I) + b3si*(S**3*I))
    ybase = (b1s*S_ref      + b2s*S_ref**2      + b3s*S_ref**3
           + b1si*(S_ref*I) + b2si*(S_ref**2*I) + b3si*(S_ref**3*I))
    return (np.exp(yfut - ybase) - 1) * 100

# ── CI via delta method through exp transform (conditioned on I) ──
def tmax_ci(T, T_ref, I=0.0, z=1.96):
    se = []
    for t in T:
        d = np.array([t - T_ref, t**2 - T_ref**2, t**3 - T_ref**3])
        se.append(np.sqrt(d @ cov_t_base @ d))
    se = np.array(se)
    yfut  = b1t*T + b2t*T**2 + b3t*T**3 + b1ti*(T*I) + b2ti*(T**2*I) + b3ti*(T**3*I)
    ybase = (b1t*T_ref + b2t*T_ref**2 + b3t*T_ref**3
           + b1ti*(T_ref*I) + b2ti*(T_ref**2*I) + b3ti*(T_ref**3*I))
    return z * np.exp(yfut - ybase) * se * 100

def spei_ci(S, S_ref, I=0.0, z=1.96):
    se = []
    for s in S:
        d = np.array([s - S_ref, s**2 - S_ref**2, s**3 - S_ref**3])
        se.append(np.sqrt(d @ cov_s_base @ d))
    se = np.array(se)
    yfut  = b1s*S + b2s*S**2 + b3s*S**3 + b1si*(S*I) + b2si*(S**2*I) + b3si*(S**3*I)
    ybase = (b1s*S_ref + b2s*S_ref**2 + b3s*S_ref**3
           + b1si*(S_ref*I) + b2si*(S_ref**2*I) + b3si*(S_ref**3*I))
    return z * np.exp(yfut - ybase) * se * 100

# ── Build full 6×6 covariance matrices (zeros for absent terms) ────
def build_cov(term_list):
    n = len(term_list)
    cov_full = np.zeros((n, n))
    existing = [t for t in term_list if t in model.cov.index]
    if existing:
        idx = [term_list.index(t) for t in existing]
        sub = model.cov.loc[existing, existing].values
        for x, ci in enumerate(idx):
            for y, cj in enumerate(idx):
                cov_full[ci, cj] = sub[x, y]
    return cov_full

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(input_hist_file)

# ─────────────────────────────────────────────
# 2. LOAD PICKLE MODEL
# ─────────────────────────────────────────────
model_file = f"{model_dir}/model_all_t3s3_irr_spei03.pickle"
with open(model_file, "rb") as f:
    model = pickle.load(f)

# ─────────────────────────────────────────────
# 3. DEFINE CROPS AND COLUMN NAMES
# ─────────────────────────────────────────────

crop_col     = 'crop'        # column identifying crop type
tmax_col     = 'tmax'        # absolute growing season tmax
spei_col     = 'spei'        # absolute SPEI

tmax_terms     = ['tmax',         'tmax2',         'tmax3'        ]
spei_terms     = ['spei',         'spei2',         'spei3'        ]
tmax_irr_terms = ['tmax_xix_irr', 'tmax2_xix_irr', 'tmax3_xix_irr']
spei_irr_terms = ['spei_xix_irr', 'spei2_xix_irr', 'spei3_xix_irr']

all_tmax_terms = tmax_terms + tmax_irr_terms
all_spei_terms = spei_terms + spei_irr_terms

# ─────────────────────────────────────────────
# 4. IRRIGATION PLOTTING SCENARIOS
# ─────────────────────────────────────────────
irr_scenarios = irr_factors
linestyles    = ['-', '--']
irr_labels    = [f'Unirrigated',
                 f'Irrigated']

# ─────────────────────────────────────────────
# 5. PLOT
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 3))

title_fontsize = 12
label_fontsize = 11
tick_fontsize = 11


# ── Subset data for this crop ──────────────────────────────────────
if crop_col in df.columns:
    df_crop = df[df[crop_col] == "all"].copy()
else:
    df_crop = df.copy()

# ── Get coefficients from model (default to 0 if term absent) ─────
params = model.params  # assumes statsmodels-style model

def get_param(name):
    return params[name] if name in params.index else 0.0

b1t  = get_param(tmax_terms[0]);  b2t  = get_param(tmax_terms[1]);  b3t  = get_param(tmax_terms[2])
b1s  = get_param(spei_terms[0]);  b2s  = get_param(spei_terms[1]);  b3s  = get_param(spei_terms[2])
b1ti = get_param(tmax_irr_terms[0]); b2ti = get_param(tmax_irr_terms[1]); b3ti = get_param(tmax_irr_terms[2])
b1si = get_param(spei_irr_terms[0]); b2si = get_param(spei_irr_terms[1]); b3si = get_param(spei_irr_terms[2])

print(f"  Tmax coeffs    : {b1t:.4f}, {b2t:.4f}, {b3t:.4f}")
print(f"  SPEI coeffs    : {b1s:.4f}, {b2s:.4f}, {b3s:.4f}")
print(f"  Tmax×Irr coeffs: {b1ti:.4f}, {b2ti:.4f}, {b3ti:.4f}")
print(f"  SPEI×Irr coeffs: {b1si:.4f}, {b2si:.4f}, {b3si:.4f}")

cov_t_full = build_cov(all_tmax_terms)
cov_s_full = build_cov(all_spei_terms)

cov_t_base = cov_t_full[:3, :3]
cov_s_base = cov_s_full[:3, :3]

# ── Compute mean (reference point) and data range ─────────────────
T_mean  = df_crop[tmax_col].mean()
S_mean  = df_crop[spei_col].mean()

T_p01   = df_crop[tmax_col].quantile(0.01)
T_p99   = df_crop[tmax_col].quantile(0.99)
S_p01   = df_crop[spei_col].quantile(0.01)
S_p99   = df_crop[spei_col].quantile(0.99)

#reference and sweep range
T_mean  = 293.0
S_mean  = 0.0
T_range = np.linspace(275, 320, 500)
S_range = np.linspace(-3.1, 3.1, 500)

# ── Convert T range to Celsius for plotting ────────────────────────
T_range_C = T_range - 273
T_mean_C  = T_mean  - 273

# labels for single-curve case
scenario_labels = irr_labels

# ── Plot tmax response ─────────────────────────────────────────────
ax    = axes[0]
color = color_tmax

for I, ls, lbl in zip(irr_scenarios, linestyles, scenario_labels):
    yield_T = tmax_response(T_range, T_mean, I=I)
    ci_T    = tmax_ci(T_range, T_mean, I=I)
    ax.plot(T_range_C, yield_T, color=color, linewidth=2.0, linestyle=ls, label=lbl)
    ax.fill_between(T_range_C, yield_T - ci_T, yield_T + ci_T,alpha=0.1, color=color)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax.axvline(T_mean_C, color='dimgrey', linewidth=1.8, linestyle=':', alpha=1.0)
ax.set_xlim(xmin_t, xmax_t)
ax.set_ylim(ymin_t, ymax_t)
ax.set_xlabel(r'Growing season $T_{\mathrm{max}}$ (°C)', fontsize=label_fontsize, labelpad=10)
ax.set_ylabel('Yield change (%)', fontsize=label_fontsize)
ax.set_title(f'Temperature response', fontsize=title_fontsize)
ax.legend(fontsize=label_fontsize, loc="lower center")
ax.grid(True, alpha=0.25)

# ── Plot SPEI response ─────────────────────────────────────────────
ax    = axes[1]
color = color_spei

for I, ls, lbl in zip(irr_scenarios, linestyles, scenario_labels):
    yield_S = spei_response(S_range, S_mean, I=I)
    ci_S    = spei_ci(S_range, S_mean, I=I)
    ax.plot(S_range, yield_S, color=color, linewidth=2.0, linestyle=ls, label=lbl)
    ax.fill_between(S_range, yield_S - ci_S, yield_S + ci_S,alpha=0.1, color=color)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax.axvline(S_mean, color='dimgrey', linewidth=1.8, linestyle=':', alpha=1.0)
ax.set_xlim(xmin_s, xmax_s)
ax.set_ylim(ymin_s, ymax_s)
ax.set_xlabel(f'SPEI', fontsize=label_fontsize, labelpad=10)
ax.set_ylabel('Yield change (%)', fontsize=label_fontsize)
ax.set_title(f'SPEI response', fontsize=title_fontsize)
ax.legend(fontsize=label_fontsize,loc="lower center")
ax.grid(True, alpha=0.25)

plt.tight_layout(w_pad=2)
plt.show()
# -*- coding: utf-8 -*-
"""
@author: Yi-Ling Hwong
"""
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

sys.path.append('..')
import utils

"""
PARAMETERS
"""
crops = ["maize", "wheat", "soy"]
pred_str = "t3s3"
pred_str_co2 = f"{pred_str}_co2"
spei_month = "03"
weight_opt = "weighted"  # "weighted", "simple"

start_year_hist = 1974
end_year_hist = 2004
start_year_fut = 1990
end_year_fut = 2019
start_year_plot = 2007
end_year_plot = 2019

root_dir = "../../data"
hist_dir = f"{root_dir}/historical/linregress_outputs"

smooth_method = "numpy"
np_mode = "full"
quantile_low = 0.25
quantile_high = 0.75

ipcc_colors = {"ldc": "crimson", "developing": "goldenrod", "developed": "dodgerblue"}
ipcc_labels = {"ldc": "LDC", "developing": "Developing", "developed": "Developed"}

ar6_region = "region_ar6_dev"
regions = ["ldc", "developing", "developed"]

impacts = ["dy", "dp", "degdp", "de"]
unit_div = {"dy": 1, "dp": 1e6, "degdp": 1, "de": 1e9}
ylabels = {
    "dy":    "Yield change [%]",
    "dp":    "Production change [Mt]",
    "degdp": "GDP loss [%]",
    "de":    "Economic loss [B US$]",
}

row_configs = [
    {"label": "ISIMIP3A", "target_dir": "isimip3a", "file_strs": ["isimip3a"], "style": "ipcc"},
]

ci_offsets_ipcc = {"global": -0.4, "ldc": -0.2, "developing": 0.0, "developed": 0.2}

title_fontsize  = 14
label_fontsize = 12
legend_fontsize = 12
tick_fontsize   = 12
year_ticks      = [2008, 2010, 2012, 2014, 2016, 2018]

ylim_top_dy = 0.4
ylim_bottom_dy = -6.0
ylim_top_dp = 10.0
ylim_bottom_dp = -100

ylim_top_degdp = 0.01
ylim_bottom_degdp = -0.075
ylim_top_de = 3.0
ylim_bottom_de = -28

ylim_top    = {"dy": ylim_top_dy,    "dp": ylim_top_dp,    "degdp": ylim_top_degdp,    "de": ylim_top_de}
ylim_bottom = {"dy": ylim_bottom_dy, "dp": ylim_bottom_dp, "degdp": ylim_bottom_degdp, "de": ylim_bottom_de}

"""
LOAD DATA
"""
hist_file = f"{root_dir}/historical/linregress_inputs/INPUT_HISTORICAL_DETRENDED_DEMEANED_spei{spei_month}.csv"
valid_countries = pd.read_csv(hist_file)["country"].unique()

df_ipcc = utils.get_ipcc_region_df()
df_ipcc = df_ipcc[["Country", ar6_region]].rename(columns={"Country": "country"})

prod_weights = utils.load_fao_prod_weights(root_dir, spei_month)
gdp_weights  = utils.load_gdp_weights(root_dir)

years_ts     = list(range(start_year_fut, end_year_fut + 1))
years_ts_str = [str(y) for y in years_ts]
years_plot   = list(range(start_year_plot, end_year_plot + 1))
plot_start_idx = years_ts.index(start_year_plot)
plot_end_idx   = years_ts.index(end_year_plot) + 1
last_yr_str    = str(end_year_plot)

"""
HELPER FUNCTIONS
"""
def load_combined(target_dir, file_str, impact, pred):
    dfs = []
    for crop in crops:
        fname = (f"{hist_dir}/{crop}/{target_dir}/{impact}_{file_str}_{pred}"
                 f"_hist{start_year_hist}-{end_year_hist}"
                 f"_fut{start_year_fut}-{end_year_fut}_spei{spei_month}.csv")
        try:
            df = pd.read_csv(fname)
            df = df[df["country"].isin(valid_countries)]
            df["crop"] = crop
            dfs.append(df)
        except FileNotFoundError:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else None


def _dy_weighted_co2(df_dy_co2, df_dy_base, df_dp_base, year_cols):
    """
    Production-weighted dy for CO2 scenario
    """
    dp_c      = df_dp_base.groupby("country")[year_cols].sum()
    dy_c_base = df_dy_base.groupby("country")[year_cols].sum()
    dy_c_co2  = df_dy_co2.groupby("country")[year_cols].sum()

    dp_mean = dp_c.mean(axis=1)
    dy_mean = dy_c_base.mean(axis=1)
    prod_w  = (dp_mean / dy_mean).replace([np.inf, -np.inf], np.nan)
    prod_w  = prod_w[prod_w > 0]

    common = prod_w.index.intersection(dy_c_co2.index)
    den    = prod_w[common].sum()
    result = []
    for yr in year_cols:
        result.append(
            float((prod_w[common] * dy_c_co2.loc[common, yr]).sum() / den)
            if den != 0 else 0.0
        )
    return np.nan_to_num(np.array(result), nan=0)


def compute_ci(df, impact, unit, final_val):
    """Quantile CI from flat country-crop values at last year, centered on final_val."""
    vals = df[last_yr_str].dropna() / unit
    if len(vals) == 0:
        return final_val, final_val
    mean_v  = vals.mean()
    ci_low  = final_val - abs(mean_v - vals.quantile(quantile_low))
    ci_high = final_val + abs(mean_v - vals.quantile(quantile_high))
    return ci_low, ci_high


def smooth(vals,window_size):
    return utils.get_smoothed_values(vals, window_size=window_size,
                                     method=smooth_method, np_mode=np_mode)


unit_labels = {"dy": "%", "dp": "Mt", "degdp": "%", "de": "B US$"}
agg_method  = {"dy": "mean", "dp": "sum", "degdp": "mean", "de": "sum"}
ts_store = {}  # impact -> {g, g_co2, reg, reg_co2}

"""
PLOT
"""
label_alphabets = utils.generate_alphabet_list(4, option="lower")
fig, axes = plt.subplots(2, 2, figsize=(9, 6))

for row_idx, row_cfg in enumerate(row_configs):
    for col_idx, impact in enumerate(impacts):
        ax   = axes[col_idx // 2, col_idx % 2]
        unit = unit_div[impact]
        target_dir = row_cfg["target_dir"]

        if row_cfg["style"] == "ipcc":
            file_str = row_cfg["file_strs"][0]
            df     = load_combined(target_dir, file_str, impact, pred_str)
            df_co2 = load_combined(target_dir, file_str, impact, pred_str_co2)
            if df is None or df_co2 is None:
                continue

            df_m     = df.merge(df_ipcc, on="country", how="left")
            df_m_co2 = df_co2.merge(df_ipcc, on="country", how="left")

            if impact == "degdp":
                g     = utils.compute_degdp_weighted(df, years_ts_str, gdp_weights=gdp_weights, weight_opt=weight_opt) / unit
                g_co2 = utils.compute_degdp_weighted(df_co2, years_ts_str, gdp_weights=gdp_weights, weight_opt=weight_opt) / unit
                reg     = {r: utils.compute_degdp_weighted(df_m[df_m[ar6_region] == r], years_ts_str, gdp_weights=gdp_weights, weight_opt=weight_opt) for r in regions}
                reg_co2 = {r: utils.compute_degdp_weighted(df_m_co2[df_m_co2[ar6_region] == r], years_ts_str, gdp_weights=gdp_weights, weight_opt=weight_opt) for r in regions}
            elif impact == "dy":
                df_dp = load_combined(target_dir, file_str, "dp", pred_str)
                dp_m  = df_dp.merge(df_ipcc, on="country", how="left")
                # Baseline: FAO production-weighted
                g   = utils.compute_dy_weighted(df, df_dp, years_ts_str, prod_weights=prod_weights, weight_opt=weight_opt) / unit
                reg = {r: utils.compute_dy_weighted(df_m[df_m[ar6_region] == r], dp_m[dp_m[ar6_region] == r], years_ts_str, prod_weights=prod_weights, weight_opt=weight_opt) for r in regions}
                # CO2: same FAO weights
                g_co2   = utils.compute_dy_weighted(df_co2, df_dp, years_ts_str, prod_weights=prod_weights, weight_opt=weight_opt) / unit
                reg_co2 = {r: utils.compute_dy_weighted(df_m_co2[df_m_co2[ar6_region] == r], dp_m[dp_m[ar6_region] == r], years_ts_str, prod_weights=prod_weights, weight_opt=weight_opt) for r in regions}
            else:  # dp, de: flat sum
                g     = np.nan_to_num(df[years_ts_str].sum().values, nan=0) / unit
                g_co2 = np.nan_to_num(df_co2[years_ts_str].sum().values, nan=0) / unit
                reg     = {r: np.nan_to_num(df_m[df_m[ar6_region] == r][years_ts_str].sum().values, nan=0) for r in regions}
                reg_co2 = {r: np.nan_to_num(df_m_co2[df_m_co2[ar6_region] == r][years_ts_str].sum().values, nan=0) for r in regions}

            ts_store[impact] = {"g": g, "g_co2": g_co2, "reg": reg, "reg_co2": reg_co2}

            if row_idx == 0:
                sg     = smooth(g,window_size=5)
                sg_co2 = smooth(g_co2,window_size=5)
            else:
                sg = smooth(g,window_size=1)
                sg_co2 = smooth(g_co2, window_size=1)


            final_global = sg[plot_end_idx - 1]
            ax.plot(years_plot, sg[plot_start_idx:plot_end_idx],
                    color="black", linewidth=1.0, linestyle="solid", label="Global")
            ax.plot(years_plot, sg_co2[plot_start_idx:plot_end_idx],
                    color="black", linewidth=1.0, linestyle="dotted", alpha=1.0)
            # IPCC regions
            for region in regions:
                r      = reg[region] / unit
                r_co2  = reg_co2[region] / unit

                if row_idx == 0:
                    sr     = smooth(r,window_size=5)
                    sr_co2 = smooth(r_co2,window_size=5)
                else:
                    sr     = smooth(r,window_size=1)
                    sr_co2 = smooth(r_co2,window_size=1)

                final_reg = sr[plot_end_idx - 1]
                ax.plot(years_plot, sr[plot_start_idx:plot_end_idx],
                        color=ipcc_colors[region], linewidth=1.0, linestyle="solid",
                        label=ipcc_labels[region])
                ax.plot(years_plot, sr_co2[plot_start_idx:plot_end_idx],
                        color=ipcc_colors[region], linewidth=1.0, linestyle="dotted", alpha=1.0)

        # Common formatting
        ax.axhline(0, color="black", linestyle="dotted", lw=0.5)
        ax.grid(True)
        ax.set_xlim(start_year_plot, end_year_plot)
        ax.set_ylim(ylim_bottom[impact], ylim_top[impact])
        ax.set_xticks(year_ticks)
        ax.tick_params(labelsize=tick_fontsize)

        panel = label_alphabets[col_idx]
        ax.text(-0.03, 1.12, panel, transform=ax.transAxes,
                fontweight="bold", fontsize=title_fontsize + 1)

        ax.set_title(ylabels[impact], fontsize=title_fontsize)
        ax.set_xlabel("Year", fontsize=label_fontsize)

"""
LEGEND
"""
axes[1, 1].legend(bbox_to_anchor=(-0.14, -0.48), loc="center",
                  fontsize=legend_fontsize, frameon=True, ncol=4)

plt.tight_layout()
plt.subplots_adjust(hspace=0.6,
                    wspace=0.15,
                    left=0.08,
                    bottom=0.19,
                    right=0.98,
                    top=0.92)

plt.show()

"""
COMPARISON TABLE: baseline vs CO2 fertilization
"""
print()
print("=" * 80)
print(f"COMPARISON TABLE: {pred_str} vs {pred_str_co2} ({start_year_plot}-{end_year_plot})")
print("=" * 80)

metrics, vals_base, vals_co2 = [], [], []
impact_order = ["dy", "degdp", "dp", "de"]
for impact in impact_order:
    ts   = ts_store[impact]
    unit = unit_div[impact]
    agg  = agg_method[impact]
    lbl  = unit_labels[impact]
    fn   = np.mean if agg == "mean" else np.sum

    g_sl     = ts["g"][plot_start_idx:plot_end_idx]
    g_co2_sl = ts["g_co2"][plot_start_idx:plot_end_idx]
    metrics.append(f"{impact.upper()} {agg.upper()} Global [{lbl}]")
    vals_base.append(round(float(fn(g_sl)), 6))
    vals_co2.append(round(float(fn(g_co2_sl)), 6))

    for region in regions:
        r_sl     = ts["reg"][region][plot_start_idx:plot_end_idx] / unit
        r_co2_sl = ts["reg_co2"][region][plot_start_idx:plot_end_idx] / unit
        metrics.append(f"{impact.upper()} {agg.upper()} {region.upper()} [{lbl}]")
        vals_base.append(round(float(fn(r_sl)), 6))
        vals_co2.append(round(float(fn(r_co2_sl)), 6))

df_comparison = pd.DataFrame({
    "Metric":       metrics,
    pred_str:       vals_base,
    pred_str_co2:   vals_co2,
})
df_comparison["Diff"] = df_comparison[pred_str_co2] - df_comparison[pred_str]
df_comparison["Diff (%)"] = (
    (df_comparison[pred_str_co2] - df_comparison[pred_str]) / df_comparison[pred_str].abs() * 100
)
print(df_comparison.to_string(index=False))
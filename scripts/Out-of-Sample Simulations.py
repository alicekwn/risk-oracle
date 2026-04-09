"""
This script is used to run out-of-sample simulations for the GJR-GARCH model with Student t-residuals.
"""

# ============================================================
# Author: Domingos Romualdo (refactored by ChatGPT)
# Date: 30 Mar 2026
# ============================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model
from scipy.stats import t
from riskenv.constants import DATA_DIR, FIGURE_DIR

# ============================================================
# 1. Load data
# ============================================================


def load_returns():
    # Read S&P data
    df_spx = pd.read_csv(DATA_DIR / "SPX.csv", parse_dates=["Date"])
    df_spx = df_spx.set_index("Date").sort_index()
    spx_returns = df_spx["SPX"].dropna().mul(100)

    # Read crypto data
    df_crypto = pd.read_csv(DATA_DIR / "Crypto.csv", parse_dates=["Date"])
    df_crypto = df_crypto.set_index("Date").sort_index()
    link_returns = df_crypto["Link"].dropna().mul(100)
    usdc_returns = df_crypto["USDC"].dropna().mul(100)

    return {
        "SPX": spx_returns,
        "Link": link_returns,
        "USDC": usdc_returns,
    }


# ============================================================
# 2. Choose series and simulation dates
# ============================================================


def choose_series(series_choice, series_dict):
    if series_choice == 0:
        return series_dict["SPX"], "SPX"
    elif series_choice == 1:
        return series_dict["Link"], "Link"
    else:
        return series_dict["USDC"], "USDC"


def get_simulation_dates(returns, series_name):
    # Require 10 years for SPX, 100 days otherwise
    min_sample = 2520 if series_name == "SPX" else 100

    # First simulation date: when minimum sample is first reached
    simulation_positions = [min_sample - 1]

    # Then add each end-of-month observation after that
    month_series = returns.index.to_series().dt.month
    month_change = month_series.ne(month_series.shift(-1))
    eom_positions = np.flatnonzero(month_change.to_numpy())

    # Keep only month-ends occurring after minimum sample and before final obs
    eom_positions = eom_positions[
        (eom_positions >= min_sample) & (eom_positions < len(returns) - 1)
    ]
    simulation_positions.extend(eom_positions.tolist())

    return min_sample, simulation_positions


# ============================================================
# 3. Run rolling / expanding estimation and OOS forecasting
# ============================================================


def run_oos_gjr_garch(returns, simulation_positions, p_val=0.05):
    # Prepare output frame indexed by full sample dates;
    # we will fill only the OOS forecast dates
    results_df = pd.DataFrame(
        index=returns.index,
        data={
            "return": returns,
            "volatility": np.nan,
            "VaR": np.nan,
            "CVaR": np.nan,
        },
    )

    for i, sample_end_pos in enumerate(simulation_positions):
        # Estimation sample: start through current simulation date
        est_sample = returns.iloc[: sample_end_pos + 1]

        # Forecast window: from next observation after sample_end_pos
        # through next simulation date, or through end of sample
        if i < len(simulation_positions) - 1:
            forecast_end_pos = simulation_positions[i + 1]
        else:
            forecast_end_pos = len(returns) - 1

        forecast_slice = returns.iloc[sample_end_pos + 1 : forecast_end_pos + 1]
        if forecast_slice.empty:
            continue

        # Estimate GJR-GARCH(1,1) with Student t innovations
        model = arch_model(est_sample, p=1, o=1, q=1, dist="t")
        fitted = model.fit(disp="off")

        # Safer parameter extraction by name
        params = fitted.params
        omega = (
            params["omega"] / 10000.0
        )  # returns were scaled by 100, so variance scales by 100^2
        alpha_garch = params["alpha[1]"]
        gamma = params["gamma[1]"]
        beta = params["beta[1]"]
        nu = params["nu"]

        # Standardized t quantile for VaR
        s_t_quantile = t.ppf(p_val, nu) * np.sqrt((nu - 2) / nu)

        # CVaR constant factor
        # z_alpha is quantile of ordinary Student t
        z_alpha = t.ppf(p_val, nu)

        term1 = -np.sqrt(nu) * (nu + z_alpha**2) / (nu - 1)
        term2 = t.pdf(z_alpha, nu) / p_val
        cvar_factor = term1 * term2

        # Last in-sample conditional volatility from fitted model
        current_vol = fitted.conditional_volatility.iloc[-1] / 100.0

        # Recursive OOS forecasting, storing directly into DataFrame
        for forecast_date, ret_val in forecast_slice.items():
            coeff = alpha_garch if ret_val >= 0 else (alpha_garch + gamma)
            current_var = omega + coeff * (ret_val / 100.0) ** 2 + beta * current_vol**2
            current_vol = np.sqrt(current_var)

            results_df.loc[forecast_date, "volatility"] = current_vol
            results_df.loc[forecast_date, "VaR"] = 100.0 * current_vol * s_t_quantile
            results_df.loc[forecast_date, "CVaR"] = 100.0 * current_vol * cvar_factor

    return results_df


# ============================================================
# 4. Plotting
# ============================================================


def plot_annualized_volatility(results_df, series_name):
    plot_df = results_df.dropna(subset=["volatility"]).copy()
    plot_df["ann_volatility"] = 100.0 * np.sqrt(252) * plot_df["volatility"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(plot_df.index, plot_df["ann_volatility"], color="royalblue", lw=1.5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.xticks(rotation=45)

    ax.set_title(f"OOS Forecasts for Annualized Conditional Volatility - {series_name}")
    ax.set_ylabel("Annualized Volatility")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"OOS_annualized_volatility_{series_name}.png")
    plt.show()


def plot_var_cvar(results_df, series_name, p_val):
    plot_df = results_df.dropna(subset=["VaR", "CVaR"]).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        plot_df.index, plot_df["return"], color="gray", alpha=0.3, label="Daily Returns"
    )
    ax.plot(
        plot_df.index,
        plot_df["VaR"],
        color="blue",
        lw=1.5,
        label=f"{100 * (1 - p_val):.0f}% VaR",
    )
    ax.plot(
        plot_df.index,
        plot_df["CVaR"],
        color="red",
        lw=1.5,
        linestyle="--",
        label=f"{100 * (1 - p_val):.0f}% CVaR (Expected Shortfall)",
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.xticks(rotation=45)

    ax.set_title(
        f"OOS Tail Risk: VaR vs. CVaR (Student's t-dist) - {series_name}", fontsize=14
    )
    ax.set_ylabel("Returns")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"OOS_var_cvar_{series_name}.png")
    plt.show()


# ============================================================
# 5. Main
# ============================================================

series_choice = 0
p_val = 0.05

series_dict = load_returns()
returns, series_name = choose_series(series_choice, series_dict)

min_sample, simulation_positions = get_simulation_dates(returns, series_name)
results_df = run_oos_gjr_garch(returns, simulation_positions, p_val=p_val)

# Keep only true OOS regionto maintain the same effective span as before
results_oos = results_df.iloc[min_sample:].copy()

plot_annualized_volatility(results_oos, series_name)

plot_var_cvar(results_oos, series_name, p_val)

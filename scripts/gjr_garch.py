"""
This script is used to estimate the GJR-GARCH model with t-residuals and compute and display annualised conditional volatility.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model
from scipy.stats import t

# *** 1. Import the log-returns data ***
# Note: the S&P returns have data for weekdays only, whereas
# the crypto returns include weekends. For now, I've kept them
# in separate files

# Read S&P log-returns data
df_spx = pd.read_csv("data/SPX.csv")
df_spx["Date"] = pd.to_datetime(df_spx["Date"])
df_spx.set_index("Date", inplace=True)
SPXReturns = df_spx["SPX"].dropna() * 100

# Read Crypto log-returns data
df_crypto = pd.read_csv("data/crypto.csv")
df_crypto["Date"] = pd.to_datetime(df_crypto["Date"])
df_crypto.set_index("Date", inplace=True)

series_dict = {
    "SPX": SPXReturns,
    **{col: df_crypto[col].dropna() * 100 for col in df_crypto.columns},
}

seriesName = "Link"  # or "SPX", "Link", etc.
returns = series_dict[seriesName]

# *** 2. Estimate the GJR-GARCH(1,1,1) model with t-residuals ***
# p=1 (Lagged variance), q=1 (Lagged squared error), o=1 (Asymmetric/GJR term)
model = arch_model(returns, p=1, o=1, q=1, dist="t")
results = model.fit(disp="off")
print(results.summary())

# *** 3. Compute and display annualized conditional volatility ***
# arch_model returns variance; we take the square root for volatility.
# Multiply by sqrt(252) to annualize (assuming 252 trading days).
cond_vol = results.conditional_volatility
ann_vol = cond_vol * np.sqrt(252)

# Plot the model-implied conditional volatility series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ann_vol, color="royalblue", lw=1.5)

# Formatting the x-axis as "mmm yy"
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
plt.xticks(rotation=45)

ax.set_title("Annualized Conditional Volatility (GJR-GARCH) - " + seriesName)
ax.set_ylabel("Annualized Volatility")
ax.set_xlabel("Date")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"figures/volatility_{seriesName}.png")
plt.show()

# *** 5. Compute and graph the model-implied VaR and CVaR (Expected Shortfall) ***
# Get the conditional volatility (sigma) from results
nu = results.params["nu"]
alpha = 0.05

# Calculate the standardized t-quantile
# We use (nu-2)/nu to scale because the GARCH model standardizes innovations
s_t_quantile = t.ppf(alpha, nu) * np.sqrt((nu - 2) / nu)

# Compute the daily Value at Risk
value_at_risk = cond_vol * s_t_quantile

# Compute the daily CVar
z_alpha = t.ppf(alpha, nu)

# Calculate the CVaR constant factor (for a standard Student's t)
# This is the expected value of X given X < z_alpha
term1 = -(nu**0.5) * (nu + z_alpha**2) / (nu - 1)
term2 = t.pdf(z_alpha, nu) / alpha
cvar_factor = term1 * term2  # Note: this will be a negative value
cvar_series = cond_vol * cvar_factor

# Plotting
plt.close()
fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(returns, color="gray", alpha=0.3, label="Daily Returns")
plt.plot(value_at_risk, color="blue", lw=1.5, label=f"{100*(1-alpha)}% VaR")
plt.plot(
    cvar_series,
    color="red",
    lw=1.5,
    linestyle="--",
    label=f"{100*(1-alpha)}% CVaR (Expected Shortfall)",
)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
plt.xticks(rotation=45)

plt.title(
    "GARCH(1,1) Tail Risk: VaR vs. CVaR (Student's t-dist) - " + seriesName, fontsize=14
)
plt.ylabel("Returns")
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.savefig(f"figures/var_cvar_{seriesName}.png")
plt.show()

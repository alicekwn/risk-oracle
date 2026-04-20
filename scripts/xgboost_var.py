"""
XGBoost quantile VaR model for BTC and ETH.

Trains a 5% quantile regression model using a rich set of market, sentiment, stress, and cross-asset features, then compares its tail-risk forecasts against the GJR-GARCH(1,1,1) baseline from Out-of-Sample Simulations.py.

Runs four ablation variants:
1. `garch_scaled` (full features, GJR-GARCH scaler),
2. `vol10d_scaled` (excluding garch, 10-day vol scaler),
3. `simple` (17 basic price/vol/F&G features),
4. `simple_top15` (simple ∪ top-15 features by gain)

"""

import warnings
from dataclasses import dataclass, field

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from arch import arch_model
from scipy.stats import t as student_t

from riskenv.constants import DATA_DIR, FIGURE_DIR
from riskenv.evaluation import (
    backtest_report,
    cvar,
    label_regimes,
)
from riskenv.features import CoinFeatureBuilder, add_cross_asset_features

warnings.filterwarnings("ignore")


# Training constants
ALPHA = 0.05  # quantile level
TRAIN_FRAC = 0.70  # fraction of data to use for training
VAL_FRAC = 0.15  # fraction of data to use for validation
SEED = 42  # random seed

# Columns to exclude from feature set
NON_FEATURE_COLS = {
    "coin",
    "is_btc",
    "target_return_t1",
}


# ---------- Model variant dataclass ----------
@dataclass
class ModelVariant:
    """
    A single XGBoost quantile VaR configuration for ablation comparisons.

    scaler_col: panel column used as target scaler. The model is trained on target_return_t1 / panel[scaler_col] and predictions are multiplied by the same column at forecast time.

    excluded_features: extra columns to drop from the feature set (on top of NON_FEATURE_COLS).
    """

    name: str
    scaler_col: str
    excluded_features: set[str] = field(default_factory=set)


# Features kept by the "simple" variant — only raw price/vol primitives and F&G.
_SIMPLE_KEEP = {
    "ret_lag1",
    "ret_lag2",
    "ret_lag3",
    "ret_lag5",
    "ret_lag7",
    "ret_lag10",
    "ret_mean_3d",
    "ret_mean_5d",
    "ret_mean_10d",
    "volatility_5d",
    "volatility_10d",
    "volatility_30d",
    "range_ma_5d",
    "range_ma_10d",
    "fear_greed_index",
    "fg_lag1",
    "fg_lag2",
}

# Top-15 features by gain from the garch_scaled feature importance chart.
_TOP15_FEATURES = {
    "range_ma_10d",
    "fg_lag1",
    "is_extreme_fear",
    "volatility_10d",
    "downside_vol_10d",
    "vol10_lag1",
    "ret_min_10d",
    "fear_greed_index",
    "ret_mean_5d",
    "fg_lag2",
    "ret_q10_10d",
    "vol5_change_lag1",
    "downside_vol_5d",
    "mcap_z_30d",
    "vol_lag1",
}

VARIANTS: list[ModelVariant] = [
    ModelVariant(
        name="garch_scaled",
        scaler_col="garch_vol_t",  # Use GARCH-vol as the scaler
        excluded_features=set(),  # No excluded features
    ),
    ModelVariant(
        name="vol10d_scaled",
        scaler_col="volatility_10d",  # Use 10-day volatility as the scaler
        excluded_features={
            "garch_vol_t"
        },  # Only exclude GARCH-vol from the feature list
    ),
    ModelVariant(
        name="simple",
        scaler_col="volatility_10d",  # Use 10-day volatility as the scaler
        excluded_features=set(),  # keeps only _SIMPLE_KEEP
    ),
    ModelVariant(
        name="simple_top15",
        scaler_col="volatility_10d",  # Use 10-day volatility as the scaler
        excluded_features=set(),  # keeps _SIMPLE_KEEP ∪ _TOP15_FEATURES
    ),
]


# ---------- Load data ----------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load price data and fear&greed index.
    """
    btc = pd.read_csv(DATA_DIR / "price_mcap_BTC.csv", parse_dates=["date"])
    eth = pd.read_csv(DATA_DIR / "price_mcap_ETH.csv", parse_dates=["date"])
    fg = pd.read_csv(DATA_DIR / "fear_greed_index.csv", parse_dates=["date"])

    for df in (btc, eth, fg):
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

    return btc, eth, fg


# --------- Feature engineering ----------


def build_panel(
    df_btc: pd.DataFrame,
    df_eth: pd.DataFrame,
    df_fg: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a feature panel from price data and fear/greed index.
    """
    # Per-coin features
    btc_feat = CoinFeatureBuilder(df_btc, df_fg, "BTC").build()
    eth_feat = CoinFeatureBuilder(df_eth, df_fg, "ETH").build()

    # Cross-asset enrichment — anchor columns are always named anchor_* so both coins share the same column schema with no NaN by design.
    btc_feat = add_cross_asset_features(btc_feat, eth_feat)  # ETH is BTC's anchor
    eth_feat = add_cross_asset_features(eth_feat, btc_feat)  # BTC is ETH's anchor

    # GJR-GARCH conditional vol series — computed per coin over the full sample.
    # Available as a feature or scaler column; the recursion uses only past returns.
    print("Computing GJR-GARCH conditional vol series …")
    btc_garch_vol = compute_garch_vol_series(df_btc, "BTC")
    eth_garch_vol = compute_garch_vol_series(df_eth, "ETH")
    btc_feat["garch_vol_t"] = btc_garch_vol
    eth_feat["garch_vol_t"] = eth_garch_vol

    # Stack into a single panel sorted by date.
    panel = pd.concat([btc_feat, eth_feat]).sort_index()
    panel.replace([np.inf, -np.inf], np.nan, inplace=True)

    panel = panel.dropna(subset=["target_return_t1"])

    nan_pct = panel.isna().mean().sort_values(ascending=False)
    top_nan = nan_pct[nan_pct > 0].head(10)
    if not top_nan.empty:
        print("NaN % per column (top 10 with any NaN):")
        for col, pct in top_nan.items():
            print(f"  {col:<35s} {pct:.1%}")

    return panel


# --------- Train / val / test split ----------


def chronological_split(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split on unique calendar dates so both coins share the same date boundaries."""
    dates = panel.index.unique().sort_values()
    n = len(dates)
    n_train = int(n * TRAIN_FRAC)  # number of training dates
    n_val = int(n * VAL_FRAC)  # number of validation dates

    train_dates = dates[:n_train]
    val_dates = dates[n_train : n_train + n_val]
    test_dates = dates[n_train + n_val :]

    train_data = panel[panel.index.isin(train_dates)]  # panel of training data
    val_data = panel[panel.index.isin(val_dates)]  # panel of validation data
    test_data = panel[panel.index.isin(test_dates)]  # panel of test data

    print(
        f"\nChronological split ({len(dates)} unique dates):\n"
        f"  Train : {train_dates[0].date()} → {train_dates[-1].date()} "
        f"({len(train_dates)} dates, {len(train_data)} rows)\n"
        f"  Val   : {val_dates[0].date()} → {val_dates[-1].date()} "
        f"({len(val_dates)} dates, {len(val_data)} rows)\n"
        f"  Test  : {test_dates[0].date()} → {test_dates[-1].date()} "
        f"({len(test_dates)} dates, {len(test_data)} rows)"
    )
    return train_data, val_data, test_data


def get_Xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Get X and y from a feature panel. X is a DataFrame and y is a Series.
    """
    return df[feature_cols].copy(), df["target_return_t1"].copy()


def impute_with_train_medians(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fill NaN using training-set column medians (computed on train only)."""
    medians = X_train.median()
    return (
        X_train.fillna(medians),
        X_val.fillna(medians),
        X_test.fillna(medians),
    )


# --------- XGBoost quantile model ----------

XGB_PARAMS = {
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "lambda": 1.0,
    "alpha": 0.1,
    "seed": SEED,
    "nthread": -1,
    "disable_default_eval_metric": 1,
}


def _pinball_objective(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> tuple[np.ndarray, np.ndarray]:
    """
    Custom pinball (quantile) loss at level ALPHA.

    Gradient derivation for L(y, f) at quantile alpha:
      L = alpha * (y - f)        if y >= f   (under-predict)
        = (alpha - 1) * (y - f)  if y <  f   (over-predict)
    dL/df:
      = -alpha                   if y >= f   → push f up
      = (1 - alpha)              if y <  f   → push f down
    The asymmetry makes the model converge to the alpha-quantile.
    Hessian = 1 everywhere (piecewise-linear loss has no curvature).
    """
    y = dtrain.get_label()
    residual = y - predt
    grad = np.where(residual >= 0, -ALPHA, 1.0 - ALPHA)
    hess = np.ones_like(predt)
    return grad, hess


def _pinball_metric(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
    """Evaluation metric consistent with the training objective."""
    y = dtrain.get_label()
    residual = y - predt
    loss = np.where(residual >= 0, ALPHA * residual, (ALPHA - 1) * residual)
    return "pinball", float(np.mean(loss))


def _fit_one(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    verbose: bool = False,
) -> xgb.Booster:
    """Fit a single XGBoost quantile model with early stopping on val pinball."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=2000,
        obj=_pinball_objective,
        custom_metric=_pinball_metric,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100 if verbose else False,
    )
    return model


def train_xgb_quantile(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> xgb.Booster:
    """Static fit (used for feature importance after rolling retrain)."""
    print("\nTraining static XGBoost model (for feature importance) …")
    model = _fit_one(X_train, y_train, X_val, y_val, verbose=True)
    print(
        f"Best iteration: {model.best_iteration}  "
        f"| val pinball: {model.best_score:.6f}"
    )
    return model


# --------- Rolling / expanding window retrain ----------

VAL_DAYS = 180  # dates held out at the tail of each expanding window for early stopping


def run_rolling_xgb(
    panel: pd.DataFrame,
    variant: ModelVariant,
    feature_cols_base: list[str],
    val_end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Weekly expanding-window retrain over the test period for a single variant.

    Scaling is variant-driven: train on r_{t+1} / panel[variant.scaler_col] and
    unscale predictions at inference time: VaR_pred = q_hat * panel[scaler_col].
    """
    all_dates = panel.index.unique().sort_values()
    test_dates = all_dates[all_dates > val_end_date]

    if len(test_dates) == 0:
        raise ValueError("No test dates after val_end_date.")

    feature_cols = [c for c in feature_cols_base if c not in variant.excluded_features]

    scaler = panel[variant.scaler_col].clip(lower=1e-8)
    y_scaled = panel["target_return_t1"] / scaler

    fit_mask = y_scaled.notna() & scaler.notna()

    # Weekly re-estimation: last trading day of each ISO week in the test period.
    test_series = pd.Series(test_dates, index=test_dates)
    week_change = test_series.dt.isocalendar().week.ne(
        test_series.dt.isocalendar().week.shift(-1)
    )
    eow_test = test_series[week_change].values  # end of week test dates
    est_dates = np.concatenate([[val_end_date], eow_test[:-1]])  # estimation dates

    records: list[pd.DataFrame] = []

    print(
        f"\n[{variant.name}] Weekly expanding-window XGBoost retrain: "
        f"{len(est_dates)} windows  (scaler={variant.scaler_col}, "
        f"features={len(feature_cols)}) …"
    )
    for i, est_end in enumerate(est_dates):
        next_est = est_dates[i + 1] if i < len(est_dates) - 1 else all_dates[-1]
        fcast_dates = all_dates[
            (all_dates > est_end) & (all_dates <= next_est)
        ]  # forecast dates

        if len(fcast_dates) == 0:
            continue

        avail_dates = all_dates[all_dates <= est_end]  # available dates
        if len(avail_dates) < VAL_DAYS + 60:
            continue

        val_cutoff = avail_dates[-VAL_DAYS]  # validation cutoff date
        train_mask = (panel.index < val_cutoff) & fit_mask  # train mask
        val_mask = (
            (panel.index >= val_cutoff) & (panel.index <= est_end) & fit_mask
        )  # validation mask
        fcast_mask = panel.index.isin(fcast_dates)  # forecast mask

        train_data = panel[train_mask]  # train data
        val_data = panel[val_mask]  # validation data
        fcast_data = panel[fcast_mask]  # forecast data

        X_tr = train_data[feature_cols].copy()
        y_tr = y_scaled[train_mask]  # training target
        X_vl = val_data[feature_cols].copy()
        y_vl = y_scaled[val_mask]  # validation target
        X_fc = fcast_data[feature_cols].copy()

        medians = X_tr.median()
        X_tr = X_tr.fillna(medians)
        X_vl = X_vl.fillna(medians)
        X_fc = X_fc.fillna(medians)

        model = _fit_one(X_tr, y_tr, X_vl, y_vl, verbose=False)  # fit the model

        scaled_preds = model.predict(xgb.DMatrix(X_fc))  # predict the model
        var_pred = (
            scaled_preds * fcast_data[variant.scaler_col].values
        )  # unscale the predictions

        chunk = pd.DataFrame(
            {
                "y_true": fcast_data["target_return_t1"].values,
                "var_pred": var_pred,
                "coin": fcast_data["coin"].values,
            },
            index=fcast_data.index,
        )
        records.append(chunk)

        if i % 4 == 0 or i == len(est_dates) - 1:
            print(
                f"  [{i+1:3d}/{len(est_dates)}] est_end={est_end.date()}  "
                f"forecast={fcast_dates[0].date()}→{fcast_dates[-1].date()}  "
                f"train_rows={len(train_data)}  best_iter={model.best_iteration}"
            )

    return pd.concat(records).sort_index()


# --------- Evaluation helpers ----------


def evaluate_xgb(
    oos_results: pd.DataFrame,
    panel: pd.DataFrame,
    label: str = "XGBoost (rolling)",
) -> pd.DataFrame:
    """
    Evaluate a pre-assembled OOS results DataFrame.

    Parameters
    ----------
    oos_results : pd.DataFrame
        Must have columns: y_true, var_pred, coin.  Indexed by date.
    panel : pd.DataFrame
        Full feature panel — used to look up volatility_10d and fear_greed_index for regime labelling.
    label : str
        Label for the printed header.
    """
    # Attach regime labels from the panel (align by index).
    panel_test = panel[panel.index.isin(oos_results.index)]  # panel of test data
    regimes = label_regimes(panel_test).reindex(oos_results.index)  # regime labels

    results = oos_results.copy()
    results["regime"] = regimes.values

    print(f"\n=== {label} — OOS test ===")
    _print_backtest_table(
        [backtest_report(results["y_true"], results["var_pred"], label="Overall")]
        + [
            backtest_report(
                results.loc[results["coin"] == c, "y_true"],
                results.loc[results["coin"] == c, "var_pred"],
                label=c,
            )
            for c in ["BTC", "ETH"]
        ]
    )

    print(f"\n--- Regime calibration ({label}) ---")
    regime_reports = []
    for regime in sorted(results["regime"].dropna().unique()):
        mask = results["regime"] == regime
        regime_reports.append(
            backtest_report(
                results.loc[mask, "y_true"],
                results.loc[mask, "var_pred"],
                label=regime,
            )
        )
    _print_backtest_table(regime_reports)

    return results


def _print_backtest_table(reports: list[dict]) -> None:
    header = f"  {'Label':<20s} {'n':>6} {'Pinball':>10} {'Exceed':>8} {'CVaR':>8} {'Kupiec-p':>10} {'Christ-p':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in reports:
        ku = (
            f"{r['kupiec_p']:.3f}"
            if r["kupiec_p"] is not None and not np.isnan(r["kupiec_p"])
            else "  N/A"
        )
        ch = (
            f"{r['christoffersen_p']:.3f}"
            if r["christoffersen_p"] is not None and not np.isnan(r["christoffersen_p"])
            else "  N/A"
        )
        cv = (
            f"{r['cvar']:.4f}"
            if r.get("cvar") is not None and not np.isnan(r["cvar"])
            else "  N/A"
        )
        print(
            f"  {r['label']:<20s} {r['n']:>6d} {r['pinball']:>10.6f} "
            f"{r['exceedance']:>8.4f} {cv:>8} {ku:>10} {ch:>10}"
        )


# --------- GJR-GARCH conditional vol series (feature + baseline) ----------


def compute_garch_vol_series(
    price_df: pd.DataFrame,
    coin: str,
) -> pd.Series:
    """
    Fit an expanding-window GJR-GARCH(1,1,1)-t on a coin's daily log-returns
    and return the one-step-ahead conditional volatility for every date.

    The conditional volatility on date t is computed recursively from
    parameters estimated up to the last month-end before t, then updated
    day-by-day using the GJR-GARCH recursion.  It is therefore available at
    end-of-day t with no look-ahead.

    The series is in log-return space (same units as price_log_return).

    Parameters
    ----------
    price_df : pd.DataFrame
        price_mcap DataFrame for one coin, indexed by date.
    coin : str
        Label used in progress output only.

    Returns
    -------
    pd.Series
        Conditional volatility, indexed by date, named "garch_vol_t".
        NaN for dates before the first GJR-GARCH forecast is available.
    """
    returns_scaled = price_df["price_log_return"].dropna() * 100.0

    min_sample = 100
    month_series = returns_scaled.index.to_series().dt.month
    month_change = month_series.ne(month_series.shift(-1))
    eom_positions = np.flatnonzero(month_change.to_numpy())
    eom_positions = eom_positions[
        (eom_positions >= min_sample) & (eom_positions < len(returns_scaled) - 1)
    ]
    simulation_positions = [min_sample - 1] + eom_positions.tolist()

    vol_series = pd.Series(np.nan, index=returns_scaled.index, name="garch_vol_t")

    print(f"  Computing GJR-GARCH vol series for {coin} …")
    for i, sample_end_pos in enumerate(simulation_positions):
        next_pos = (
            simulation_positions[i + 1]
            if i < len(simulation_positions) - 1
            else len(returns_scaled) - 1
        )
        forecast_slice = returns_scaled.iloc[sample_end_pos + 1 : next_pos + 1]
        if forecast_slice.empty:
            continue

        est_sample = returns_scaled.iloc[: sample_end_pos + 1]
        garch = arch_model(est_sample, p=1, o=1, q=1, dist="t")
        fitted = garch.fit(disp="off")

        params = fitted.params
        omega = params["omega"] / 10_000.0
        alpha_garch = params["alpha[1]"]
        gamma = params["gamma[1]"]
        beta = params["beta[1]"]

        current_vol = fitted.conditional_volatility.iloc[-1] / 100.0

        for forecast_date, ret_val in forecast_slice.items():
            coeff = alpha_garch if ret_val >= 0 else (alpha_garch + gamma)
            current_var = omega + coeff * (ret_val / 100.0) ** 2 + beta * current_vol**2
            current_vol = np.sqrt(current_var)
            vol_series.loc[forecast_date] = current_vol  # log-return space

    return vol_series


def garch_vol_to_var(
    garch_vol: pd.Series,
    price_df: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    Convert a pre-computed GJR-GARCH conditional vol series to a 5% VaR series.

    Fits a single full-sample GJR-GARCH to extract the Student-t degrees of
    freedom (nu) for the quantile scaling factor, then applies
      VaR_t = sigma_t * q_{0.05}(t_nu)
    to every date in test_dates.

    This reuses the garch_vol_t series already computed in build_panel,
    so no second expanding-window fitting loop is needed.
    """
    returns_scaled = price_df["price_log_return"].dropna() * 100.0

    # Fit once on the full sample just to get a stable nu estimate.
    fitted = arch_model(returns_scaled, p=1, o=1, q=1, dist="t").fit(disp="off")
    nu = fitted.params["nu"]
    s_t_quantile = student_t.ppf(ALPHA, nu) * np.sqrt((nu - 2.0) / nu)

    var_series = (garch_vol * s_t_quantile).reindex(test_dates)
    return var_series


# --------- Comparison plot + table ----------


def compare_with_garch(
    xgb_results: pd.DataFrame,
    garch_var: pd.Series,
    price_df: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    coin: str = "BTC",
    variant_name: str = "xgb",
) -> None:
    """
    Compare XGBoost and GJR-GARCH VaR predictions.
    """
    ret = price_df["price_log_return"].reindex(test_dates)
    xgb_var = xgb_results[xgb_results["coin"] == coin]["var_pred"].reindex(test_dates)

    compare = pd.DataFrame(
        {"y_true": ret, "xgb_var": xgb_var, "garch_var": garch_var}
    ).dropna()

    xgb_report = backtest_report(compare["y_true"], compare["xgb_var"], label="XGBoost")
    garch_report = backtest_report(
        compare["y_true"], compare["garch_var"], label="GJR-GARCH"
    )

    print(f"\n=== {coin} VaR Backtest: XGBoost vs GJR-GARCH (test set) ===")
    _print_backtest_table([xgb_report, garch_report])

    # Exceedance masks for CVaR scatter points.
    xgb_exc_mask = compare["y_true"] < compare["xgb_var"]
    garch_exc_mask = compare["y_true"] < compare["garch_var"]

    # --- Plot: two panels, VaR on top, CVaR exceedances on bottom ---
    _, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top panel — returns + VaR lines
    ax1.plot(
        compare.index,
        compare["y_true"],
        color="gray",
        alpha=0.4,
        lw=1,
        label=f"{coin} Return",
    )
    ax1.plot(
        compare.index, compare["xgb_var"], color="blue", lw=1.5, label="XGBoost 5% VaR"
    )
    ax1.plot(
        compare.index,
        compare["garch_var"],
        color="red",
        lw=1.5,
        ls="--",
        label="GJR-GARCH 5% VaR",
    )
    ax1.set_title(f"{coin} Next-Day 5% VaR & CVaR: XGBoost vs GJR-GARCH (Test Set)")
    ax1.set_ylabel("Log Return")
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)

    # Bottom panel — realized tail losses (exceedance days only) as scatter.
    # Each dot is a day where the return breached the VaR; its y-value is the actual realized return, i.e. the realized loss in the tail.
    xgb_exc_dates = compare.index[xgb_exc_mask]
    garch_exc_dates = compare.index[garch_exc_mask]

    ax2.axhline(0, color="black", lw=0.8, ls=":")
    ax2.scatter(
        xgb_exc_dates,
        compare.loc[xgb_exc_mask, "y_true"],
        color="blue",
        s=25,
        alpha=0.7,
        label=f"XGBoost exceedances (n={xgb_exc_mask.sum()})",
    )
    ax2.scatter(
        garch_exc_dates,
        compare.loc[garch_exc_mask, "y_true"],
        color="red",
        s=25,
        alpha=0.7,
        marker="x",
        label=f"GJR-GARCH exceedances (n={garch_exc_mask.sum()})",
    )

    # Horizontal CVaR reference lines (full-period scalar).
    xgb_cvar_val = cvar(compare["y_true"], compare["xgb_var"])
    garch_cvar_val = cvar(compare["y_true"], compare["garch_var"])
    ax2.axhline(
        xgb_cvar_val,
        color="blue",
        lw=1.5,
        ls="--",
        label=f"XGBoost CVaR = {xgb_cvar_val:.4f}",
    )
    ax2.axhline(
        garch_cvar_val,
        color="red",
        lw=1.5,
        ls=":",
        label=f"GJR-GARCH CVaR = {garch_cvar_val:.4f}",
    )

    ax2.set_ylabel("Realized tail loss (log return)")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    path = FIGURE_DIR / f"xgb_vs_garch_var_{variant_name}_{coin}.png"
    plt.savefig(path, dpi=150)
    print(f"Figure saved: {path}")
    plt.show()


# --------- Feature importance plot ----------


def plot_feature_importance(
    model: xgb.Booster,
    feature_cols: list[str],
    top_n: int = 30,
    variant_name: str = "xgb",
) -> None:
    """
    Plot feature importance from the xgboost model.
    """
    scores = model.get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "gain": [scores.get(f, 0.0) for f in feature_cols],
            }
        )
        .sort_values("gain", ascending=False)
        .head(top_n)
    )

    _, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=imp_df, x="gain", y="feature", ax=ax, color="steelblue")
    ax.set_title(f"XGBoost Feature Importance — {variant_name} — Top {top_n} (Gain)")
    ax.set_xlabel("Gain")
    ax.set_ylabel("")
    plt.tight_layout()
    path = FIGURE_DIR / f"xgb_feature_importance_{variant_name}.png"
    plt.savefig(path, dpi=150)
    print(f"Figure saved: {path}")
    plt.show()


def main() -> None:
    btc_df, eth_df, fg_df = load_data()

    # Feature engineering — shared across all variants.
    panel = build_panel(btc_df, eth_df, fg_df)
    feature_cols_base = [c for c in panel.columns if c not in NON_FEATURE_COLS]
    print(f"\nBase feature count: {len(feature_cols_base)}")

    # Initial train / val / test split (boundaries shared with rolling retrain).
    train, val, _ = chronological_split(panel)
    val_end_date = val.index.max()

    # Fill excluded_features for variants whose keep-set depends on panel columns.
    for v in VARIANTS:
        if v.name == "simple":
            v.excluded_features = {
                c for c in feature_cols_base if c not in _SIMPLE_KEEP
            }
        elif v.name == "simple_top15":
            v.excluded_features = {
                c
                for c in feature_cols_base
                if c not in (_SIMPLE_KEEP | _TOP15_FEATURES)
            }

    # GJR-GARCH baseline VaR is the same across variants — compute once.
    btc_full_dates = panel[panel["coin"] == "BTC"].index
    eth_full_dates = panel[panel["coin"] == "ETH"].index
    btc_garch_vol = panel[panel["coin"] == "BTC"]["garch_vol_t"]
    eth_garch_vol = panel[panel["coin"] == "ETH"]["garch_vol_t"]
    print("\nConverting GJR-GARCH vol → VaR (fitting nu on full sample) …")
    btc_garch_var_full = garch_vol_to_var(btc_garch_vol, btc_df, btc_full_dates)
    eth_garch_var_full = garch_vol_to_var(eth_garch_vol, eth_df, eth_full_dates)

    for variant in VARIANTS:
        print(f"\n{'=' * 70}")
        print(f"Running variant: {variant.name}")
        print(f"{'=' * 70}")

        feature_cols = [
            c for c in feature_cols_base if c not in variant.excluded_features
        ]

        oos_results = run_rolling_xgb(panel, variant, feature_cols_base, val_end_date)
        xgb_results = evaluate_xgb(oos_results, panel, label=variant.name)

        # Static model on train+val for feature importance only (variant's scaler).
        scaler_train = train[variant.scaler_col].clip(lower=1e-8)
        scaler_val = val[variant.scaler_col].clip(lower=1e-8)
        y_train = train["target_return_t1"] / scaler_train
        y_val = val["target_return_t1"] / scaler_val
        fit_mask_tr = y_train.notna()
        fit_mask_vl = y_val.notna()

        X_train = train.loc[fit_mask_tr, feature_cols].copy()
        y_train = y_train[fit_mask_tr]
        X_val = val.loc[fit_mask_vl, feature_cols].copy()
        y_val = y_val[fit_mask_vl]
        X_train, X_val, _ = impute_with_train_medians(X_train, X_val, X_val.copy())
        static_model = train_xgb_quantile(X_train, y_train, X_val, y_val)

        btc_test_dates = (
            xgb_results[xgb_results["coin"] == "BTC"].index.unique().sort_values()
        )
        eth_test_dates = (
            xgb_results[xgb_results["coin"] == "ETH"].index.unique().sort_values()
        )

        btc_garch_var = btc_garch_var_full.reindex(btc_test_dates)
        eth_garch_var = eth_garch_var_full.reindex(eth_test_dates)

        compare_with_garch(
            xgb_results,
            btc_garch_var,
            btc_df,
            btc_test_dates,
            coin="BTC",
            variant_name=variant.name,
        )
        compare_with_garch(
            xgb_results,
            eth_garch_var,
            eth_df,
            eth_test_dates,
            coin="ETH",
            variant_name=variant.name,
        )

        plot_feature_importance(static_model, feature_cols, variant_name=variant.name)

    print("\nDone.")


if __name__ == "__main__":
    main()

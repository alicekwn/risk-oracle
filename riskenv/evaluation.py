"""
VaR evaluation metrics.

var_pred is the predicted lower-tail bound, i.e. a *negative* number for a 5% VaR expressed in log-return space
e.g. -0.04 means "we expect only a 5% chance of the return falling below -4%".

An exceedance occurs on day t when: y_true_t < var_pred_t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def pinball_loss(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> float:
    """
    Mean pinball (quantile) loss at level *alpha*.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    loss = np.where(residual >= 0, alpha * residual, (alpha - 1) * residual)
    return float(np.mean(loss))


def exceedance_rate(
    y_true: np.ndarray | pd.Series,
    var_pred: np.ndarray | pd.Series,
) -> float:
    """
    Fraction of observations where the actual return fell below the VaR bound.
    """
    y_true = np.asarray(y_true, dtype=float)
    var_pred = np.asarray(var_pred, dtype=float)
    return float(np.mean(y_true < var_pred))


def cvar(
    y_true: np.ndarray | pd.Series,
    var_pred: np.ndarray | pd.Series,
) -> float:
    """
    Conditional Value-at-Risk (Expected Shortfall): mean return on days where the actual return fell below the VaR bound.

    Returns NaN when there are no exceedances.
    """
    y_true = np.asarray(y_true, dtype=float)
    var_pred = np.asarray(var_pred, dtype=float)
    exceedances = y_true[y_true < var_pred]
    return float(np.mean(exceedances)) if len(exceedances) > 0 else np.nan


def kupiec_test(
    y_true: np.ndarray | pd.Series,
    var_pred: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> dict:
    """
    Likelihood-ratio(LR) test of H0: P(exceedance) == alpha.
    Under H0 the LR statistic is asymptotically chi-squared with 1 degree of freedom.
    A p-value > 0.05 means we fail to reject adequate coverage.
    """
    y_true = np.asarray(y_true, dtype=float)  # realised loss
    var_pred = np.asarray(var_pred, dtype=float)  # predicted VaR

    n = len(y_true)
    exc = int(np.sum(y_true < var_pred))  # number of exceedances
    p = exc / n if n > 0 else np.nan  # probability of exceedance

    if exc == 0 or exc == n:
        return {"exc": exc, "n": n, "p_hat": p, "lr_stat": np.nan, "p_value": np.nan}

    # likelihood ratio
    lr = -2.0 * (
        exc * np.log(alpha / p) + (n - exc) * np.log((1.0 - alpha) / (1.0 - p))
    )
    p_value = float(chi2.sf(lr, df=1))
    return {"exc": exc, "n": n, "p_hat": p, "lr_stat": float(lr), "p_value": p_value}


def christoffersen_test(
    y_true: np.ndarray | pd.Series,
    var_pred: np.ndarray | pd.Series,
) -> dict:
    """
    Likelihood-ratio test of H0: exceedances are independently distributed.
    A clustered VaR model (violations follow violations) fails this test.
    Under H0 the LR statistic is asymptotically chi-squared with 1 degree of freedom.
    A p-value > 0.05 means we fail to reject independence.
    """
    y_true = np.asarray(y_true, dtype=float)
    var_pred = np.asarray(var_pred, dtype=float)

    hits = (y_true < var_pred).astype(int)

    n00 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 0)))
    n01 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 1)))
    n10 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 0)))
    n11 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 1)))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else np.nan
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else np.nan
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if len(hits) > 1 else np.nan

    def _safe_log(x: float) -> float:
        return np.log(x) if x > 0 else 0.0

    if any(v is np.nan for v in [pi01, pi11, pi]):
        return {
            "n00": n00,
            "n01": n01,
            "n10": n10,
            "n11": n11,
            "pi01": pi01,
            "pi11": pi11,
            "lr_stat": np.nan,
            "p_value": np.nan,
        }

    lr = -2.0 * (
        n00 * _safe_log(1.0 - pi)
        + n01 * _safe_log(pi)
        + n10 * _safe_log(1.0 - pi)
        + n11 * _safe_log(pi)
        - n00 * _safe_log(1.0 - pi01)
        - n01 * _safe_log(pi01)
        - n10 * _safe_log(1.0 - pi11)
        - n11 * _safe_log(pi11)
    )
    p_value = float(chi2.sf(lr, df=1))
    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "pi01": pi01,
        "pi11": pi11,
        "lr_stat": float(lr),
        "p_value": p_value,
    }


# Regime labelling
def label_regimes(panel_test: pd.DataFrame) -> pd.Series:
    """
    Labels:
      'extreme_fear' : fear_greed_index < 20
      'high_vol'     : volatility_10d > 75th pctile over test set
      'low_vol'      : volatility_10d < 25th pctile over test set
      'normal'       : everything else
    """
    regimes = pd.Series("normal", index=panel_test.index, dtype=str)

    p25 = panel_test["volatility_10d"].quantile(0.25)
    p75 = panel_test["volatility_10d"].quantile(0.75)

    regimes[panel_test["volatility_10d"] < p25] = "low_vol"
    regimes[panel_test["volatility_10d"] > p75] = "high_vol"
    regimes[panel_test["fear_greed_index"] < 20] = "extreme_fear"

    return regimes


def backtest_report(
    y_true: np.ndarray | pd.Series,
    var_pred: np.ndarray | pd.Series,
    alpha: float = 0.05,
    label: str = "",
) -> dict:
    """
    Aggregate all backtest metrics into a single dict.
    """
    return {
        "label": label,
        "n": len(np.asarray(y_true)),
        "pinball": pinball_loss(y_true, var_pred, alpha),
        "exceedance": exceedance_rate(y_true, var_pred),
        "cvar": cvar(y_true, var_pred),
        "kupiec_p": kupiec_test(y_true, var_pred, alpha)["p_value"],
        "christoffersen_p": christoffersen_test(y_true, var_pred)["p_value"],
    }

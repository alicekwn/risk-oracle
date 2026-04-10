"""
Feature engineering for the XGBoost quantile VaR model.

Every feature is a snapshot of what is known at the end of day t, predicting the 5th percentile of return on day t+1.
  - Same-day OHLCV (open, high, low, close, volume) → available at EOD.
  - .shift(1) on returns / vol → yesterday's value ("lag" features).
  - .rolling(W) without prior shift → window ending on today (OK for price levels).
  - .rolling(W) after .shift(1) → window of the W days before today (used for stats over past returns so today's return is never included).
"""

import numpy as np
import pandas as pd


def _downside_std(x: np.ndarray) -> float:
    """Std of negative values only; NaN when fewer than 2 negatives exist."""
    neg = x[x < 0]
    return np.std(neg, ddof=1) if len(neg) >= 2 else np.nan


class CoinFeatureBuilder:
    """
    Builds the full feature DataFrame for a single coin.

    Parameters
    ----------
    df : pd.DataFrame
        Price / OHLCV DataFrame indexed by date, as produced by scripts/fetch_cryptocompare.py.
        Must contain columns:
        open, high, low, close, volumefrom, volumefrom_log_return,
        market_cap, market_cap_log_return, price_log_return,
        volatility_5d, volatility_10d, volatility_30d,
        volatility_5d_log_return, volatility_10d_log_return,
        volatility_30d_log_return.
    fg_df : pd.DataFrame
        Fear & Greed DataFrame indexed by date.  Must contain columns:
        fear_greed_index, fear_greed_log_return.
    coin : str
        Coin identifier, e.g. "BTC" or "ETH".
    """

    def __init__(self, df: pd.DataFrame, fg_df: pd.DataFrame, coin: str) -> None:
        self.df = df.copy()
        self.fg_df = fg_df.copy()
        self.coin = coin

    def build(self) -> pd.DataFrame:
        """Return the complete feature DataFrame (one row per date)."""
        feat = pd.DataFrame(index=self.df.index)
        feat["coin"] = self.coin
        feat["is_btc"] = 1 if self.coin == "BTC" else 0

        self._add_return_lags(feat)
        self._add_rolling_stats(feat)
        self._add_volume_mcap(feat)
        self._add_volatility(feat)
        self._add_stress_proxies(feat)
        self._add_sentiment(feat)
        self._add_target(feat)

        return feat

    # ------- Helpers — one method per feature group --------

    def _add_return_lags(self, feat: pd.DataFrame) -> None:
        ret = self.df["price_log_return"]
        vol_v = self.df["volumefrom_log_return"]
        mcap_lr = self.df["market_cap_log_return"]

        for n in [1, 2, 3, 5, 7, 10]:
            feat[f"ret_lag{n}"] = ret.shift(n)

        for n in [1, 2, 3, 5]:
            feat[f"vol_lag{n}"] = vol_v.shift(n)

        for n in [1, 2, 3]:
            feat[f"mcap_ret_lag{n}"] = mcap_lr.shift(n)

    def _add_rolling_stats(self, feat: pd.DataFrame) -> None:
        # Shift by 1 so rolling windows contain only past returns, not today's.
        ret_lag1 = self.df["price_log_return"].shift(1)

        for w in [3, 5, 10]:
            feat[f"ret_mean_{w}d"] = ret_lag1.rolling(w).mean()
            feat[f"abs_ret_mean_{w}d"] = ret_lag1.abs().rolling(w).mean()

        for w in [5, 10, 30]:
            feat[f"downside_vol_{w}d"] = ret_lag1.rolling(
                w, min_periods=max(w // 2, 5)
            ).apply(_downside_std, raw=True)

        feat["ret_q10_10d"] = ret_lag1.rolling(10, min_periods=5).quantile(0.10)
        feat["ret_q05_30d"] = ret_lag1.rolling(30, min_periods=15).quantile(0.05)
        feat["ret_min_10d"] = ret_lag1.rolling(10, min_periods=5).min()
        feat["ret_min_30d"] = ret_lag1.rolling(30, min_periods=15).min()

    def _add_volume_mcap(self, feat: pd.DataFrame) -> None:
        volume = self.df["volumefrom"]
        vol_lr = self.df["volumefrom_log_return"]
        mcap = self.df["market_cap"]
        mcap_lr = self.df["market_cap_log_return"]

        # Abnormal volume: today's divided by rolling mean of *past* W days.
        vol_shift1 = volume.shift(1)
        feat["abnormal_volume_5d"] = volume / vol_shift1.rolling(5).mean()
        feat["abnormal_volume_10d"] = volume / vol_shift1.rolling(10).mean()
        feat["volume_spike_dummy"] = (feat["abnormal_volume_5d"] > 2.0).astype(int)

        # Volume variability on log-returns (already stationary).
        vol_lr_lag1 = vol_lr.shift(1)
        feat["volume_std_5d"] = vol_lr_lag1.rolling(5).std()
        feat["volume_std_10d"] = vol_lr_lag1.rolling(10).std()

        # Market cap: rolling z-score eliminates the secular growth trend.
        mcap_shift1 = mcap.shift(1)
        roll_mean = mcap_shift1.rolling(30).mean()
        roll_std = mcap_shift1.rolling(30).std()
        feat["mcap_z_30d"] = (mcap - roll_mean) / roll_std
        feat["mcap_ret_std_10d"] = mcap_lr.shift(1).rolling(10).std()

    def _add_volatility(self, feat: pd.DataFrame) -> None:
        vol5 = self.df["volatility_5d"]
        vol10 = self.df["volatility_10d"]
        vol30 = self.df["volatility_30d"]

        # Raw levels (end-of-day snapshot).
        feat["volatility_5d"] = vol5
        feat["volatility_10d"] = vol10
        feat["volatility_30d"] = vol30

        # Yesterday's levels.
        feat["vol5_lag1"] = vol5.shift(1)
        feat["vol10_lag1"] = vol10.shift(1)
        feat["vol30_lag1"] = vol30.shift(1)

        # Term-structure spreads (today's values).
        feat["vol5_minus_vol30"] = vol5 - vol30
        feat["vol10_minus_vol30"] = vol10 - vol30

        # Acceleration / change (log-returns of vol, shifted by 1).
        feat["vol5_change_lag1"] = self.df["volatility_5d_log_return"].shift(1)
        feat["vol10_change_lag1"] = self.df["volatility_10d_log_return"].shift(1)
        feat["vol30_change_lag1"] = self.df["volatility_30d_log_return"].shift(1)
        feat["vol5_change_mean_3d"] = (
            self.df["volatility_5d_log_return"].shift(1).rolling(3).mean()
        )

    def _add_stress_proxies(self, feat: pd.DataFrame) -> None:
        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]
        vol10 = self.df["volatility_10d"]
        ret = self.df["price_log_return"]

        # Drawdown: today's close vs rolling max including today.
        feat["drawdown_7d"] = close / close.rolling(7, min_periods=3).max() - 1
        feat["drawdown_30d"] = close / close.rolling(30, min_periods=10).max() - 1

        # Intraday range — all today's values, available at EOD.
        intraday = (high - low) / close
        feat["intraday_range"] = intraday
        feat["range_ma_5d"] = intraday.shift(1).rolling(5).mean()
        feat["range_ma_10d"] = intraday.shift(1).rolling(10).mean()

        # Crash / shock dummies (based on *yesterday's* return).
        ret_lag1 = ret.shift(1)
        roll_q05_30 = ret_lag1.rolling(30, min_periods=15).quantile(0.05)
        roll_mean_30 = ret_lag1.rolling(30, min_periods=15).mean()
        roll_std_30 = ret_lag1.rolling(30, min_periods=15).std()

        feat["crash_1d"] = (ret_lag1 < roll_q05_30).astype(int)
        feat["large_neg_2sd"] = (ret_lag1 < roll_mean_30 - 2 * roll_std_30).astype(int)

        # Volatility regime dummies.
        roll_med_vol10 = vol10.shift(1).rolling(30, min_periods=15).median()
        feat["high_vol_regime"] = (vol10 > roll_med_vol10).astype(int)

        vol10_chg = self.df["volatility_10d_log_return"].shift(1)
        roll_q90_chg = vol10_chg.rolling(30, min_periods=15).quantile(0.90)
        feat["vol_spike_dummy"] = (vol10_chg > roll_q90_chg).astype(int)

    def _add_sentiment(self, feat: pd.DataFrame) -> None:
        # Align Fear & Greed to coin's date index; forward-fill weekend gaps.
        fg = (
            self.fg_df[["fear_greed_index", "fear_greed_log_return"]]
            .reindex(feat.index)
            .ffill()
        )
        fg_raw = fg["fear_greed_index"]
        fg_lr = fg["fear_greed_log_return"]

        feat["fear_greed_index"] = fg_raw  # today's EOD value
        feat["fg_lag1"] = fg_raw.shift(1)
        feat["fg_lag2"] = fg_raw.shift(2)
        feat["fg_change_lag1"] = fg_lr.shift(1)
        feat["fg_change_lag2"] = fg_lr.shift(2)
        feat["is_extreme_fear"] = (fg_raw < 20).astype(int)
        feat["is_fear"] = (fg_raw < 40).astype(int)

    def _add_target(self, feat: pd.DataFrame) -> None:
        # Raw next-day return — used for evaluation (pinball loss, exceedance).
        # target_scaled_t1 and vol_scale_t are added later in build_panel once garch_vol_t is available for use as the scaler.
        feat["target_return_t1"] = self.df["price_log_return"].shift(-1)


# Cross-asset enrichment
def add_cross_asset_features(
    own_feat: pd.DataFrame,  # feature frame for coin being enriched
    anchor_feat: pd.DataFrame,  # feature frame for the anchor coin
) -> pd.DataFrame:
    """
    Signals derived from the anchor coin.
    e.g. Bitcoin's features enriched with signals derived from Ethereum's features.
    """
    own_feat["anchor_ret_lag1"] = anchor_feat["ret_lag1"]
    own_feat["anchor_ret_lag2"] = anchor_feat["ret_lag2"]
    own_feat["anchor_vol10_lag1"] = anchor_feat["vol10_lag1"]
    own_feat["anchor_drawdown_7d"] = anchor_feat["drawdown_7d"]
    own_feat["anchor_abnormal_volume_5d"] = anchor_feat["abnormal_volume_5d"]

    # Spread vs anchor (both lag1 columns are already yesterday's values).
    own_feat["ret_minus_anchor_ret"] = own_feat["ret_lag1"] - anchor_feat["ret_lag1"]
    own_feat["own_vol10_over_anchor_vol10"] = (
        own_feat["vol10_lag1"] / anchor_feat["vol10_lag1"]
    )

    return own_feat

"""
This script fetches daily OHLCV price data and circulating supply data from CryptoCompare API,
and computes market-cap, price log-return, volumefrom log-return, market-cap log-return, and rolling volatility. It then saves the data to a CSV file.

CryptoCompare API can only fetch the following crypto: BTC, ETH, LTC, DOGE, BCH.

The date range is:
2013-01-01 → today for BTC,
2016-01-01 → today for ETH,
2013-01-01 → today for LTC,
2014-01-01 → today for DOGE,
2018-01-01 → today for BCH.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from riskenv.constants import DATA_DIR
from riskenv.settings import CRYPTOCOMPARE_API

if not CRYPTOCOMPARE_API:
    raise RuntimeError("Please set your CRYPTOCOMPARE_API environment variable")

HEADERS = {"authorization": f"Apikey {CRYPTOCOMPARE_API}"}
BASE_PRICE_URL = "https://min-api.cryptocompare.com/data/v2/histoday"
BASE_SUPPLY_URL = "https://min-api.cryptocompare.com/data/blockchain/histo/day"
SYMBOL = "ETH"
VS_CURRENCY = "USD"
START_TS = int(datetime(2018, 2, 1).timestamp())
END_TS = int(datetime(2026, 3, 31).timestamp())


def get_histoday(start_ts: int, end_ts: int) -> pd.DataFrame:
    """
    Page daily OHLCV price data in 2 000-day chunks.
    Returns DataFrame with columns: time, open, high, low, close, volumefrom, volumeto.
    """
    frames = []
    to_ts = end_ts
    while to_ts > start_ts:
        params = {"fsym": SYMBOL, "tsym": VS_CURRENCY, "limit": 2000, "toTs": to_ts}
        r = requests.get(
            BASE_PRICE_URL, headers=HEADERS, params=params, timeout=30
        ).json()
        data = r["Data"]["Data"]
        df = pd.DataFrame(data)
        frames.append(df)
        to_ts = df["time"].min() - 86400  # step back one day
        time.sleep(0.25)
    df_all = pd.concat(frames, ignore_index=True).drop_duplicates("time")
    return df_all[(df_all.time >= start_ts) & (df_all.time <= end_ts)]


def get_supply(start_ts: int, end_ts: int) -> pd.DataFrame:
    """
    Page daily circulating-supply in 2 000-day chunks.
    Returns DataFrame with columns: time, supply.
    """
    frames = []
    to_ts = end_ts
    while to_ts > start_ts:
        url = f"{BASE_SUPPLY_URL}?fsym={SYMBOL}" f"&limit=2000&toTs={to_ts}"
        r = requests.get(url, headers=HEADERS, timeout=30).json()
        if r.get("Response") == "Error":
            raise RuntimeError(f"CryptoCompare error: {r.get('Message')}")
        data = r["Data"]["Data"]
        df = pd.DataFrame(data).rename(columns={"current_supply": "supply"})
        frames.append(df[["time", "supply"]])
        to_ts = df["time"].min() - 86400
        time.sleep(0.25)
    df_all = pd.concat(frames, ignore_index=True).drop_duplicates("time")
    return df_all[(df_all.time >= start_ts) & (df_all.time <= end_ts)]


if __name__ == "__main__":
    # Fetch price & supply
    price_df = get_histoday(START_TS, END_TS)
    supply_df = get_supply(START_TS, END_TS)

    # Merge & compute market-cap
    df = (
        price_df.merge(supply_df, on="time", how="left")
        .assign(
            date=lambda d: pd.to_datetime(d.time, unit="s"),
            market_cap=lambda d: d.close * d.supply,
        )
        .set_index("date", drop=False)
        .sort_index()
    )
    df["price_log_return"] = df["close"].pct_change().apply(lambda x: np.log(1 + x))
    df["volumefrom_log_return"] = (
        df["volumefrom"].pct_change().apply(lambda x: np.log(1 + x))
    )
    df["market_cap_log_return"] = (
        df["market_cap"].pct_change().apply(lambda x: np.log(1 + x))
    )
    volatility_days = [5, 10, 30]  # days for rolling volatility
    for days in volatility_days:
        df[f"volatility_{days}d"] = (
            df["price_log_return"].rolling(window=days).std()
        )  # rolling volatility
        df[f"volatility_{days}d_log_return"] = (
            df[f"volatility_{days}d"].pct_change().apply(lambda x: np.log(1 + x))
        )

    out_file = DATA_DIR / f"price_mcap_{SYMBOL}.csv"
    df[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volumefrom",
            "volumeto",
            "supply",
            "market_cap",
            "price_log_return",
            "volumefrom_log_return",
            "market_cap_log_return",
            "volatility_5d",
            "volatility_10d",
            "volatility_30d",
            "volatility_5d_log_return",
            "volatility_10d_log_return",
            "volatility_30d_log_return",
        ]
    ].to_csv(out_file, index=False)
    print(f"Saved {len(df)} rows to {out_file}")

from fear_and_greed import FearAndGreedIndex
import pandas as pd
import numpy as np
from datetime import datetime
from riskenv.constants import DATA_DIR

fng = FearAndGreedIndex()

start_date = datetime(2018, 1, 1)

historical_data = fng.get_historical_data(start_date)

df = pd.DataFrame(historical_data)
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df.rename(columns={"value": "fear_greed_index"}, inplace=True)
df["date"] = pd.to_datetime(df["timestamp"], unit="s")
df = df.drop(columns=["time_until_update"])
df = df.sort_values("date", ascending=True)
df = df.reset_index(drop=True)
df["fear_greed_log_return"] = (
    df["fear_greed_index"].pct_change().apply(lambda x: np.log(1 + x))
)

df.to_csv(DATA_DIR / "fear_greed_index.csv", index=False)

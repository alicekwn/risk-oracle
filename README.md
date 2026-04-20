# Risk Oracle

## Setup

```
git clone https://github.com/alicekwn/risk-oracle.git
cd risk-oracle
```

### Give execute permission to your script and then run `setup_repo.sh`

```
chmod +x setup_repo.sh
./setup_repo.sh
. venv/bin/activate
```

or follow the step-by-step instructions below between the two horizontal rules:

---

#### Create a python virtual environment

- MacOS / Linux

```bash
python3 -m venv venv
```

- Windows

```bash
python -m venv venv
```

#### Activate the virtual environment

- MacOS / Linux

```bash
. venv/bin/activate
```

- Windows (in Command Prompt, NOT Powershell)

```bash
venv\Scripts\activate.bat
```

#### Install toml

```
pip install toml
```

#### Install the project in editable mode

```bash
pip install -e ".[dev]"
```

---

## Scripts 

### Fetch data

Fetch OHLC, market cap, volume data from cyptocompare (only for BTC, ETH, LTC, DOGE, BCH):
```
python scripts/fetch_cryptocompare.py
```
Fetch fear and greed index:
```
python scripts/fetch_fear_greed.py
```


### Traditional Economics Model
Estimate the GJR-GARCH model with t-residuals and compute and display annualised conditional volatility.
```
python scripts/gjr_garch.py
```

### XGBoost Model

Trains VaR model for BTC and ETH using an expanding-window weekly retrain. 

Runs four ablation variants:
1. `garch_scaled` (full features, GJR-GARCH scaler), 
2. `vol10d_scaled` (excluding garch, 10-day vol scaler), 
3. `simple` (17 basic price/vol/F&G features), 
4. `simple_top15` (simple ∪ top-15 features by gain)

```
python scripts/xgboost_var.py
```
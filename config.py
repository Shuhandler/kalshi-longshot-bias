"""
Settings shared by sample.py, replicate.py and the later backtest scripts.

Every constant is defined once here. download_data.py keeps its own download settings at its top.
"""
import datetime as dt
from pathlib import Path

# --- Paths ---
DATA = Path("data")
RESULTS = Path("results")

# --- Sample dates ---
PAPER_START = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)
PAPER_END = dt.datetime(2025, 5, 1, tzinfo=dt.timezone.utc)     # paper sample ends April 2025
OOS_END = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)      # out of sample: May 2025 to 31 Oct 2026
PERIODS = {"paper": (PAPER_START, PAPER_END), "oos": (PAPER_END, OOS_END)}

# --- Extra markets the series listing skips (sports, and daily markets inside "hourly" series) ---
INCLUDE_EXTRA = True        # data/paper_extra and data/oos_extra from download_data.py

# --- Paper sample filters (Bürgi, Deng and Whelan, section 3.1) ---
MIN_VOLUME_DOLLARS = 1_000  # Kalshi reports volume in contracts, each with $1 notional
MAX_FINAL_SPREAD = 0.20
MIN_HOURS_OPEN = 24
SNAPSHOT_DAYS = 10          # prices at the last trade and 1 to 10 days before it

# --- Fees and strategy ---
TAKER_FEE_RATE = 0.07       # fee = 0.07 * p * (1 - p) per contract, rounded up to the cent per order
FEE_ORDER_SIZE = 100        # the paper prices the round-up on an order of 100 contracts
INTEREST_RATE = 0.035       # annual interest on cash and positions
FAVOURITE_MIN, FAVOURITE_MAX = 0.85, 0.97
MIN_DAYS_TO_CLOSE, MAX_DAYS_TO_CLOSE = 1, 21
EVENT_CAP = 0.03

# --- Table 2 price ranges. Symmetric around 50c: "41c-50c" holds 41-49c and "50c-59c" holds 50-59c,
#     which is how the paper's counts add up. ---
PRICE_RANGES = ["1c-10c", "11c-20c", "21c-30c", "31c-40c", "41c-50c",
                "50c-59c", "60c-69c", "70c-79c", "80c-89c", "90c-99c"]

# --- Paper categories (Table 8). Kalshi's Elections counts as Politics; the rest is Other. ---
PAPER_CATEGORY = {"Financials": "Financials", "Climate and Weather": "Climate & Weather", "Crypto": "Crypto",
                  "Politics": "Politics", "Elections": "Politics", "Entertainment": "Entertainment",
                  "Economics": "Economics"}
NUMERICAL_STRIKES = ["greater", "greater_or_equal", "less", "less_or_equal", "between"]

# --- The paper's published numbers, for comparison (January 2026 version) ---
PAPER = {
    "counts": {"events": 12_403, "Yes contracts": 46_282, "Yes prices": 156_986, "Yes and No prices": 313_972},
    "table_1": [19_338, 12_861, 1_723, 1_083, 1_311, 1_288, 1_461, 159, 150, 154, 6_754],
    "table_2": [106_209, 20_395, 12_558, 10_049, 7_199, 8_351, 10_049, 12_558, 20_395, 106_209],
    "average_returns": {"all contracts, pre-fee": -0.20, "Takers, post-fee": -0.3146,
                        "Makers": -0.0964, "Makers at 50c and over": 0.026},
    "table_4_psi": {"All": 0.034, "Single": 0.045, "Non-exclusive": 0.036, "Exclusive numerical": 0.014,
                    "Exclusive other": 0.020, "All exclusive": 0.017},
    "table_4_constant": -1.736,
    "table_5_psi": [0.036, 0.017, 0.041, 0.036, 0.032, 0.029, 0.021, 0.037, 0.035, 0.040, 0.038],
    "table_6_psi": [0.045, 0.039, 0.032, 0.036, 0.032],
    "table_7_psi": [0.036, 0.033, 0.038, 0.034, 0.043],
    "table_8_psi": {"Financials": 0.032, "Climate & Weather": 0.031, "Crypto": 0.058, "Politics": 0.022,
                    "Entertainment": 0.020, "Economics": 0.034, "Other": 0.053},
    "table_9_psi": {2021: 0.041, 2022: 0.023, 2023: 0.036, 2024: 0.048, 2025: 0.021},
    "table_10_maker_share": [43.5, 46.7, 48.9, 47.8, 49.5, 50.4, 52.2, 51.1, 53.3, 56.5],
}

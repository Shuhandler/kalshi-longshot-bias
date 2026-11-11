"""
Download the Kalshi data used by kalshi_longshot.ipynb.

    .venv/bin/python download_data.py          # series, market listing and daily candles
    .venv/bin/python download_data.py trades   # trades for the tickers in data/trade_tickers.parquet

Everything is cached under data/ (git-ignored), one Parquet file per series, so a run that stops
can be restarted and skips what is already on disk. When it finishes it writes two combined files
for the notebook: data/markets.parquet and data/candles.parquet.

Kalshi splits its data in two. Markets that settled before the historical cutoff are only under
/historical/...; newer ones are on the live endpoints. Each function reads both. Historical markets
can only be filtered by series, so the listing loops over every series.

Values are stored as text exactly as the API returns them; the notebook converts them to numbers.
Run from Jordan's Mac: the Kalshi API cannot be reached from the cloud environment.
"""
import base64
import datetime as dt
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import polars as pl
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# --- Settings ---
BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
DATA = Path("data")
PAUSE = 0               # seconds between calls
WORKERS = 2             # requests sent at once for candles and trades; the Advanced tier allows 30 a second

# Which markets to download
SKIP_FREQUENCIES = {"hourly", "fifteen_min"}   # the paper drops hourly crypto and index markets
SKIP_CATEGORIES = {"Sports"}                   # sports left out for now (Jordan, 2026-11-11)
MIN_CONTRACTS_TRADED = 1_000                   # contracts >= dollars, so this keeps every $1,000 market
MIN_HOURS_OPEN = 24
DOWNLOAD_END = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)   # end of the out-of-sample window

MARKET_DROP = {"rules_primary", "rules_secondary", "custom_strike", "price_ranges", "mve_selected_legs"}


# --- API access ---
session = requests.Session()

# Market data is public, but logged-in requests get the account's rate limit (Advanced: 30 per second).
# The key ID and private key path come from .env, which is git-ignored. They are never printed.
env = {}
if Path(".env").exists():
    for line in Path(".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip("'").strip('"')
API_KEY_ID = env.get("KALSHI_API_KEY")
PRIVATE_KEY = None
if API_KEY_ID:
    key_file = Path(env["KALSHI_PRIVATE_KEY_PATH"]).expanduser()
    PRIVATE_KEY = serialization.load_pem_private_key(key_file.read_bytes(), password=None)


def auth_headers(path):
    """Signature headers for a GET request (empty without an API key, which still works at a lower limit)."""
    if PRIVATE_KEY is None:
        return {}
    timestamp = str(int(time.time() * 1000))
    message = timestamp + "GET" + urlparse(BASE_URL + path).path
    signature = PRIVATE_KEY.sign(
        message.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": API_KEY_ID,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode()}


def get(path, **params):
    """GET one API page, waiting and retrying when rate-limited or on a server error."""
    for attempt in range(8):
        time.sleep(PAUSE)
        try:
            r = session.get(BASE_URL + path, params=params, headers=auth_headers(path), timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (400, 404):    # e.g. a ticker containing "%", which the API cannot read
            print(f"Skipped {path}: {r.status_code}")
            return {}
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(f"Gave up on {path} {params}")


def get_all(path, key, **params):
    """Follow the cursor until the last page and return every item under `key`."""
    items, cursor = [], None
    while True:
        page = get(path, **params, cursor=cursor)
        items += page.get(key, [])
        cursor = page.get("cursor")
        if not cursor:
            return items


def to_text_table(rows):
    """Rows of flat dicts -> Polars table with every column stored as text."""
    rows = [{k: None if v is None else str(v) for k, v in row.items()} for row in rows]
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def ts(text):
    """ISO time string -> Unix seconds."""
    return int(dt.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


def combine(folder):
    """Stack every per-series file in a cache folder into one table."""
    files = sorted((DATA / folder).glob("*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


# --- Downloaders (each caches one series per file) ---
def pull_series():
    """All series with their category and frequency."""
    path = DATA / "series.parquet"
    if not path.exists():
        rows = [{k: v for k, v in s.items() if not isinstance(v, (list, dict))}
                | {"categories": ";".join(s.get("categories") or [])}
                for s in get("/series")["series"]]
        to_text_table(rows).write_parquet(path)
    return pl.read_parquet(path)


def pull_markets(series_ticker):
    """Every settled market in one series: historical archive plus live endpoint."""
    path = DATA / "markets" / f"{series_ticker}.parquet"
    if not path.exists():
        rows = []
        for source, endpoint, extra in [("historical", "/historical/markets", {}),
                                        ("live", "/markets", {"status": "settled"})]:
            for m in get_all(endpoint, "markets", series_ticker=series_ticker, limit=1000, **extra):
                rows.append({k: v for k, v in m.items() if k not in MARKET_DROP}
                            | {"series_ticker": series_ticker, "source": source})
        to_text_table(rows).write_parquet(path)
    return pl.read_parquet(path)


def pick(d, name):
    """Read a candle field. The historical endpoint calls it `close`, the live one `close_dollars` or `volume_fp`."""
    d = d or {}
    return d.get(name, d.get(name + "_dollars", d.get(name + "_fp")))


def candles_for_market(series_ticker, m):
    """Daily candles (price, bid, ask, volume) for one market, as flat rows."""
    if m["source"] == "historical":
        endpoint = f"/historical/markets/{m['ticker']}/candlesticks"
    else:
        endpoint = f"/series/{series_ticker}/markets/{m['ticker']}/candlesticks"
    page = get(endpoint, start_ts=ts(m["open_time"]), end_ts=ts(m["close_time"]) + 86_400, period_interval=1440)
    return [{
        "ticker": m["ticker"],
        "end_period_ts": c["end_period_ts"],
        "price_close": pick(c.get("price"), "close"),
        "price_mean": pick(c.get("price"), "mean"),
        "price_previous": pick(c.get("price"), "previous"),
        "yes_bid_close": pick(c.get("yes_bid"), "close"),
        "yes_ask_close": pick(c.get("yes_ask"), "close"),
        "volume": pick(c, "volume"),
        "open_interest": pick(c, "open_interest"),
    } for c in page.get("candlesticks", [])]


def pull_candles(series_ticker, markets):
    """Daily candles for the given markets of one series, WORKERS requests at a time."""
    path = DATA / "candles" / f"{series_ticker}.parquet"
    if not path.exists():
        with ThreadPoolExecutor(WORKERS) as pool:
            per_market = pool.map(lambda m: candles_for_market(series_ticker, m), markets.iter_rows(named=True))
            rows = [row for market_rows in per_market for row in market_rows]
        to_text_table(rows).write_parquet(path)
    return pl.read_parquet(path)


def trades_for_market(m):
    """Every trade, with the taker's side, for one market."""
    endpoint = "/historical/trades" if m["source"] == "historical" else "/markets/trades"
    return get_all(endpoint, "trades", ticker=m["ticker"], limit=1000)


def pull_trades(series_ticker, markets):
    """Every trade for the given markets of one series, WORKERS requests at a time."""
    path = DATA / "trades" / f"{series_ticker}.parquet"
    if not path.exists():
        with ThreadPoolExecutor(WORKERS) as pool:
            rows = [t for market_trades in pool.map(trades_for_market, markets.iter_rows(named=True))
                    for t in market_trades]
        to_text_table(rows).write_parquet(path)
    return pl.read_parquet(path)


# --- Steps ---
def download_markets():
    """Step 1: the market listing for every series we keep. Returns the combined, de-duplicated table."""
    series = pull_series()
    keep = series.filter(
        ~pl.col("frequency").is_in(list(SKIP_FREQUENCIES))
        & ~pl.col("category").is_in(list(SKIP_CATEGORIES))
        & ~pl.col("ticker").str.starts_with("KXMVE")          # multivariate combo series
    )["ticker"].to_list()
    print(f"{len(series):,} series, keeping {len(keep):,}")

    start = time.time()
    for i, ticker in enumerate(keep, 1):
        pull_markets(ticker)
        if i % 1000 == 0:
            print(f"{i:,} series done, {(time.time() - start) / 60:.0f} min")

    markets = pl.concat([pl.read_parquet(DATA / "markets" / f"{t}.parquet") for t in keep], how="diagonal_relaxed")
    markets = markets.unique("ticker", keep="first")      # a market can be in both the archive and the live listing
    markets.write_parquet(DATA / "markets.parquet")
    print(f"{len(markets):,} settled markets -> data/markets.parquet")
    return markets


def candle_markets(markets):
    """Loose pre-filter for the candle download. The paper's exact filters are applied in the notebook."""
    return (
        markets
        .with_columns(
            pl.col("volume_fp").cast(pl.Float64).alias("contracts"),
            pl.col("open_time").str.to_datetime(time_zone="UTC").alias("open"),
            pl.col("close_time").str.to_datetime(time_zone="UTC").alias("close"),
        )
        .filter(
            (pl.col("market_type") == "binary")
            & pl.col("result").is_in(["yes", "no"])
            & (pl.col("contracts") >= MIN_CONTRACTS_TRADED)
            & ((pl.col("close") - pl.col("open")) >= pl.duration(hours=MIN_HOURS_OPEN))
            & (pl.col("close") < DOWNLOAD_END)
        )
    )


def download_candles(markets):
    """Step 2: daily candles for the pre-filtered markets, one call per market."""
    wanted = candle_markets(markets)
    print(f"{len(wanted):,} markets pass the pre-filter")
    groups = sorted(wanted.partition_by("series_ticker", as_dict=True).items())
    start = time.time()
    for i, ((ticker,), group) in enumerate(groups, 1):
        pull_candles(ticker, group)
        if i % 200 == 0:
            print(f"{i:,} of {len(groups):,} series done, {(time.time() - start) / 60:.0f} min")

    candles = combine("candles")
    candles.write_parquet(DATA / "candles.parquet")
    print(f"{len(candles):,} daily candles for {candles['ticker'].n_unique():,} markets -> data/candles.parquet")


def download_trades():
    """Optional step: trades for the tickers the notebook saved to data/trade_tickers.parquet."""
    tickers = pl.read_parquet(DATA / "trade_tickers.parquet")["ticker"]
    markets = pl.read_parquet(DATA / "markets.parquet").filter(pl.col("ticker").is_in(tickers))
    groups = sorted(markets.partition_by("series_ticker", as_dict=True).items())
    print(f"Trades for {len(markets):,} markets in {len(groups):,} series")
    start = time.time()
    for i, ((ticker,), group) in enumerate(groups, 1):
        pull_trades(ticker, group)
        if i % 100 == 0:
            print(f"{i:,} of {len(groups):,} series done, {(time.time() - start) / 60:.0f} min")

    trades = combine("trades")
    trades.write_parquet(DATA / "trades.parquet")
    print(f"{len(trades):,} trades -> data/trades.parquet")


if __name__ == "__main__":
    for folder in ["markets", "candles", "trades"]:
        (DATA / folder).mkdir(parents=True, exist_ok=True)
    print("Historical cutoff:", get("/historical/cutoff"))
    if sys.argv[1:] == ["trades"]:
        download_trades()
    else:
        download_candles(download_markets())

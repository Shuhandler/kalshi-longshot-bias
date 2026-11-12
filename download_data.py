"""
Download the Kalshi data used by kalshi_longshot.ipynb.

    .venv/bin/python download_data.py          # series, market listing and daily candles
    .venv/bin/python download_data.py trades   # trades for the sample in data/trade_tickers.parquet
    .venv/bin/python download_data.py paper-extra   # sports and daily crypto/index markets up to April 2025
    .venv/bin/python download_data.py events   # event details (mutually exclusive or not) for the sample

Everything is cached under data/ (git-ignored), one Parquet file per series, so a run that stops
can be restarted and skips what is already on disk. The first command also writes two combined files
for the notebook: data/markets.parquet and data/candles.parquet. Trades stay one file per series.

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


TRADE_COLUMNS = ["ticker", "created_time", "yes_price_dollars", "count_fp", "taker_outcome_side", "is_block_trade"]


def trades_for_market(m):
    """Every trade for one market, keeping only the columns the notebook uses.

    Trades have their own archive cutoff, so a market that settled recently can still have its older
    trades in the archive. Archived markets need only the archive; live markets need both.
    """
    trades = get_all("/historical/trades", "trades", ticker=m["ticker"], limit=1000)
    if m["source"] == "live":
        trades += get_all("/markets/trades", "trades", ticker=m["ticker"], limit=1000)
    unique = {t["trade_id"]: t for t in trades}.values()      # a trade could be in both while it moves
    return [{k: t.get(k) for k in TRADE_COLUMNS} for t in unique]


def pull_trades(name, markets):
    """Every trade for the given markets, saved as data/trades/{name}.parquet, WORKERS requests at a time."""
    path = DATA / "trades" / f"{name}.parquet"
    if not path.exists():
        with ThreadPoolExecutor(WORKERS) as pool:
            rows = [t for market_trades in pool.map(trades_for_market, markets.iter_rows(named=True))
                    for t in market_trades]
        to_text_table(rows).write_parquet(path)


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
    """Trades for the sample the notebook saved to data/trade_tickers.parquet, paper period first.

    One file per series and period under data/trades/. There is no combined file: the full trade
    history is too large to hold in memory, so the notebook reads these files lazily.
    """
    sample = pl.read_parquet(DATA / "trade_tickers.parquet")              # columns: ticker, period
    markets = pl.read_parquet(DATA / "markets.parquet").join(sample, on="ticker")
    for period in ["paper", "oos"]:
        groups = sorted(markets.filter(pl.col("period") == period).partition_by("series_ticker", as_dict=True).items())
        n_markets = sum(len(g) for _, g in groups)
        print(f"{period}: trades for {n_markets:,} markets in {len(groups):,} series")
        start = time.time()
        for i, ((ticker,), group) in enumerate(groups, 1):
            pull_trades(f"{ticker}__{period}", group)
            if i % 100 == 0:
                print(f"  {i:,} of {len(groups):,} series done, {(time.time() - start) / 60:.0f} min")


# --- Extra markets for the paper period (Jordan, 2026-11-12) ---
# The series listing skips sports and every series tagged hourly or 15-minute. The paper instead keeps
# any market open at least 24 hours, and many markets in "hourly" crypto and index series (KXBTCD, INXU,
# ...) are daily markets. Before May 2025 the whole exchange traded little, so the quickest way to find
# these markets is to download every trade by day, take the traded tickers our listing does not have,
# and look those markets up by ticker. The 24-hour rule is applied in the notebook, so markets that
# really are hourly stay out of the sample.
EXTRA = DATA / "paper_extra"
EXTRA_START = dt.date(2021, 6, 30)
EXTRA_END = dt.date(2025, 5, 1)          # paper sample ends April 2025


def pull_trades_for_day(day):
    """Every trade on the exchange on one day (UTC), saved as data/paper_extra/trades_by_day/{day}.parquet."""
    path = EXTRA / "trades_by_day" / f"{day}.parquet"
    if not path.exists():
        start = int(dt.datetime.combine(day, dt.time(), dt.timezone.utc).timestamp())
        trades = get_all("/historical/trades", "trades", min_ts=start, max_ts=start + 86_400, limit=1000)
        to_text_table([{k: t.get(k) for k in TRADE_COLUMNS} for t in trades]).write_parquet(path)


def download_paper_extra():
    """Markets missing from the listing that closed before May 2025, with their candles and trades."""
    (EXTRA / "trades_by_day").mkdir(parents=True, exist_ok=True)

    # 1. Every trade on the exchange, one file per day
    days = [EXTRA_START + dt.timedelta(d) for d in range((EXTRA_END - EXTRA_START).days)]
    start = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, _ in enumerate(pool.map(pull_trades_for_day, days), 1):
            if i % 100 == 0:
                print(f"{i:,} of {len(days):,} days of trades, {(time.time() - start) / 60:.0f} min")

    # 2. Traded markets our listing does not have, apart from multivariate combos.
    #    A market ticker starts with its series ticker (KXBTCD-25JAN1717-T100000).
    listed = pl.read_parquet(DATA / "markets.parquet", columns=["ticker"])["ticker"]
    all_trades = combine("paper_extra/trades_by_day").unique()     # a trade on a midnight boundary is in two days
    extra_trades = all_trades.filter(~pl.col("ticker").is_in(listed.implode()) & ~pl.col("ticker").str.starts_with("KXMVE"))
    tickers = extra_trades["ticker"].unique().sort().to_list()
    print(f"{len(all_trades):,} trades on the exchange; {len(extra_trades):,} in {len(tickers):,} markets not in our listing")

    # 3. Their market records, 100 tickers per request; keep those settled before May 2025
    path = EXTRA / "markets.parquet"
    if not path.exists():
        rows = []
        for i in range(0, len(tickers), 100):
            for m in get("/historical/markets", tickers=",".join(tickers[i:i + 100]), limit=1000).get("markets", []):
                rows.append({k: v for k, v in m.items() if k not in MARKET_DROP}
                            | {"series_ticker": m["ticker"].split("-")[0], "source": "historical"})
        markets = to_text_table(rows)
        end = dt.datetime.combine(EXTRA_END, dt.time(), dt.timezone.utc)
        markets.filter(pl.col("close_time").str.to_datetime(time_zone="UTC") < end).write_parquet(path)
    markets = pl.read_parquet(path)
    print(f"{len(markets):,} extra markets closing before May 2025 -> {path}")

    # 4. Their trades, and daily candles for those that pass the loose pre-filter
    extra_trades.filter(pl.col("ticker").is_in(markets["ticker"].implode())).write_parquet(EXTRA / "trades.parquet")
    wanted = candle_markets(markets)
    path = EXTRA / "candles.parquet"
    if not path.exists():
        with ThreadPoolExecutor(WORKERS) as pool:
            per_market = pool.map(lambda m: candles_for_market(m["series_ticker"], m), wanted.iter_rows(named=True))
            to_text_table([row for rows in per_market for row in rows]).write_parquet(path)
    print(f"Candles for {len(wanted):,} extra markets -> {path}")


# --- Event details for the sample (paper Table 4 needs to know if an event is mutually exclusive) ---
EVENT_COLUMNS = ["event_ticker", "series_ticker", "title", "category", "mutually_exclusive"]


def event_details(event_ticker):
    """One event's details, without its markets."""
    event = get(f"/events/{event_ticker}").get("event", {})
    return {k: event.get(k) for k in EVENT_COLUMNS} | {"event_ticker": event_ticker}


def download_events():
    """Details for every event in data/trade_tickers.parquet, saved as data/events.parquet."""
    path = DATA / "events.parquet"
    if path.exists():
        return
    sample = pl.read_parquet(DATA / "trade_tickers.parquet")
    markets = pl.read_parquet(DATA / "markets.parquet", columns=["ticker", "event_ticker"])
    extra = EXTRA / "markets.parquet"
    if extra.exists():
        markets = pl.concat([markets, pl.read_parquet(extra, columns=["ticker", "event_ticker"])])
    events = markets.join(sample, on="ticker")["event_ticker"].unique().sort().to_list()
    print(f"Details for {len(events):,} events")
    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(event_details, events))
    to_text_table(rows).write_parquet(path)
    print(f"-> {path}")


if __name__ == "__main__":
    for folder in ["markets", "candles", "trades"]:
        (DATA / folder).mkdir(parents=True, exist_ok=True)
    print("Historical cutoff:", get("/historical/cutoff"))
    if sys.argv[1:] == ["trades"]:
        download_trades()
    elif sys.argv[1:] == ["paper-extra"]:
        download_paper_extra()
    elif sys.argv[1:] == ["events"]:
        download_events()
    else:
        download_candles(download_markets())

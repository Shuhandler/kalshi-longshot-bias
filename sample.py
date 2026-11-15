"""
Build the paper's sample and its price observations from the files in data/.

    .venv/bin/python sample.py

Steps (Bürgi, Deng and Whelan, section 3.1):
1. Load markets and daily candles (data/markets.parquet and data/candles.parquet from download_data.py),
   keeping sports and the daily markets inside "hourly" series if config.py says so.
2. Apply the paper's filters: settled Yes or No, at least $1,000 volume, open at least 24 hours,
   final bid-ask spread between 0 and 20c. The "final" spread comes from the last daily candle, which ends
   at midnight New York time, so it is measured a median of 10 to 16 hours before close (REVIEW.md M5).
3. Price snapshots from the trades: the last trade, and the price as of exactly 1 to 10 days
   (24-hour steps) before it, carrying the most recent earlier trade forward. Of the readings we tried
   (24-hour windows, calendar days, windows from the close) this one comes closest to the paper's Table 1:
   rows 2 to 9 match closely, but "closing day only" is short by about 6,000 contracts (REVIEW.md M6).
4. One Yes and one No observation per snapshot.

Close times: `close` is the final close, which for markets that can close early is only known afterwards.
`expected_expiration` is also a final value on today's market record, so it may leak too. `scheduled_close`
comes from the date written into the event ticker (e.g. KXHIGHNY-25JAN17), which is fixed when the market is
listed: the end of that day in New York. Use it for any strategy timing (REVIEW.md H4). Tickers with only a
month (e.g. CPI-23SEP, the data month, not the release date) or only a year give no scheduled close (null).

Writes data/observations_paper.parquet and data/observations_oos.parquet, which replicate.py reads,
and data/trade_tickers.parquet, which `download_data.py trades` reads.
"""
import polars as pl

from config import (DATA, PERIODS, INCLUDE_SPORTS, INCLUDE_HOURLY_SERIES, MIN_VOLUME_DOLLARS, MAX_FINAL_SPREAD, MIN_HOURS_OPEN,
                    SNAPSHOT_DAYS, NUMERICAL_STRIKES)
from stats import price_range


def load_markets_and_candles():
    """Markets (one row each, typed) and daily candles, filtered by INCLUDE_SPORTS and INCLUDE_HOURLY_SERIES."""
    events = pl.read_parquet(DATA / "events.parquet").select(
        "event_ticker", (pl.col("mutually_exclusive") == "True").alias("mutually_exclusive"),
        pl.col("category").alias("event_category"))
    markets = (
        pl.read_parquet(DATA / "markets.parquet")
        .select("ticker", "event_ticker", "series_ticker", "market_type", "strike_type", "result", "source", "category",
                "hourly_series", (pl.col("can_close_early") == "True").alias("can_close_early"),
                pl.col("expected_expiration_time").str.to_datetime(time_zone="UTC", strict=False).alias("expected_expiration"),
                pl.col("volume_fp").cast(pl.Float64).alias("volume"),
                pl.col("last_price_dollars").cast(pl.Float64).alias("reported_last_price"),
                pl.col("open_time").str.to_datetime(time_zone="UTC").alias("open"),
                pl.col("close_time").str.to_datetime(time_zone="UTC").alias("close"))
        .join(events, on="event_ticker", how="left")
        .with_columns(pl.coalesce("category", "event_category").alias("category"),   # series not in Kalshi's list
                      scheduled_close(pl.col("event_ticker")).alias("scheduled_close"))
        .drop("event_category")
    )
    if not INCLUDE_SPORTS:
        markets = markets.filter(pl.col("category") != "Sports")
    if not INCLUDE_HOURLY_SERIES:
        markets = markets.filter(~pl.col("hourly_series"))
    candles = pl.read_parquet(DATA / "candles.parquet").filter(pl.col("ticker").is_in(markets["ticker"].implode()))
    return markets, candles


MONTHS = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def scheduled_close(event_ticker):
    """End of the day written into the event ticker (YYMMMDD after the series part), in New York time, as UTC.
    Null when the ticker has no full date."""
    parts = event_ticker.str.extract_groups(r"-[A-Z]*(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})")
    day = pl.concat_str(parts.struct.field("1"),
                        parts.struct.field("2").replace_strict(MONTHS, default=None, return_dtype=pl.Int32).cast(pl.String).str.zfill(2),
                        parts.struct.field("3")).str.to_date("%y%m%d", strict=False)        # impossible dates -> null
    return (day.cast(pl.Datetime("us")) + pl.duration(days=1)).dt.replace_time_zone("America/New_York").dt.convert_time_zone("UTC")


def final_spread(markets, candles):
    """Bid-ask spread on the last daily candle ending no later than one hour after the market closes.
    Later candles show the book after trading stopped (bid 0, ask 1)."""
    return (
        candles
        .select("ticker",
                pl.from_epoch(pl.col("end_period_ts").cast(pl.Int64), time_unit="s").dt.replace_time_zone("UTC").alias("candle_end"),
                pl.col("yes_bid_close").cast(pl.Float64).alias("final_bid"),
                pl.col("yes_ask_close").cast(pl.Float64).alias("final_ask"))
        .join(markets.select("ticker", "close"), on="ticker")
        .filter(pl.col("candle_end") <= pl.col("close") + pl.duration(hours=1))
        .sort("candle_end")
        .group_by("ticker").last()
        .select("ticker", (pl.col("final_ask") - pl.col("final_bid")).alias("final_spread"))
    )


def contract_types(sample):
    """Paper Table 4 types: Single, Non-exclusive, Exclusive numerical, Exclusive other.
    Contracts per event are counted within the sample (traded contracts that pass the filters), as the paper
    speaks of traded contracts per event; numerical means Kalshi's strike type is greater, less or between."""
    per_event = sample.group_by("event_ticker").agg(pl.col("ticker").n_unique().alias("contracts_in_event"))
    return sample.join(per_event, on="event_ticker").with_columns(
        pl.when(pl.col("contracts_in_event") == 1).then(pl.lit("Single"))
          .when(pl.col("mutually_exclusive").is_null()).then(pl.lit("Unknown"))     # event details not downloaded
          .when(~pl.col("mutually_exclusive")).then(pl.lit("Non-exclusive"))
          .when(pl.col("strike_type").is_in(NUMERICAL_STRIKES)).then(pl.lit("Exclusive numerical"))
          .otherwise(pl.lit("Exclusive other")).alias("contract_type"))


def build_sample(markets, candles):
    """Markets passing the paper's filters, with their period ("paper" or "oos")."""
    paper_start, _ = PERIODS["paper"]
    paper_end, oos_end = PERIODS["oos"]
    sample = (
        markets
        .join(final_spread(markets, candles), on="ticker")
        .filter(
            (pl.col("market_type") == "binary")
            & pl.col("result").is_in(["yes", "no"])
            & (pl.col("volume") >= MIN_VOLUME_DOLLARS)
            & (pl.col("final_spread") >= 0) & (pl.col("final_spread") <= MAX_FINAL_SPREAD)   # a crossed book is bad data
            & ((pl.col("close") - pl.col("open")) >= pl.duration(hours=MIN_HOURS_OPEN))
            & (pl.col("close") >= paper_start) & (pl.col("close") < oos_end)
        )
        .with_columns(pl.when(pl.col("close") < paper_end).then(pl.lit("paper")).otherwise(pl.lit("oos")).alias("period"))
    )
    return contract_types(sample)


def trade_files(period, series):
    """Trade files for one period, keeping only series in the sample. Files are named {series}__{period}[__chunk];
    paper_extra__paper.parquet mixes many series (the markets added for the paper period)."""
    files = sorted((DATA / "trades").glob(f"*__{period}*.parquet"))
    files = [f for f in files if f.name.split("__")[0] in series or f.name.startswith("paper_extra__")]
    return [f for f in files if pl.read_parquet_schema(f)]


def snapshots_from_trades(raw):
    """Price snapshots for the markets in one trade file, plus each market's average trade size in dollars."""
    # De-duplicate on trade_id where the file has it. Older files have no trade_id; they were repaired so each
    # market's trades appear once (REVIEW.md H2), and identical-looking rows there are distinct trades (M4).
    trades = (raw.unique("trade_id") if "trade_id" in raw.columns else raw).select(
        "ticker",
        pl.col("created_time").str.to_datetime(time_zone="UTC", format="%Y-%m-%dT%H:%M:%S%.fZ").alias("time"),
        pl.col("yes_price_dollars").cast(pl.Float64).alias("yes_price"),
        pl.col("count_fp").cast(pl.Float64).alias("contracts"),
        pl.col("taker_outcome_side").alias("taker_side"),
    )
    # When one order fills at several prices the fills share a timestamp. The book is swept from the best
    # price outwards, so the last fill is the highest Yes price when the Taker bought Yes, the lowest when No.
    sweep_order = pl.when(pl.col("taker_side") == "yes").then(pl.col("yes_price")).otherwise(-pl.col("yes_price"))
    trades = trades.sort(pl.col("time"), sweep_order)

    snapshot_times = (
        trades.group_by("ticker").agg(pl.col("time").max().alias("last_time"))
        .join(pl.DataFrame({"days_before": list(range(SNAPSHOT_DAYS + 1))}, schema={"days_before": pl.Int32}), how="cross")
        .with_columns((pl.col("last_time") - pl.duration(days=1) * pl.col("days_before")).alias("snapshot_time"))
        .sort("snapshot_time")
    )
    snapshots = (
        snapshot_times
        .join_asof(trades.select("ticker", "time", "yes_price", "taker_side"),
                   left_on="snapshot_time", right_on="time", by="ticker", strategy="backward",
                   check_sortedness=False)                    # both sides are sorted by time above
        .filter(pl.col("time").is_not_null())                 # the contract had not traded yet
        .select("ticker", "days_before", "time", "yes_price", "taker_side")
    )
    taker_price = pl.when(pl.col("taker_side") == "yes").then(pl.col("yes_price")).otherwise(1 - pl.col("yes_price"))
    trade_size = trades.group_by("ticker").agg((pl.col("contracts") * taker_price).mean().alias("mean_trade_size"))
    return snapshots.join(trade_size, on="ticker")


def build_observations(sample, period):
    """Yes and No observations for one period."""
    in_period = sample.filter(pl.col("period") == period)
    tickers = in_period["ticker"].implode()
    snapshots = pl.concat([snapshots_from_trades(pl.read_parquet(f).filter(pl.col("ticker").is_in(tickers)))
                           for f in trade_files(period, set(in_period["series_ticker"]))])
    snapshots = snapshots.join(in_period.drop("period"), on="ticker")

    # The paper dropped 63 contracts whose last trade did not match Kalshi's reported last price (footnote 2)
    mismatch = snapshots.filter((pl.col("days_before") == 0) & ((pl.col("yes_price") - pl.col("reported_last_price")).abs() > 1e-6))["ticker"]
    snapshots = snapshots.filter(~pl.col("ticker").is_in(mismatch.implode()))
    print(f"{period}: dropped {len(mismatch):,} contracts whose last trade does not match the reported last price (paper: 63)")

    duplicates = snapshots.group_by("ticker", "days_before").len().filter(pl.col("len") > 1)
    assert len(duplicates) == 0, f"{period}: {len(duplicates):,} repeated (ticker, days_before) snapshots"

    yes_rows = snapshots.with_columns(pl.lit("yes").alias("side"), pl.col("yes_price").alias("price"),
                                      (pl.col("result") == "yes").alias("won"))
    no_rows = snapshots.with_columns(pl.lit("no").alias("side"), (1 - pl.col("yes_price")).round(4).alias("price"),
                                     (pl.col("result") == "no").alias("won"))
    return (pl.concat([yes_rows, no_rows])
            .with_columns(price_range(pl.col("price")).alias("price_range"),
                          (pl.col("side") == pl.col("taker_side")).alias("taker"))
            .sort("ticker", "days_before", "side"))


if __name__ == "__main__":
    markets, candles = load_markets_and_candles()
    sample = build_sample(markets, candles)
    sample.select("ticker", "period").write_parquet(DATA / "trade_tickers.parquet")
    print(sample.group_by("period").agg(pl.len().alias("Yes contracts"), pl.col("event_ticker").n_unique().alias("events")))

    for period in PERIODS:
        observations = build_observations(sample, period)
        observations.write_parquet(DATA / f"observations_{period}.parquet")
        yes = observations.filter(pl.col("side") == "yes")
        print(f"{period}: {yes['ticker'].n_unique():,} Yes contracts, {yes['event_ticker'].n_unique():,} events, "
              f"{len(yes):,} Yes prices, {len(observations):,} Yes and No prices -> data/observations_{period}.parquet")

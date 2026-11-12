"""
Build the paper's sample and its price observations from the files in data/.

    .venv/bin/python sample.py

Steps (Bürgi, Deng and Whelan, section 3.1):
1. Load markets and daily candles, including the extra markets in data/paper_extra and data/oos_extra.
2. Apply the paper's filters: settled Yes or No, at least $1,000 volume, open at least 24 hours,
   final bid-ask spread of 20c or less.
3. Price snapshots from the trades: the last trade, and the price as of exactly 1 to 10 days
   (24-hour steps) before it, carrying the most recent earlier trade forward. This reading
   reproduces the paper's Table 1; 24-hour windows, calendar days and windows from the close do not.
4. One Yes and one No observation per snapshot.

Writes data/observations_paper.parquet and data/observations_oos.parquet, which replicate.py reads,
and data/trade_tickers.parquet, which `download_data.py trades` reads.
"""
import polars as pl

from config import (DATA, PERIODS, INCLUDE_EXTRA, MIN_VOLUME_DOLLARS, MAX_FINAL_SPREAD, MIN_HOURS_OPEN,
                    SNAPSHOT_DAYS, NUMERICAL_STRIKES)
from stats import price_range


def read_folder(folder):
    """Stack every non-empty Parquet file in a folder."""
    files = [f for f in sorted(folder.glob("*.parquet")) if pl.read_parquet_schema(f)]
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def load_markets_and_candles():
    """Markets (one row each, typed) and daily candles, with the extra markets added when INCLUDE_EXTRA is on."""
    markets = pl.read_parquet(DATA / "markets.parquet").with_columns(pl.lit(False).alias("extra"))
    candles = pl.read_parquet(DATA / "candles.parquet")
    if INCLUDE_EXTRA:
        extra_markets = [pl.read_parquet(DATA / "paper_extra" / "markets.parquet"), read_folder(DATA / "oos_extra" / "markets")]
        extra_candles = [pl.read_parquet(DATA / "paper_extra" / "candles.parquet"), read_folder(DATA / "oos_extra" / "candles")]
        markets = pl.concat([markets, *[m.with_columns(pl.lit(True).alias("extra")) for m in extra_markets]], how="diagonal_relaxed")
        candles = pl.concat([candles, *extra_candles], how="diagonal_relaxed")
    markets = markets.unique("ticker", keep="first")

    series = pl.read_parquet(DATA / "series.parquet").select(pl.col("ticker").alias("series_ticker"), "category")
    events = pl.read_parquet(DATA / "events.parquet").select(
        "event_ticker", (pl.col("mutually_exclusive") == "True").alias("mutually_exclusive"),
        pl.col("category").alias("event_category"))
    markets = (
        markets
        .select("ticker", "event_ticker", "series_ticker", "market_type", "strike_type", "result", "extra",
                pl.col("volume_fp").cast(pl.Float64).alias("volume"),
                pl.col("last_price_dollars").cast(pl.Float64).alias("reported_last_price"),
                pl.col("open_time").str.to_datetime(time_zone="UTC").alias("open"),
                pl.col("close_time").str.to_datetime(time_zone="UTC").alias("close"))
        .join(series, on="series_ticker", how="left")
        .join(events, on="event_ticker", how="left")
        .with_columns(pl.coalesce("category", "event_category").alias("category"))
        .drop("event_category")
    )
    return markets, candles


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


def contract_types(markets):
    """Paper Table 4 types: Single, Non-exclusive, Exclusive numerical, Exclusive other."""
    per_event = markets.group_by("event_ticker").agg(pl.col("ticker").n_unique().alias("contracts_in_event"))
    return (
        markets.join(per_event, on="event_ticker")
        .select("ticker",
                pl.when(pl.col("contracts_in_event") == 1).then(pl.lit("Single"))
                  .when(~pl.col("mutually_exclusive")).then(pl.lit("Non-exclusive"))
                  .when(pl.col("strike_type").is_in(NUMERICAL_STRIKES)).then(pl.lit("Exclusive numerical"))
                  .otherwise(pl.lit("Exclusive other")).alias("contract_type"))
    )


def build_sample(markets, candles):
    """Markets passing the paper's filters, with their period ("paper" or "oos")."""
    paper_start, _ = PERIODS["paper"]
    paper_end, oos_end = PERIODS["oos"]
    return (
        markets
        .join(final_spread(markets, candles), on="ticker")
        .join(contract_types(markets), on="ticker")
        .filter(
            (pl.col("market_type") == "binary")
            & pl.col("result").is_in(["yes", "no"])
            & (pl.col("volume") >= MIN_VOLUME_DOLLARS)
            & (pl.col("final_spread") <= MAX_FINAL_SPREAD)
            & ((pl.col("close") - pl.col("open")) >= pl.duration(hours=MIN_HOURS_OPEN))
            & (pl.col("close") >= paper_start) & (pl.col("close") < oos_end)
        )
        .with_columns(pl.when(pl.col("close") < paper_end).then(pl.lit("paper")).otherwise(pl.lit("oos")).alias("period"))
    )


def trade_files(period):
    """The trade files for one period: one per series (and chunk), plus the extra markets."""
    files = sorted((DATA / "trades").glob(f"*__{period}.parquet"))
    if INCLUDE_EXTRA:
        files += [DATA / "paper_extra" / "trades.parquet"] if period == "paper" else sorted((DATA / "oos_extra" / "trades").glob("*.parquet"))
    return [f for f in files if pl.read_parquet_schema(f)]


def snapshots_from_trades(raw):
    """Price snapshots for the markets in one trade file, plus each market's average trade size in dollars."""
    trades = raw.select(
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
                           for f in trade_files(period)])
    snapshots = snapshots.join(in_period.drop("period"), on="ticker")

    # The paper dropped 63 contracts whose last trade did not match Kalshi's reported last price (footnote 2)
    mismatch = snapshots.filter((pl.col("days_before") == 0) & ((pl.col("yes_price") - pl.col("reported_last_price")).abs() > 1e-6))["ticker"]
    snapshots = snapshots.filter(~pl.col("ticker").is_in(mismatch.implode()))
    print(f"{period}: dropped {len(mismatch):,} contracts whose last trade does not match the reported last price (paper: 63)")

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

"""
Backtest favourite buying on Kalshi (PLAN.md step 8).

    .venv/bin/python backtest.py --period paper    # choose and check the settings here
    .venv/bin/python backtest.py --period oos      # then run once out of sample with the same config.py

Reads data/markets.parquet, data/candles.parquet (daily candles ending at midnight New York time),
data/events.parquet, data/fee_changes.parquet, data/series.parquet and data/trades/. Writes to results/{period}/:
backtest_summary.csv, backtest_categories.csv, backtest_trades.csv, backtest_nav.csv and backtest_equity.png.

Rules (Jordan, 2026-11-15). Every setting is in config.py.
1. Data at entry. Each day uses only that day's candle: closing bid and ask, the volume traded that day, and the
   days left to scheduled_close (the day in the event ticker, else latest_expiration_time; see sample.py).
   Final volume, final spread, close_time and expected_expiration_time are never used to choose a trade.
2. Universe each day: open binary markets with 1 to 21 days to scheduled_close, at least MIN_DAY_VOLUME contracts
   traded that day, a bid-ask spread of at most MAX_ENTRY_SPREAD that day, and scheduled_close before the end of
   the period (so nothing is held past the data).
3. Strategies (same machinery, different rule). The favourite is the side (Yes or No) whose ask is 85c to 97c.
   A  taker: buy the favourite at the ask.
   B  A, limited to FOCUS_CATEGORIES (Weather, Politics, Financials, Entertainment).
   C  maker: post a limit order at the favourite's bid. It rests for MAKER_ORDER_DAYS and fills only against
      later trades printed at or through that price, up to their volume. No fills are assumed.
   D  C, limited to FOCUS_CATEGORIES.
4. Costs. Taker pays the ask plus the taker fee; maker pays the bid plus the maker fee where the series charges one.
   Paper period: the paper's flat 0.07 taker fee and free makers. Out of sample: the fee in force on that date
   (replicate.fees_in_force). Size is at most VOLUME_SHARE of that day's traded volume, in whole contracts.
5. Sizing. Equal money per event: an event gets at most EVENT_CAP of NAV, split equally between its qualifying
   contracts that day. Open positions (and resting maker orders) count toward the event cap and toward
   CATEGORY_CAP per category, both at cost. No borrowing. Buy once and hold to settlement: no top-ups, no stops.
6. Accounting, every calendar day: NAV = cash + positions marked at that day's last trade price (carried forward;
   the mid if the market has not traded yet). Interest at rate / 365 on the previous day's NAV, at each of
   config.INTEREST_RATES. Settlement pays $1 a contract on the winning side and $0 otherwise, on the close date.

Limitations (in data, not in the rules):
- Candles were only downloaded for markets with at least $1,000 final volume, and trades only for markets that
  also passed the paper's 20c final-spread filter. The universe is therefore mildly selected on final liquidity,
  and maker orders on markets without trade data never fill.
- The ticker-day check (universe "ticker_day") keeps only markets whose close date comes from the event ticker,
  because latest_expiration_time leaks a little in the paper period (sample.py).
"""
import argparse
import datetime as dt
import math
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from config import (DATA, RESULTS, PERIODS, PAPER_CATEGORY, PAPER_FLAT_FEES, TAKER_FEE_RATE, MAKER_FEE_RATE,
                    MAKER_FEES_START, FAVOURITE_MIN, FAVOURITE_MAX, MIN_DAYS_TO_CLOSE, MAX_DAYS_TO_CLOSE,
                    EVENT_CAP, INTEREST_RATES, START_NAV, MIN_DAY_VOLUME, MAX_ENTRY_SPREAD, VOLUME_SHARE, CATEGORY_CAP,
                    MAKER_ORDER_DAYS, FOCUS_CATEGORIES)
from replicate import fees_in_force
from sample import load_markets_and_candles, trade_files
from stats import fee_per_contract

STRATEGIES = {"A": {"maker": False, "focus": False}, "B": {"maker": False, "focus": True},
              "C": {"maker": True, "focus": False}, "D": {"maker": True, "focus": True}}
NEW_YORK = "America/New_York"


# --- Data at entry ---

def daily_quotes(period):
    """One row per market and day in the period: that day's candle, with days to scheduled_close.
    `day` is the New York date the candle covers (it ends at midnight at the start of the next day)."""
    start, end = PERIODS[period]
    markets, candles = load_markets_and_candles()
    markets = (markets.filter(pl.col("result").is_in(["yes", "no"]) & (pl.col("market_type") == "binary")
                              & (pl.col("scheduled_close") < end))
               .with_columns(pl.col("category").replace_strict(PAPER_CATEGORY, default="Other").alias("paper_category"))
               .select("ticker", "event_ticker", "series_ticker", "paper_category", "result", "open", "close",
                       "scheduled_close", "close_date_source"))
    quotes = (
        candles.select(
            "ticker",
            pl.from_epoch(pl.col("end_period_ts").cast(pl.Int64), time_unit="s").dt.replace_time_zone("UTC").alias("time"),
            pl.col("yes_bid_close").cast(pl.Float64).alias("yes_bid"),
            pl.col("yes_ask_close").cast(pl.Float64).alias("yes_ask"),
            pl.col("price_close").cast(pl.Float64).alias("last_price"),
            pl.col("volume").cast(pl.Float64).alias("day_volume"))
        .join(markets, on="ticker")
        .filter((pl.col("time") >= start) & (pl.col("time") < end))
        .sort("ticker", "time")
        .with_columns(
            (pl.col("time").dt.convert_time_zone(NEW_YORK) - pl.duration(hours=1)).dt.date().alias("day"),
            ((pl.col("scheduled_close") - pl.col("time")).dt.total_seconds() / 86_400).alias("days_to_close"),
            pl.col("last_price").forward_fill().over("ticker").alias("last_price"))
    )
    return quotes


def candidates(quotes, period):
    """Rows that pass the daily universe and the favourite band, with the side, prices and fees."""
    in_band = lambda p: (p >= FAVOURITE_MIN - 1e-9) & (p <= FAVOURITE_MAX + 1e-9)
    yes_fav, no_fav = in_band(pl.col("yes_ask")), in_band(1 - pl.col("yes_bid"))
    cands = (
        quotes.filter((pl.col("time") >= pl.col("open")) & (pl.col("time") < pl.col("close"))       # open that day
                      & (pl.col("days_to_close") >= MIN_DAYS_TO_CLOSE) & (pl.col("days_to_close") <= MAX_DAYS_TO_CLOSE)
                      & (pl.col("day_volume") >= MIN_DAY_VOLUME)
                      & (pl.col("yes_bid") > 0) & (pl.col("yes_ask") < 1) & (pl.col("yes_bid") < pl.col("yes_ask"))
                      & (pl.col("yes_ask") - pl.col("yes_bid") <= MAX_ENTRY_SPREAD + 1e-9)
                      & (yes_fav | no_fav))
        .with_columns(pl.when(yes_fav).then(pl.lit("yes")).otherwise(pl.lit("no")).alias("side"))
        .with_columns(
            pl.when(pl.col("side") == "yes").then(pl.col("yes_ask")).otherwise(1 - pl.col("yes_bid")).round(4).alias("ask"),
            pl.when(pl.col("side") == "yes").then(pl.col("yes_bid")).otherwise(1 - pl.col("yes_ask")).round(4).alias("bid"))
    )
    if period == "paper" and PAPER_FLAT_FEES:
        return cands.with_columns(fee_per_contract(pl.col("ask"), TAKER_FEE_RATE).alias("taker_fee"),
                                  pl.lit(0.0).alias("maker_fee"))
    maker_pays = pl.col("fee_type").str.contains("maker") & (pl.col("time") >= MAKER_FEES_START)
    return fees_in_force(cands).with_columns(
        fee_per_contract(pl.col("ask"), TAKER_FEE_RATE, pl.col("fee_multiplier")).alias("taker_fee"),
        pl.when(maker_pays).then(fee_per_contract(pl.col("bid"), MAKER_FEE_RATE, pl.col("fee_multiplier")))
          .otherwise(0.0).alias("maker_fee"))


def maker_fills(cands, period):
    """For each candidate (ticker, day): contracts that later traded at or through our bid while the order rested
    (the MAKER_ORDER_DAYS after the candle), and the New York date of the first such trade.
    Our Yes bid is filled by a trade at a Yes price <= the bid; our No bid by a Yes price >= the Yes ask."""
    orders = cands.select("ticker", "day", "time", "side", "yes_bid", "yes_ask")
    tickers = orders["ticker"].unique().implode()
    found = []
    for f in trade_files(period, set(cands["series_ticker"])):
        raw = pl.read_parquet(f).filter(pl.col("ticker").is_in(tickers))
        if raw.is_empty():
            continue
        trades = (raw.unique("trade_id") if "trade_id" in raw.columns else raw).select(
            "ticker",
            pl.col("created_time").str.to_datetime(time_zone="UTC", format="%Y-%m-%dT%H:%M:%S%.fZ").alias("trade_time"),
            pl.col("yes_price_dollars").cast(pl.Float64).alias("trade_price"),
            pl.col("count_fp").cast(pl.Float64).alias("contracts"))
        found.append(
            orders.join(trades, on="ticker")
            .filter((pl.col("trade_time") > pl.col("time"))
                    & (pl.col("trade_time") <= pl.col("time") + pl.duration(days=MAKER_ORDER_DAYS))
                    & pl.when(pl.col("side") == "yes").then(pl.col("trade_price") <= pl.col("yes_bid") + 1e-9)
                        .otherwise(pl.col("trade_price") >= pl.col("yes_ask") - 1e-9))
            .group_by("ticker", "day").agg(pl.col("contracts").sum().alias("fillable"),
                                           pl.col("trade_time").min().alias("first_fill")))
    fills = (pl.concat(found).group_by("ticker", "day").agg(pl.col("fillable").sum(), pl.col("first_fill").min())
             if found else pl.DataFrame(schema={"ticker": pl.String, "day": pl.Date, "fillable": pl.Float64,
                                                "first_fill": pl.Datetime("us", "UTC")}))
    return (cands.join(fills, on=["ticker", "day"], how="left")
            .with_columns(pl.col("fillable").fill_null(0.0),
                          pl.col("first_fill").dt.convert_time_zone(NEW_YORK).dt.date().alias("fill_day")))


def daily_marks(quotes, tickers):
    """{ticker: {day: Yes mark}} for the given tickers: the last trade price, else the mid."""
    marks = (quotes.filter(pl.col("ticker").is_in(tickers.implode()))
             .select("ticker", "day", pl.coalesce("last_price", (pl.col("yes_bid") + pl.col("yes_ask")) / 2).alias("mark")))
    out = {}
    for ticker, day, mark in marks.iter_rows():
        out.setdefault(ticker, {})[day] = mark
    return out


# --- Simulation ---

def simulate(cands, marks, maker, rate):
    """Run one strategy at one interest rate. Returns the daily NAV table and one row per trade."""
    price_col, fee_col = ("bid", "maker_fee") if maker else ("ask", "taker_fee")
    by_day = {day[0]: rows.sort("event_ticker", "ticker").to_dicts()
              for day, rows in cands.partition_by("day", as_dict=True).items()}
    last_settlement = cands["close"].dt.convert_time_zone(NEW_YORK).dt.date().max()
    days = pl.date_range(min(by_day), last_settlement, eager=True).to_list()

    cash, nav_prev = float(START_NAV), float(START_NAV)
    positions, orders, bought = {}, [], set()      # open positions by ticker, resting maker orders, tickers bought
    nav_rows, trades = [], []
    for day in days:
        cash += nav_prev * rate / 365

        # Maker orders: book fills on the day of the first qualifying trade, cancel the rest when they expire.
        for order in [o for o in orders if o["fill_day"] == day or o["expires"] <= day]:
            orders.remove(order)
            if order["fill_day"] == day and order["filled"] >= 1:
                open_position(positions, bought, order, order["filled"], price_col, fee_col)
                cash -= positions[order["ticker"]]["cost"]

        # Settlement on the close date.
        for ticker in [t for t, p in positions.items() if p["settle_day"] <= day]:
            p = positions.pop(ticker)
            payout = p["contracts"] * (1.0 if p["result"] == p["side"] else 0.0)
            cash += payout
            trades.append({**p["record"], "payout": payout, "pnl": payout - p["cost"], "settle_day": day})

        # Mark open positions.
        for ticker, p in positions.items():
            yes_mark = marks.get(ticker, {}).get(day)
            if yes_mark is not None:
                p["mark"] = yes_mark if p["side"] == "yes" else 1 - yes_mark
        nav = cash + sum(p["contracts"] * p["mark"] for p in positions.values())

        # New orders, event by event. Resting maker orders reserve their cash and count toward the caps.
        reserved = sum(o["reserve"] for o in orders)
        event_cost, category_cost = {}, {}
        for item in list(positions.values()) + orders:
            event_cost[item["event_ticker"]] = event_cost.get(item["event_ticker"], 0) + item["reserve"]
            category_cost[item["paper_category"]] = category_cost.get(item["paper_category"], 0) + item["reserve"]
        busy = set(positions) | {o["ticker"] for o in orders} | bought
        today = [c for c in by_day.get(day, []) if c["ticker"] not in busy]
        for event in sorted({c["event_ticker"] for c in today}):
            contracts = [c for c in today if c["event_ticker"] == event]
            for i, c in enumerate(contracts):
                event_room = EVENT_CAP * nav - event_cost.get(event, 0)
                category_room = CATEGORY_CAP * nav - category_cost.get(c["paper_category"], 0)
                budget = min(event_room / (len(contracts) - i), category_room, cash - reserved)
                unit = c[price_col] + c[fee_col]
                size = math.floor(min(budget / unit, VOLUME_SHARE * c["day_volume"]))
                if size < 1:
                    continue
                cost = size * unit
                event_cost[event] = event_cost.get(event, 0) + cost
                category_cost[c["paper_category"]] = category_cost.get(c["paper_category"], 0) + cost
                reserved += cost
                if maker:
                    expires = day + dt.timedelta(days=MAKER_ORDER_DAYS)
                    orders.append({**c, "reserve": cost, "filled": min(size, math.floor(c["fillable"])),
                                   "fill_day": c["fill_day"], "expires": expires})
                else:
                    open_position(positions, bought, c, size, price_col, fee_col)
                    cash -= cost
                    reserved -= cost
        nav = cash + sum(p["contracts"] * p["mark"] for p in positions.values())
        exposure = sum(p["cost"] for p in positions.values())
        nav_rows.append({"day": day, "nav": nav, "cash": cash, "exposure": exposure, "positions": len(positions)})
        nav_prev = nav
    return pl.DataFrame(nav_rows), pl.DataFrame(trades)


def open_position(positions, bought, c, size, price_col, fee_col):
    """Add a position bought at the candidate's price plus fee."""
    cost = size * (c[price_col] + c[fee_col])
    settle_day = c["close"].astimezone(ZoneInfo(NEW_YORK)).date()
    record = {k: c[k] for k in ["ticker", "event_ticker", "series_ticker", "paper_category", "side", "day",
                                "days_to_close", "close_date_source", "day_volume"]}
    record.update({"price": c[price_col], "fee": c[fee_col], "contracts": size, "cost": cost})
    positions[c["ticker"]] = {"event_ticker": c["event_ticker"], "paper_category": c["paper_category"], "side": c["side"],
                              "result": c["result"], "contracts": size, "cost": cost, "reserve": cost,
                              "mark": c[price_col], "settle_day": settle_day, "record": record}
    bought.add(c["ticker"])


# --- Results ---

def performance(nav, trades, rate):
    """Statistics from the daily NAV series (calendar days, 365 a year) and the trade list."""
    values = np.concatenate([[START_NAV], nav["nav"].to_numpy()])
    daily = values[1:] / values[:-1] - 1
    years = len(daily) / 365
    vol = daily.std(ddof=1) * math.sqrt(365)
    var_95 = -np.quantile(daily, 0.05)
    weekly = values[7:] / values[:-7] - 1
    staked = trades["cost"].sum() if len(trades) else 0.0
    return {
        "annual_return": (values[-1] / START_NAV) ** (1 / years) - 1,
        "volatility": vol,
        "sharpe": (daily.mean() * 365 - rate) / vol if vol > 0 else float("nan"),   # in excess of the interest rate
        "max_drawdown": (1 - values / np.maximum.accumulate(values)).max(),
        "var_95_daily": var_95,
        "es_95_daily": -daily[daily <= -var_95].mean(),
        "worst_week": weekly.min() if len(weekly) else float("nan"),
        "hit_rate": (trades["pnl"] > 0).mean() if len(trades) else float("nan"),   # share of trades that won
        "trades": len(trades),
        "events": trades["event_ticker"].n_unique() if len(trades) else 0,
        "return_on_stake": trades["pnl"].sum() / staked if staked else float("nan"),
        "turnover": staked / nav["nav"].mean() / years,                             # money bought / average NAV, a year
        "average_invested": (nav["exposure"] / nav["nav"]).mean(),
        "final_nav": values[-1],
        "days": len(daily),
    }


def by_category(trades):
    """Trades, money staked, profit after fees and return on money staked per paper category (no interest)."""
    groups = trades.group_by("strategy", "paper_category").agg(
        pl.len().alias("trades"), pl.col("event_ticker").n_unique().alias("events"),
        pl.col("cost").sum().alias("staked"), pl.col("pnl").sum().alias("pnl"),
        (pl.col("pnl") > 0).mean().alias("hit_rate"), pl.col("price").mean().alias("average_price"))
    total = trades.group_by("strategy").agg(
        pl.lit("All").alias("paper_category"), pl.len().alias("trades"), pl.col("event_ticker").n_unique().alias("events"),
        pl.col("cost").sum().alias("staked"), pl.col("pnl").sum().alias("pnl"),
        (pl.col("pnl") > 0).mean().alias("hit_rate"), pl.col("price").mean().alias("average_price"))
    return (pl.concat([groups, total])
            .with_columns((pl.col("pnl") / pl.col("staked")).alias("return_on_stake"),
                          (pl.col("pnl") / START_NAV).alias("contribution_to_nav"))
            .sort("strategy", "paper_category"))


def equity_chart(navs, period, path):
    """NAV of each strategy: solid at 0% interest, dashed at the highest rate."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = dict(zip(STRATEGIES, ["C0", "C1", "C2", "C3"]))
    for (universe, strategy, rate), nav in navs.items():
        if universe != "all" or rate not in (min(INTEREST_RATES), max(INTEREST_RATES)):
            continue
        ax.plot(nav["day"], nav["nav"], color=colours[strategy], ls="-" if rate == min(INTEREST_RATES) else "--", lw=1,
                label=f"{strategy} at {rate:.1%}")
    ax.axhline(START_NAV, color="k", lw=0.6)
    ax.set(title=f"Favourite buying, {period} period (A/B taker, C/D maker; B/D focus categories)",
           ylabel="NAV ($)")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(period):
    out = RESULTS / period
    out.mkdir(parents=True, exist_ok=True)
    quotes = daily_quotes(period)
    cands = maker_fills(candidates(quotes, period), period)
    marks = daily_marks(quotes, cands["ticker"].unique())
    print(f"{period}: {len(cands):,} candidate market-days, {cands['ticker'].n_unique():,} markets, "
          f"{cands['event_ticker'].n_unique():,} events; maker orders with any fill: {(cands['fillable'] >= 1).mean():.1%}")

    universes = {"all": cands, "ticker_day": cands.filter(pl.col("close_date_source") == "day")}
    summary, all_trades, navs = [], [], {}
    for universe, pool in universes.items():
        for name, rule in STRATEGIES.items():
            pool_s = pool.filter(pl.col("paper_category").is_in(FOCUS_CATEGORIES)) if rule["focus"] else pool
            for rate in INTEREST_RATES:
                nav, trades = simulate(pool_s, marks, rule["maker"], rate)
                navs[(universe, name, rate)] = nav
                summary.append({"universe": universe, "strategy": name, "interest_rate": rate,
                                **performance(nav, trades, rate)})
                if rate == INTEREST_RATES[0] and len(trades):
                    all_trades.append(trades.with_columns(pl.lit(universe).alias("universe"), pl.lit(name).alias("strategy")))

    summary = pl.DataFrame(summary)
    trades = pl.concat(all_trades)
    categories = by_category(trades.filter(pl.col("universe") == "all"))
    nav_table = pl.concat([nav.with_columns(pl.lit(u).alias("universe"), pl.lit(s).alias("strategy"), pl.lit(r).alias("interest_rate"))
                           for (u, s, r), nav in navs.items()])

    summary.write_csv(out / "backtest_summary.csv")
    categories.write_csv(out / "backtest_categories.csv")
    trades.write_csv(out / "backtest_trades.csv")
    nav_table.write_csv(out / "backtest_nav.csv")
    equity_chart(navs, period, out / "backtest_equity.png")
    with pl.Config(tbl_rows=60, tbl_cols=20, float_precision=3, tbl_width_chars=220):
        print(summary)
        print(categories)
    print(f"Saved to {out}/backtest_*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--period", choices=["paper", "oos"], default="paper")
    main(parser.parse_args().period)

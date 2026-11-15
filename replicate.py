"""
Replicate the main tables and figures of Bürgi, Deng and Whelan, "Makers and Takers".

    .venv/bin/python replicate.py --period paper    # the paper's sample, 2021 to April 2025
    .venv/bin/python replicate.py --period oos      # the same code, May 2025 to October 2026

Reads data/observations_{period}.parquet (from sample.py). Writes each table as CSV and each figure
as PNG to results/{period}/, and prints them with the paper's numbers alongside.

Definitions (paper sections 3.2 to 4):
- Y is 1 if the contract won, P the price paid in dollars.
- Pre-fee return: (Y - P) / P.
- Post-fee return (equation 2): the fee C is added to the cost, (Y - P - C) / (P + C).
- Fees. Paper period: the paper's flat taker fee (0.07), makers free. Out of sample: the fee in force on the
  snapshot date for that series (multiplier and fee type from data/fee_changes.parquet, falling back to the
  series' current settings), and maker fees (0.0175) on series that charge them (REVIEW.md M1).
- Each snapshot trade had a Taker on one side and a Maker on the other, so half the observations are each.
  On Kalshi a Taker selling Yes is recorded as buying No; for returns this is the same trade.
- The closing-day price is often set after the outcome is all but known (REVIEW.md H3), so the main return
  tables are also shown for 1 to 10 days before the last trade only (suffix _days_1_10).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from config import (DATA, RESULTS, PAPER, PRICE_RANGES, PAPER_CATEGORY, SNAPSHOT_DAYS, TAKER_FEE_RATE, MAKER_FEE_RATE,
                    MAKER_FEES_START, PAPER_FLAT_FEES, FAVOURITE_MIN, FAVOURITE_MAX)
from stats import fee_per_contract, mean_with_ci, mincer_zarnowitz

RANGE_ORDER = pl.DataFrame({"price_range": PRICE_RANGES, "order": range(len(PRICE_RANGES))})


def fees_in_force(obs):
    """Each observation's series fee multiplier and fee type on its snapshot date.
    Kalshi publishes fee changes from October 2025 on: before a series' first change we assume the standard
    taker-only fee (multiplier 1); series that never changed keep their current settings."""
    changes = (pl.read_parquet(DATA / "fee_changes.parquet")
               .select("series_ticker", pl.col("scheduled_ts").str.to_datetime(time_zone="UTC").alias("changed"),
                       pl.col("fee_multiplier").cast(pl.Float64).alias("changed_multiplier"),
                       pl.col("fee_type").alias("changed_type"))
               .sort("changed"))
    current = (pl.read_parquet(DATA / "series.parquet")
               .select(pl.col("ticker").alias("series_ticker"), pl.col("fee_multiplier").cast(pl.Float64).alias("current_multiplier"),
                       pl.col("fee_type").alias("current_type")))
    has_changes = changes.select("series_ticker").unique().with_columns(pl.lit(True).alias("has_changes"))
    return (
        obs.sort("time")
        .join_asof(changes, left_on="time", right_on="changed", by="series_ticker", strategy="backward", check_sortedness=False)
        .join(current, on="series_ticker", how="left").join(has_changes, on="series_ticker", how="left")
        .with_columns(
            pl.when(pl.col("changed").is_not_null()).then(pl.col("changed_multiplier"))
              .when(pl.col("has_changes")).then(pl.lit(1.0))
              .otherwise(pl.col("current_multiplier").fill_null(1.0)).alias("fee_multiplier"),
            pl.when(pl.col("changed").is_not_null()).then(pl.col("changed_type"))
              .when(pl.col("has_changes")).then(pl.lit("quadratic"))
              .otherwise(pl.col("current_type").fill_null("quadratic")).alias("fee_type"))
        .drop("changed", "changed_multiplier", "changed_type", "current_multiplier", "current_type", "has_changes")
    )


def load(period):
    """Observations with fees, returns, paper categories and year added."""
    won = pl.col("won").cast(pl.Float64)
    obs = pl.read_parquet(DATA / f"observations_{period}.parquet")
    if period == "paper" and PAPER_FLAT_FEES:
        obs = obs.with_columns(pl.when(pl.col("taker")).then(fee_per_contract(pl.col("price"), TAKER_FEE_RATE))
                               .otherwise(0.0).alias("fee"))
    else:
        maker_pays = pl.col("fee_type").str.contains("maker") & (pl.col("time") >= MAKER_FEES_START)
        obs = fees_in_force(obs).with_columns(
            pl.when(pl.col("taker")).then(fee_per_contract(pl.col("price"), TAKER_FEE_RATE, pl.col("fee_multiplier")))
              .when(maker_pays).then(fee_per_contract(pl.col("price"), MAKER_FEE_RATE, pl.col("fee_multiplier")))
              .otherwise(0.0).alias("fee"))
    return (
        obs.with_columns(pl.col("category").replace_strict(PAPER_CATEGORY, default="Other").alias("paper_category"),
                         pl.col("close").dt.year().alias("year"))
        .with_columns(((won - pl.col("price")) / pl.col("price")).alias("return_pre_fee"),
                      ((won - pl.col("price") - pl.col("fee")) / (pl.col("price") + pl.col("fee"))).alias("return_post_fee"))
    )


def by_price_range(df):
    """Sort a table that has a price_range column into the paper's order."""
    return df.join(RANGE_ORDER, on="price_range").sort("order").drop("order")


# --- Tables ---

def headline_counts(obs):
    yes = obs.filter(pl.col("side") == "yes")
    main = yes.filter(pl.col("source") == "listing")
    count = lambda d: [d["event_ticker"].n_unique(), d["ticker"].n_unique(), len(d), 2 * len(d)]
    return pl.DataFrame({"count": list(PAPER["counts"]), "ours": count(yes),
                         "listing_only": count(main), "paper": list(PAPER["counts"].values())})


def table_1(obs):
    """Yes contracts by the furthest day before close with a price."""
    return (obs.filter(pl.col("side") == "yes")
            .group_by("ticker").agg(pl.col("days_before").max().alias("furthest_day"))
            .group_by("furthest_day").len().sort("furthest_day")
            .join(pl.DataFrame({"furthest_day": range(SNAPSHOT_DAYS + 1), "paper": PAPER["table_1"]},
                               schema={"furthest_day": pl.Int32, "paper": pl.Int64}), on="furthest_day", how="left"))


def table_2(obs):
    """Yes and No observations by price range."""
    return (by_price_range(obs.group_by("price_range").len())
            .with_columns(pl.Series("paper", PAPER["table_2"]),
                          (100 * pl.col("len") / pl.col("len").sum()).round(1).alias("percent"),
                          (100 * pl.Series(PAPER["table_2"]) / sum(PAPER["table_2"])).round(1).alias("paper_percent")))


def average_returns(obs):
    makers = obs.filter(~pl.col("taker"))
    ours = {"all contracts, pre-fee": obs["return_pre_fee"].mean(),
            "all contracts, post-fee": obs["return_post_fee"].mean(),
            "Takers, post-fee": obs.filter(pl.col("taker"))["return_post_fee"].mean(),
            "Makers": makers["return_post_fee"].mean(),
            "Makers at 50c and over": makers.filter(pl.col("price") >= 0.50)["return_post_fee"].mean()}
    return pl.DataFrame({"measure": list(ours), "ours": list(ours.values()),
                         "paper": [PAPER["average_returns"].get(k) for k in ours]})


def returns_by_price(obs):
    """Win rate and returns by price range (Figure 5) and Makers vs Takers (Table 10, Figure 6)."""
    return by_price_range(
        obs.group_by("price_range").agg(
            pl.len().alias("observations"),
            pl.col("won").mean().alias("win_rate"),
            pl.col("price").mean().alias("avg_price"),
            pl.col("return_pre_fee").mean().alias("return_pre_fee"),
            pl.col("return_post_fee").filter(pl.col("taker")).mean().alias("taker_return"),
            pl.col("return_post_fee").filter(~pl.col("taker")).mean().alias("maker_return"),
            (100 * (~pl.col("taker")).mean()).alias("maker_share"),
        ).join(mean_with_ci(obs, "price_range", "return_post_fee").select(
            "price_range", pl.col("mean").alias("return_post_fee"), "ci_low", "ci_high"), on="price_range")
    ).with_columns(pl.Series("paper_maker_share", PAPER["table_10_maker_share"]))


def regressions(obs):
    """Mincer-Zarnowitz regressions on Yes contracts: Tables 4, 5, 6, 7, 8 and 9 in one long table."""
    yes = obs.filter(pl.col("side") == "yes")
    closing = yes.filter(pl.col("days_before") == 0)
    quintile = lambda c: (pl.col(c).rank("ordinal") * 5 / (pl.len() + 1)).floor().cast(pl.Int32).add(1)
    closing = closing.with_columns(quintile("volume").alias("volume_q"), quintile("mean_trade_size").alias("size_q"))
    event = ("event_ticker",)

    rows = []
    def add(table, group, subset, paper_psi, cluster_by=("event_ticker", "ticker")):
        if len(subset) > 2:
            rows.append({"table": table, "group": str(group), **mincer_zarnowitz(subset, cluster_by), "paper_psi": paper_psi})

    t4 = PAPER["table_4_psi"]
    add("4", "All", yes, t4["All"])
    add("4", "Single", yes.filter(pl.col("contract_type") == "Single"), t4["Single"], cluster_by=("ticker",))
    for kind in ["Non-exclusive", "Exclusive numerical", "Exclusive other"]:
        add("4", kind, yes.filter(pl.col("contract_type") == kind), t4[kind])
    add("4", "All exclusive", yes.filter(pl.col("contract_type").str.starts_with("Exclusive")), t4["All exclusive"])
    for k in range(SNAPSHOT_DAYS + 1):
        add("5", f"{k} days", yes.filter(pl.col("days_before") == k), PAPER["table_5_psi"][k], cluster_by=event)
    for q in range(1, 6):
        add("6", f"volume quintile {q}", closing.filter(pl.col("volume_q") == q), PAPER["table_6_psi"][q - 1], cluster_by=event)
        add("7", f"trade size quintile {q}", closing.filter(pl.col("size_q") == q), PAPER["table_7_psi"][q - 1], cluster_by=event)
    for category, psi in PAPER["table_8_psi"].items():
        if category != "Other":
            add("8", category, yes.filter(pl.col("paper_category") == category), psi)
    add("8", "Other excl. Sports", yes.filter(pl.col("paper_category") == "Other"), None)
    add("8", "Sports", yes.filter(pl.col("paper_category") == "Sports"), None)
    add("8", "Other incl. Sports (paper's Other)", yes.filter(pl.col("paper_category").is_in(["Other", "Sports"])),
        PAPER["table_8_psi"]["Other"])
    for year in sorted(yes["year"].unique()):
        add("9", year, yes.filter(pl.col("year") == year), PAPER["table_9_psi"].get(year))
    return pl.DataFrame(rows).sort("table", maintain_order=True)


def table_3(obs):
    """Average final volume by decile of closing Yes contracts (paper Table 3). Checks that volume in
    contracts is the paper's dollar volume."""
    closing = obs.filter((pl.col("side") == "yes") & (pl.col("days_before") == 0))
    return (closing.with_columns((pl.col("volume").rank("ordinal") * 10 / (pl.len() + 1)).floor().cast(pl.Int32).add(1).alias("decile"))
            .group_by("decile").agg(pl.col("volume").mean().round(0).alias("mean_volume"), pl.len()).sort("decile")
            .with_columns(pl.Series("paper_mean_volume", PAPER["table_3_mean_volume"])))


def favourites_by_day(obs):
    """Favourites priced FAVOURITE_MIN to FAVOURITE_MAX, average return after fees by days before the last trade,
    for Takers and Makers, with event-clustered intervals. The strategy question (REVIEW.md H3)."""
    fav = obs.filter((pl.col("price") >= FAVOURITE_MIN) & (pl.col("price") <= FAVOURITE_MAX))
    taker = mean_with_ci(fav.filter(pl.col("taker")), "days_before", "return_post_fee").rename(
        {"mean": "taker_return", "ci_low": "taker_ci_low", "ci_high": "taker_ci_high", "n": "taker_n"})
    maker = mean_with_ci(fav.filter(~pl.col("taker")), "days_before", "return_post_fee").select(
        "days_before", pl.col("mean").alias("maker_return"), pl.col("n").alias("maker_n"))
    return taker.drop("clusters").join(maker, on="days_before").sort("days_before")


def early_close(obs):
    """Leakage checks for close times (REVIEW.md H4), one row per contract (closing Yes snapshot).

    - scheduled_close, by source (day, month, year from the ticker; latest = latest_expiration_time). Markets
      that closed more than a day before it closed early; if their Yes rate differs, the final close leaks.
    - expected_expiration: a value on today's market record. If its position relative to the close predicts
      the outcome, it was updated after the fact and leaks too.
    """
    contracts = obs.filter((pl.col("side") == "yes") & (pl.col("days_before") == 0))
    by_ticker = (
        contracts.with_columns(
            pl.concat_str(pl.col("close_date_source"), pl.lit(": "),
                          pl.when(pl.col("close") < pl.col("scheduled_close") - pl.duration(days=1))
                            .then(pl.lit("closed > 1 day before scheduled close"))
                            .otherwise(pl.lit("closed on or after scheduled close"))).alias("group"))
        .group_by("group").agg(pl.len().alias("contracts"), (pl.col("result") == "yes").mean().alias("yes_rate"))
        .with_columns(pl.lit("scheduled_close").alias("check"))
    )
    by_expected = (
        contracts.with_columns(
            pl.when(pl.col("expected_expiration").is_null()).then(pl.lit("missing"))
              .when(pl.col("expected_expiration") < pl.col("close")).then(pl.lit("expected expiration before close"))
              .when(pl.col("expected_expiration") > pl.col("close") + pl.duration(days=1)).then(pl.lit("expected expiration > 1 day after close"))
              .otherwise(pl.lit("expected expiration within a day after close")).alias("group"))
        .group_by("group").agg(pl.len().alias("contracts"), (pl.col("result") == "yes").mean().alias("yes_rate"))
        .with_columns(pl.lit("expected_expiration").alias("check"))
    )
    return pl.concat([by_ticker, by_expected]).select("check", "group", "contracts", "yes_rate").sort("check", "group")


def returns_by_group(obs, column):
    return (obs.group_by(column).agg(pl.len().alias("observations"),
                                     pl.col("return_pre_fee").mean(), pl.col("return_post_fee").mean())
            .sort(column))


# --- Figures ---

def figure_win_rate_and_returns(obs, by_price, path):
    """Paper Figures 3 and 5: win rate by price, and returns including Taker fees by price range."""
    by_cent = mean_with_ci(obs.with_columns((pl.col("price") * 100).round(0).alias("cents"), pl.col("won").cast(pl.Float64)),
                           "cents", "won").sort("cents")
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    left.fill_between(by_cent["cents"], 100 * by_cent["ci_low"], 100 * by_cent["ci_high"], alpha=0.3)
    left.plot(by_cent["cents"], 100 * by_cent["mean"], lw=1)
    left.plot([0, 100], [0, 100], "k--", lw=0.8)
    left.set(title="Win rate by price (Figure 3)", xlabel="Price (cents)", ylabel="Win rate (%)")

    x = range(len(PRICE_RANGES))
    right.bar(x, 100 * by_price["return_post_fee"],
              yerr=[100 * (by_price["return_post_fee"] - by_price["ci_low"]), 100 * (by_price["ci_high"] - by_price["return_post_fee"])])
    right.axhline(0, color="k", lw=0.8)
    right.set_xticks(list(x), PRICE_RANGES, rotation=45)
    right.set(title="Return incl. Taker fees by price range (Figure 5)", ylabel="Average return (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_makers_takers(by_price, path):
    """Paper Figure 6: returns by price range for Takers and Makers."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(PRICE_RANGES)))
    ax.bar([i - 0.2 for i in x], 100 * by_price["taker_return"], width=0.4, label="Takers (after fees)")
    ax.bar([i + 0.2 for i in x], 100 * by_price["maker_return"], width=0.4, label="Makers")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, PRICE_RANGES, rotation=45)
    ax.set(title="Return by price range, Takers vs Makers (Figure 6)", ylabel="Average return (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Run ---

def show(name, table, out):
    """Print a table and save it as CSV."""
    table.write_csv(out / f"{name}.csv")
    with pl.Config(tbl_rows=40, tbl_cols=12, float_precision=3, tbl_width_chars=160):
        print(f"\n{name}\n{table}")


def main(period):
    out = RESULTS / period
    out.mkdir(parents=True, exist_ok=True)
    obs = load(period)
    print(f"Period: {period}. The 'paper' columns are the paper's published numbers for 2021 to April 2025.")

    show("counts", headline_counts(obs), out)
    show("table_1", table_1(obs), out)
    show("table_2", table_2(obs), out)
    show("average_returns", average_returns(obs), out)
    by_price = returns_by_price(obs)
    show("returns_by_price", by_price, out)
    show("regressions", regressions(obs), out)
    show("returns_by_category", returns_by_group(obs, "paper_category"), out)
    show("returns_by_year", returns_by_group(obs, "year"), out)
    show("table_3", table_3(obs), out)

    later = obs.filter(pl.col("days_before") >= 1)        # without the closing-day price (REVIEW.md H3)
    show("average_returns_days_1_10", average_returns(later), out)
    show("returns_by_price_days_1_10", returns_by_price(later), out)
    show("favourites_by_day", favourites_by_day(obs), out)
    show("early_close", early_close(obs), out)

    figure_win_rate_and_returns(obs, by_price, out / "figure_3_5.png")
    figure_makers_takers(by_price, out / "figure_6.png")
    print(f"\nTables and figures saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--period", choices=["paper", "oos"], default="paper")
    main(parser.parse_args().period)

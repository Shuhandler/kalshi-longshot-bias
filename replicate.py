"""
Replicate the main tables and figures of Bürgi, Deng and Whelan, "Makers and Takers".

    .venv/bin/python replicate.py --period paper    # the paper's sample, 2021 to April 2025
    .venv/bin/python replicate.py --period oos      # the same code, May 2025 to October 2026

Reads data/observations_{period}.parquet (from sample.py). Writes each table as CSV and each figure
as PNG to results/{period}/, and prints them with the paper's numbers alongside.

Definitions (paper sections 3.2 to 4):
- Y is 1 if the contract won, P the price paid in dollars.
- Pre-fee return: (Y - P) / P.
- Post-fee return for Takers (equation 2): the fee C is added to the cost, (Y - P - C) / (P + C).
  Makers paid no fees in the paper's sample; we apply the same rule out of sample so the code is unchanged.
- Each snapshot trade had a Taker on one side and a Maker on the other, so half the observations are each.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from config import DATA, RESULTS, PAPER, PRICE_RANGES, PAPER_CATEGORY, SNAPSHOT_DAYS
from stats import taker_fee_per_contract, mean_with_ci, mincer_zarnowitz

RANGE_ORDER = pl.DataFrame({"price_range": PRICE_RANGES, "order": range(len(PRICE_RANGES))})


def load(period):
    """Observations with returns, paper categories and year added."""
    won = pl.col("won").cast(pl.Float64)
    return (
        pl.read_parquet(DATA / f"observations_{period}.parquet")
        .with_columns(taker_fee_per_contract(pl.col("price")).alias("fee"),
                      pl.col("category").replace_strict(PAPER_CATEGORY, default="Other").alias("paper_category"),
                      pl.col("close").dt.year().alias("year"))
        .with_columns(
            ((won - pl.col("price")) / pl.col("price")).alias("return_pre_fee"),
            pl.when(pl.col("taker"))
              .then((won - pl.col("price") - pl.col("fee")) / (pl.col("price") + pl.col("fee")))
              .otherwise((won - pl.col("price")) / pl.col("price"))
              .alias("return_post_fee"),
        )
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
            *mean_with_ci(pl.col("return_post_fee")),
            pl.col("return_post_fee").filter(pl.col("taker")).mean().alias("taker_return"),
            pl.col("return_post_fee").filter(~pl.col("taker")).mean().alias("maker_return"),
            (100 * (~pl.col("taker")).mean()).alias("maker_share"),
        ).rename({"mean": "return_post_fee"})
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
    for category, psi in {**PAPER["table_8_psi"], "Sports": None}.items():
        add("8", category, yes.filter(pl.col("paper_category") == category), psi)
    for year in sorted(yes["year"].unique()):
        add("9", year, yes.filter(pl.col("year") == year), PAPER["table_9_psi"].get(year))
    return pl.DataFrame(rows).sort("table", maintain_order=True)


def returns_by_group(obs, column):
    return (obs.group_by(column).agg(pl.len().alias("observations"),
                                     pl.col("return_pre_fee").mean(), pl.col("return_post_fee").mean())
            .sort(column))


# --- Figures ---

def figure_win_rate_and_returns(obs, by_price, path):
    """Paper Figures 3 and 5: win rate by price, and returns including Taker fees by price range."""
    by_cent = (obs.with_columns((pl.col("price") * 100).round(0).alias("cents"))
               .group_by("cents").agg(*mean_with_ci(pl.col("won").cast(pl.Float64))).sort("cents"))
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

    figure_win_rate_and_returns(obs, by_price, out / "figure_3_5.png")
    figure_makers_takers(by_price, out / "figure_6.png")
    print(f"\nTables and figures saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--period", choices=["paper", "oos"], default="paper")
    main(parser.parse_args().period)

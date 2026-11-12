"""
Small statistical helpers shared by the analysis scripts.
"""
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

from config import TAKER_FEE_RATE, FEE_ORDER_SIZE, PRICE_RANGES


def taker_fee_per_contract(price):
    """Taker fee per contract in dollars (Polars expression). Kalshi charges 0.07 * contracts * p * (1 - p),
    rounded up to the cent on the whole order; the paper prices an order of 100 contracts."""
    order_fee = (TAKER_FEE_RATE * FEE_ORDER_SIZE * price * (1 - price) * 100 - 1e-9).ceil() / 100
    return order_fee / FEE_ORDER_SIZE


def price_range(price):
    """The paper's 10c price ranges (Table 2) as a Polars expression."""
    cents = (price * 100).round(2)
    return (pl.when(cents <= 10).then(pl.lit(PRICE_RANGES[0]))
            .when(cents <= 20).then(pl.lit(PRICE_RANGES[1]))
            .when(cents <= 30).then(pl.lit(PRICE_RANGES[2]))
            .when(cents <= 40).then(pl.lit(PRICE_RANGES[3]))
            .when(cents < 50).then(pl.lit(PRICE_RANGES[4]))
            .when(cents < 60).then(pl.lit(PRICE_RANGES[5]))
            .when(cents < 70).then(pl.lit(PRICE_RANGES[6]))
            .when(cents < 80).then(pl.lit(PRICE_RANGES[7]))
            .when(cents < 90).then(pl.lit(PRICE_RANGES[8]))
            .otherwise(pl.lit(PRICE_RANGES[9])))


def mean_with_ci(values):
    """Mean, 95% confidence interval and count as Polars expressions, for group_by().agg()."""
    se = values.std() / values.count().sqrt()
    return [values.mean().alias("mean"), (values.mean() - 1.96 * se).alias("ci_low"),
            (values.mean() + 1.96 * se).alias("ci_high"), values.count().alias("n")]


def mincer_zarnowitz(df, cluster_by=("event_ticker", "ticker")):
    """The paper's equation 3 on Yes contracts, in cents: (100 * Y - price) = a + psi * price.

    Standard errors are clustered on the given columns (event and contract by default).
    Returns one dict: psi, constant, their standard errors, observations and the F-test p-value of a = psi = 0.
    """
    data = df.select(
        (100 * pl.col("won").cast(pl.Float64) - 100 * pl.col("price")).alias("profit"),
        (100 * pl.col("price")).alias("price"),
        *cluster_by,
    )
    data = pd.DataFrame({c: data[c].to_numpy() for c in data.columns})     # no pyarrow needed
    groups = np.column_stack([pd.factorize(data[c])[0] for c in cluster_by])
    fit = smf.ols("profit ~ price", data).fit(
        cov_type="cluster", cov_kwds={"groups": groups if len(cluster_by) > 1 else groups[:, 0]})
    return {"psi": fit.params["price"], "psi_se": fit.bse["price"],
            "constant": fit.params["Intercept"], "constant_se": fit.bse["Intercept"],
            "observations": int(fit.nobs), "F_test_p": float(fit.f_test("Intercept = 0, price = 0").pvalue)}

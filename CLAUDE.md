# kalshi-longshot-bias

Columbia hedge fund strategies class, Module 3: replicate the favourite-longshot bias on Kalshi
(Bürgi, Deng and Whelan, "Makers and Takers"), extend it out of sample, and backtest favourite
buying (85–97c favourites, interest carry on cash). The full plan is in [PLAN.md](PLAN.md).

## Code style

- **The code must not be overly complex. It must be readable and reproducible.**
- Two files only (Jordan, 2026-11-11): `download_data.py` downloads from the Kalshi API, and the notebook
  `kalshi_longshot.ipynb` does all cleaning and evaluation. The notebook never calls the API. No package,
  no other scripts, and do not use Jordan's research toolkit here.
- Plain pandas, Polars, statsmodels, matplotlib and requests. Prefer a short, obvious function over a
  clever one. Every constant is defined once: download settings at the top of `download_data.py`, analysis
  settings (dates, filters, fees, interest rate) in the notebook's Setup section.
- The notebook must run top to bottom from the files in `data/`. The script caches per series, so a rerun
  resumes instead of downloading again.

## Environment

- Virtual environment: `.venv` (Python 3.13). Recreate with
  `python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
- Jupyter kernel: "Python (kalshi-longshot)".
- The Kalshi API is only reachable from Jordan's Mac, not from the cloud environment.

## Data

- Base URL `https://external-api.kalshi.com/trade-api/v2`. Requests are signed with the key in `.env` (Advanced tier,
  30 requests a second; candles use 2 workers). All market data used here (series, events,
  markets, candlesticks, trades) is public and needs no authentication.
- Markets that settled before `GET /historical/cutoff` are only in `/historical/markets`,
  `/historical/markets/{ticker}/candlesticks` and `/historical/trades`; newer ones are in the live
  `/markets`, `/series/{series}/markets/{ticker}/candlesticks` and `/markets/trades`. Pull both.
- Historical markets can only be filtered by series or event ticker, so the pull loops over every series
  from `GET /series`. Old tickers were renamed (`FED-21DEC` sits under series `KXFED`).
- Sports series are left out of every pull for now (Jordan, 2026-11-11).
- Exception for the paper period only (Jordan, 2026-11-12): `download_data.py paper-extra` downloads every
  exchange trade by day (Jul 2021 to Apr 2025) and adds the traded markets our series listing skipped
  (sports, and daily markets inside "hourly" crypto/index series such as KXBTCD and INXU) under
  `data/paper_extra/`. The notebook's 24-hour filter keeps truly hourly markets out; flag `INCLUDE_PAPER_EXTRA`.
- `download_data.py events` saves event details (incl. `mutually_exclusive`) for every sample event to
  `data/events.parquet`; used for the paper's Table 4 contract types.
- Skip multivariate combo series (tickers starting `KXMVE`); they dominate the raw listing. `mve_filter`
  cannot be combined with `series_ticker`.
- Usable history starts July 2021. Prices come as dollar strings (`*_dollars`), volumes as `*_fp` strings.
- Cache everything as Parquet (Polars) under `data/`, which is git-ignored. The script writes combined
  `data/series.parquet`, `markets.parquet`, `candles.parquet` (and trades with the `trades` argument,
  for tickers the notebook saves to `data/trade_tickers.parquet`). Trades stay one file per series and period
  (`data/trades/{series}__{paper|oos}.parquet`), with no combined file; read them lazily with `pl.scan_parquet`.
- Trades have their own archive cutoff: a live market can have older trades in `/historical/trades`, so live
  markets query both trade endpoints.

## Secrets

- The key ID and private key path are in `.env`; the private keys are in `keys/`. Both are git-ignored.
  Never print, copy or commit them.

## Sources

- Never write a citation or a figure from memory. Check it against the paper PDF or the Kalshi docs.

# Kalshi favourite-longshot replication plan

2026-11-11 · Jordan Shuhandler

Copied from the Claude Doc "Kalshi favourite-longshot replication plan"
(https://claude.ai/artifact/877yP43FMri2Lyx6zdkM7u). The only change is that the
notebook lives in this repo rather than the Module 3 folder.

## Goal and scope

The project has three layers, built in order: **replicate** the paper's main tables on its own sample (Kalshi, 2021 to April 2025), **extend** the same tables out of sample (May 2025 to October 2026), then **trade** the finding as a weekly-rebalanced favourite-buying portfolio with a calibration model as the improvement step.

The paper is [Bürgi, Deng and Whelan, "Makers and Takers: The Economics of the Kalshi Prediction Market"](https://www.karlwhelan.com/Papers/Kalshi.pdf) (Jan 2026 version; earlier [UCD WP25/19](https://www.ucd.ie/economics/t4media/WP2025_19.pdf)). It is descriptive: it measures returns by price and by maker/taker side, but never tests a trading strategy, so layer 3 is our own construction.

The code is plain Python scripts: `download_data.py` pulls the Kalshi data, and `config.py`, `stats.py`, `sample.py` and `replicate.py` (later `backtest.py` and `calibration.py`) do the cleaning and evaluation (see the code structure section). They use plain pandas, Polars, statsmodels and matplotlib, and do not depend on the research toolkit.

## Decisions to make before building

These are the design choices worth settling first; the suggested default is listed for each.

| Decision | Options | Suggested default |
|---|---|---|
| Price used for returns | Last trade, mid, or ask at a fixed time before close | Match the paper for replication (check its data section); use the ask for the strategy |
| Snapshot times | Final price only, or prices N days before close | The paper's pre-closing days for replication; daily snapshots for the strategy |
| Sports | Exclude, include, or separate check | Exclude from the main results, report separately |
| Out-of-sample window | May 2025 to latest settled, or a fixed end date | May 2025 to 31 Oct 2026, frozen once chosen |
| Strategy execution | Taker (cross the spread) or maker (rest at the bid) | Taker as the main case, maker as a sensitivity |
| Position sizing | Equal per event, or scaled by model edge | Equal per event with 3% event cap and a category cap |
| Storage format | CSV or Parquet via Polars | Parquet via Polars (no pyarrow needed) |

## Step-by-step plan

1. **Read the paper's data and method sections closely.** Note the exact price definition, the snapshot days before close, how returns are computed for Yes and No, how makers and takers are identified, and which tables and figures you will reproduce (Table 2 counts, the return-by-price figure, the price regression, maker vs taker returns).
2. **Test Kalshi data access from your Mac.** The cloud environment cannot reach Kalshi, so run the first notebook cells via Remote Control. Base URL `https://external-api.kalshi.com/trade-api/v2`; check `/historical/cutoff`, list settled markets, and pull candlesticks and trades for a few tickers. Confirm the fields you need exist: settlement result, open and close times, volume, series category, bid/ask in candles, and the taker side on trades. Note the rate limits.
3. **Write the data download functions in `download_data.py`.** Paginated downloaders for series, events, markets, candlesticks and trades, with caching to disk so the full pull can resume after interruption (see the code structure section).
4. **Pull the full history.** All settled markets from 2021 to the latest settled date, then candles and trades for markets that pass a loose pre-filter. Expect this to take hours; run it once and keep the cache.
5. **Build the paper's sample.** Apply its filters: at least $1,000 final volume, final spread of 20c or less, open at least 24 hours, hourly crypto and index markets excluded, sample ending April 2025. Create one row per contract side (Yes and No) per snapshot. Check your counts against the paper's 46,282 contracts and 12,403 events.
6. **Replicate the main results.** Win rate and average return by 10c price bucket, before and after the taker fee; the overall average return (about -20% before fees, -22% after); the regression of returns on price; maker vs taker average returns using the trade-level taker side; results by category and by year.
7. **Extend out of sample.** Rerun step 6 unchanged on May 2025 to October 2026. This is the first real test: does the bias still exist after publication, and is it weaker?
8. **Backtest the rules-based strategy.** Weekly, buy the leading side of qualifying non-sports contracts priced 85–97c with 1–21 days to close, using liquidity measured on the entry date. Equal weight per event, 3% event cap, category cap, hold to settlement, taker fees and spread, 3.5% interest on cash and positions. Report annual return, volatility, Sharpe, maximum drawdown, VaR and expected shortfall, hit rate and worst week, all computed in `backtest.py`.
9. **Add the calibration model.** Fit a logistic regression of win on log-odds of price, days to close and category using only pre-May-2025 data, check calibration, freeze it, and trade only where the predicted edge beats costs. Compare it with step 8 out of sample.
10. **Write up.** Strategy summary, origin, replication results, out-of-sample results, backtest, improvement, and a judgment section on skew, capacity, crowding and legal risk.

## Code structure

The analysis moved from the notebook into plain scripts on 2026-11-12 (the notebook is kept but frozen). `download_data.py` writes the data to `data/`; the other scripts read only from there.

| Script | Contents |
|---|---|
| `download_data.py` | Paginated, signed downloaders for series, markets, candlesticks, trades and events, with on-disk caching and resume; run from the Mac |
| `config.py` | Sample dates, filters, fee rate, interest rate, strategy settings, the paper's published numbers |
| `stats.py` | Taker fee with round-up, price ranges, mean with confidence interval, Mincer-Zarnowitz regression with clustered errors |
| `sample.py` | Paper filters, price snapshots, Yes/No observations for both periods (steps 5 and 7 inputs) |
| `replicate.py --period paper` | Step 6: Tables 1, 2, 4 to 10 and Figures 3, 5, 6, each next to the paper's number |
| `replicate.py --period oos` | Step 7: the same code on May 2025 to October 2026 |
| `backtest.py` (to do) | Step 8: rules-based favourite buying, NAV, performance and tail statistics |
| `calibration.py` (to do) | Step 9: fit on pre-May-2025 data, reliability check, out-of-sample comparison with step 8 |

The settlement portfolio function is the largest piece of new code. Keep it general (any table of entries with instrument, entry date, price, size and settlement value) so `backtest.py` and `calibration.py` share it.

## Validation targets and risks

The replication is on track if it lands close to these published figures; small gaps are expected because the API may differ from the authors' data pull.

| Result | Paper's figure |
|---|---|
| Sample size | 46,282 contracts, 12,403 events, 313,972 Yes and No observations |
| Average return, all contracts | About -20% before fees, -22% after |
| Contracts under 10c | Lose over 60% of money staked |
| Contracts above 70c | Small positive returns after fees, statistically significant |
| Takers vs makers | About -32% vs about -10% |
| Taker fee | 0.07 × price × (1 − price) per contract, rounded up to the cent |
| 2025 sub-sample | Bias weaker and less significant |

Some of these figures came from summaries of the paper's later sections; confirm each against the PDF before relying on it.

**Risks to plan for:**

- **Data gaps.** The API may not serve the oldest markets, bid/ask history, or taker side for every trade. If so, document the gap and replicate on what is available.
- **Pull time and rate limits.** Tens of thousands of markets, each needing separate calls. Cache everything and make the downloader resumable.
- **Look-ahead bias.** The paper filters on final volume and final spread. That is fine for replication, but the strategy must use values known on the entry date.
- **A null result.** Deleep et al. (2026) argue the bias is largely a timing artifact, and it was already weaker in 2025. A finding that the edge has gone after costs is still a valid result for the write-up.
- **Fee schedule changes.** Kalshi started charging makers after April 2025 and updated the schedule in July 2026; apply the fee in force on each trade date.

## Suggested timeline

Module 3 runs over weeks 8 to 12 at 3 to 6 hours a week, so roughly one layer per week. The deadline and deliverables are still unconfirmed; check the syllabus.

1. **Week 8:** steps 1 to 3, covering the paper read, data access test and adapter.
2. **Week 9:** steps 4 and 5, covering the full pull and the paper's sample with counts checked.
3. **Week 10:** steps 6 and 7, the replication and out-of-sample tables.
4. **Week 11:** steps 8 and 9, the strategy backtest and calibration model.
5. **Week 12:** step 10, the write-up and robustness checks.

## Sources

- [Bürgi, Deng and Whelan (Jan 2026)](https://www.karlwhelan.com/Papers/Kalshi.pdf) and [UCD WP25/19](https://www.ucd.ie/economics/t4media/WP2025_19.pdf)
- [Kalshi historical data docs](https://docs.kalshi.com/getting_started/historical_data.md) and [candlesticks endpoint](https://docs.kalshi.com/api-reference/market/get-market-candlesticks.md)
- [Kalshi fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf) and [Kalshi APY](https://help.kalshi.com/navigating-the-exchange/your-portfolio/apy-on-kalshi)
- [Deleep et al. (2026), summary](https://quantpedia.com/how-wise-is-the-crowd-in-prediction-markets/)

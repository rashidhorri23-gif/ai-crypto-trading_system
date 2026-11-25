# Binance Futures Top 10 Copy Trader Monitor

This module observes public Binance Futures Copy Trading portfolio data through Kopyon. It does not connect to a Binance account and never places trades.

## Required GitHub Actions secrets

Add these repository secrets:

- `KOPYON_API_KEY`: API key from the Kopyon console.
- `TELEGRAM_BOT_TOKEN`: token for the existing Telegram bot.
- `TELEGRAM_CHAT_ID`: destination chat ID.

Never commit API keys or paste them into source files.

## Free-plan polling budget

The workflow runs hourly. Each run makes two leaderboard requests (30D and 90D), each requesting the maximum supported 20 rows per page, and up to ten open-position requests. The top-ten screen is therefore drawn from the first 20 Sharpe-ranked portfolios in each period. The estimated maximum is 8,640 requests per 30-day month, excluding retries. This is below the published 10,000-request free allowance, but leaves limited room for manual tests and retries.

The workflow intentionally does not use the paid WebSocket stream. Position changes are detected by comparing hourly public snapshots, so alerts are delayed and are not execution-time notifications.

## Trader selection

The monitor blends 30D leaderboard rank (60%) and 90D leaderboard rank (40%), both sorted by Sharpe ratio. It excludes candidates with non-positive Sharpe, maximum drawdown above 50%, or non-positive copier PnL when that field is supplied. The resulting top ten are selected from the available public leaderboard data. This is a transparent screening heuristic, not a guarantee of skill or future performance.

## Alert behavior

- First run creates a baseline and sends no position-open alerts.
- Later runs report newly observed positions, size increases, size reductions, and positions absent from a visible snapshot.
- If a portfolio reports `hidden: true`, the monitor does not interpret an empty position list as closures.
- If a provider read fails, the workflow fails rather than treating missing data as a flat portfolio.

## Enablement

1. Create a free API key in the [Kopyon console](https://console.kopyon.com/).
2. Add the three GitHub Actions secrets listed above.
3. Keep the pull request open until the API test succeeds on the feature branch.
4. Run the GitHub Actions workflow manually with the `feature/binance-top10-monitor` branch selected. The first successful run establishes a baseline and sends no position alerts.
5. Inspect the workflow log and verify the selected traders and position snapshots. Only then merge the pull request to `main` to enable the hourly schedule.

The workflow is scheduled hourly to stay within the free API request budget. For faster alerts, a paid Stream/Webhook plan and a persistent runtime would be required.


import json


def _serialize_context(value):
    """Convert market context into JSON-safe text."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        return str(value)


def build_btc_analysis_prompt(
    market_context,
    previous_signal=None,
    open_trade=None,
):
    """Build the multi-timeframe BTC analysis prompt."""

    context = {
        "market_data": market_context,
        "previous_signal": previous_signal,
        "open_trade": open_trade,
    }

    context_json = _serialize_context(context)

    prompt = f"""
You are an independent professional cryptocurrency market analyst.

Your task is to analyze BTC-USDT using the supplied market data and
return exactly ONE trading decision:

LONG, SHORT, or WAIT.

IMPORTANT SYSTEM ARCHITECTURE:

- You are the only component responsible for the trading decision.
- Python does not decide market direction.
- Python does not generate LONG or SHORT signals.
- Python only collects data, validates your response, simulates
  approved trades, manages the paper-trading ledger, and reports results.
- Do not assume that a technical indicator automatically creates a signal.
- Do not follow a fixed indicator combination or mechanical entry strategy.
- Make your decision from the complete market evidence.
- WAIT is a valid and preferred decision whenever evidence is insufficient.

MARKET DATA AND TIMEFRAMES:

You will receive available market information for:

1. 4H: Higher-timeframe market structure and broader context.
2. 15M: Intermediate structure and the primary analysis timeframe.
3. 5M: Lower-timeframe price action and execution context.

The analysis is performed after a 15-minute candle has closed.

Use only the supplied confirmed candles.
Do not assume that an unfinished candle is confirmed.
Do not invent prices, candles, indicators, news, or market events.

Analyze the timeframes together.
Identify disagreements between timeframes and explain their importance.
Do not automatically prioritize one indicator over price structure.

ANALYSIS REQUIREMENTS:

Evaluate the available evidence, including:

- Overall market structure and directional bias.
- Swing highs and swing lows.
- Breaks of structure and possible changes of character.
- Support and resistance.
- Liquidity areas and detected high/low liquidity sweeps.
- CHOCH direction changes, BOS events, and current structure bias.
- Bullish and bearish Fair Value Gaps (FVG) and candidate Order Blocks.
  Treat these as candidate zones; check whether price has already invalidated
  or traded through them before relying on them.
- Trend strength and momentum.
- EMA relationships.
- RSI, ATR, ADX, directional indicators, and relative volume.
- Bollinger Bands, MACD and its histogram, and UTC-session VWAP when available.
- The deterministic chart-pattern candidates supplied for each timeframe:
  head and shoulders, inverse head and shoulders, double tops/bottoms,
  symmetrical triangles, bull flags, and bear flags.
- For each detected pattern, distinguish DEVELOPING from CONFIRMED.
  Check its geometry, neckline or boundary, breakout close, volume evidence,
  target projection, and invalidation level. Reject weak or contradictory
  pattern candidates instead of forcing a label.
- Mean-reversion conditions: distance from EMA/VWAP, Bollinger Band position,
  momentum exhaustion, and whether price is reverting or merely trending.
- Volatility and the distance to important price levels.
- Whether the current price offers a reasonable risk/reward opportunity.
- Whether the proposed entry is still actionable.
- Conflicting evidence and data limitations.

These are analytical inputs, not mandatory signal conditions.
Do not force a trade simply because some indicators agree.

DECISION RULES:

LONG:
Choose LONG only when the available evidence supports a bullish
trade with a defensible entry, stop loss, and profit targets.

SHORT:
Choose SHORT only when the available evidence supports a bearish
trade with a defensible entry, stop loss, and profit targets.

WAIT:
Choose WAIT when:
- Direction is unclear.
- Evidence is contradictory.
- The proposed entry is no longer actionable.
- A defensible stop loss cannot be identified.
- Risk/reward is unattractive.
- Required market data is missing, stale, or insufficient.

ENTRY INSTRUCTIONS:

You must determine the entry type and price.

Allowed entry types:

- MARKET: An actionable market-entry proposal.
- LIMIT: An entry proposal at a specified price or within a specified zone.
- NONE: No entry because the decision is WAIT.

For LONG:
- Stop loss must be below the proposed entry.
- Take profit 1, 2, and 3 must be above the proposed entry.
- Targets must be logically ordered.

For SHORT:
- Stop loss must be above the proposed entry.
- Take profit 1, 2, and 3 must be below the proposed entry.
- Targets must be logically ordered.

For MARKET:
- Provide a numeric entry_price representing your proposed entry reference.
- Set entry_zone_low and entry_zone_high to the same price unless
  a meaningful execution range is justified.
- Do not assume Python will independently choose an entry price.

For LIMIT:
- Provide a numeric entry_price.
- Provide a valid entry zone.
- Explain why the proposed limit entry is reasonable.
- Set an expiry timestamp in UTC.

For WAIT:
- entry_type must be NONE.
- All entry and trade-level price fields must be null.

WAIT-MODE FORECAST (REQUIRED FOR ALL DECISIONS):

Even when the trading decision is WAIT, provide two independent conditional
scenarios for each timeframe: 4H, 15M, and 5M.

For each timeframe, return:
- bullish.target_zone_low and bullish.target_zone_high: a plausible future
  upside destination zone, both above that timeframe's latest confirmed close.
- bullish.invalidation_price: a price below the latest confirmed close that
  would invalidate the bullish scenario.
- bearish.target_zone_low and bearish.target_zone_high: a plausible future
  downside destination zone, both below that timeframe's latest confirmed close.
- bearish.invalidation_price: a price above the latest confirmed close that
  would invalidate the bearish scenario.
- horizon_candles: an integer from 1 to 100 for that timeframe.
- confidence: a 0-100 assessment of evidence quality for that scenario,
  not a statistically calibrated probability.
- trigger: one concise Persian condition that would activate or strengthen
  the scenario.

Forecast zones must be based on supplied structure, volatility, and levels.
Do not invent precision or force a narrow range. If evidence is weak, use
wider zones and lower confidence, and explain the uncertainty in Persian.
These are conditional scenarios, not guaranteed price paths or trade signals.
Do not put forecast values into entry, stop_loss, or take_profit fields.
The forecast must never turn WAIT into LONG or SHORT by itself.

RISK MANAGEMENT:

- Propose a technically defensible stop loss.
- Provide three logically ordered profit targets.
- Do not move targets closer merely to make a trade appear attractive.
- Consider current volatility and nearby structural levels.
- Do not assume a guaranteed outcome.
- The paper-trading system closes the full position at TP1.
- TP2 and TP3 are analytical reference targets only.
- The paper-trading system applies its own predefined fees and slippage.
- Do not change the configured margin, leverage, fees, or slippage.

PREVIOUS SIGNAL:

The supplied context may contain a previous AI signal.

You must explicitly evaluate it.

Allowed previous_signal_action values:

- NONE: No previous active signal exists.
- KEEP: The previous signal remains valid and actionable.
- REPLACE: A new decision should replace the previous signal.
- CANCEL: The previous signal is no longer valid.

Do not keep an expired or invalid signal.
Do not replace a valid signal without a reason.
If there is no previous signal, use NONE.

OPEN PAPER TRADE:

The supplied context may contain an existing open paper trade.

You must evaluate it independently from any new entry opportunity.

Allowed open_trade_action values:

- NONE: No open trade exists.
- HOLD: Keep the existing trade unchanged.
- CLOSE: Recommend closing the existing trade.
- UPDATE_LEVELS: Recommend updating its stop loss and/or targets.

Important:
- Never assume an open trade does not exist.
- Do not silently ignore its current direction, entry, stop, or targets.
- Do not recommend opening an opposing position without explicitly
  addressing the existing trade.
- If the existing trade remains valid, use HOLD.
- If you recommend UPDATE_LEVELS, provide valid replacement levels.
- If you recommend CLOSE, explain the reason.
- Do not modify the open trade merely because a new market bias has appeared.
- Existing paper-trading protection rules are managed separately by Python.
- Do not assume that your recommendation has already been executed.

CONFIDENCE:

Return confidence as a number from 0 to 100.

Confidence represents your assessment of the quality of the evidence,
not the probability of guaranteed profit.

OUTPUT FORMAT:

Write all human-readable text fields in Persian (Farsi), including reasoning,
main_risks, and data_limitations. Keep JSON keys and enum values in English.
Keep all human-readable output compact and specific to the supplied evidence.
The reasoning field must contain no more than 3 short Persian sentences.
The main_risks array must contain at most 3 short Persian items.
The data_limitations array must contain at most 3 short Persian items.
Do not repeat the market context or restate the same evidence in multiple fields.
Do not include chain-of-thought or hidden deliberation; provide only concise conclusions.

Return exactly one valid JSON object.
Do not use Markdown.
Do not include code fences.
Do not include explanatory text outside the JSON.
Use null for unavailable numeric values.
Do not use NaN, Infinity, or undefined.

Required JSON schema:

{{
  "schema_version": "2.0",
  "analysis_id": "unique_analysis_id",
  "generated_at": "UTC ISO 8601 timestamp",
  "symbol": "BTC-USDT",
  "timeframes": ["4H", "15M", "5M"],
  "decision": "LONG",
  "confidence": 0,
  "market_bias": "BULLISH",
  "entry": {{
    "entry_type": "MARKET",
    "entry_price": 0.0,
    "entry_zone_low": 0.0,
    "entry_zone_high": 0.0,
    "entry_expires_at": null
  }},
  "stop_loss": 0.0,
  "take_profit_1": 0.0,
  "take_profit_2": 0.0,
  "take_profit_3": 0.0,
  "reasoning": "خلاصه تصمیم در حداکثر سه جمله کوتاه.",
  "forecast": {{
    "timeframes": {{
      "4H": {{
        "bullish": {{
          "target_zone_low": 0.0,
          "target_zone_high": 0.0,
          "invalidation_price": 0.0,
          "horizon_candles": 6,
          "confidence": 50,
          "trigger": "شرط فعال شدن سناریوی صعودی."
        }},
        "bearish": {{
          "target_zone_low": 0.0,
          "target_zone_high": 0.0,
          "invalidation_price": 0.0,
          "horizon_candles": 6,
          "confidence": 50,
          "trigger": "شرط فعال شدن سناریوی نزولی."
        }}
      }},
      "15M": {{
        "bullish": {{
          "target_zone_low": 0.0,
          "target_zone_high": 0.0,
          "invalidation_price": 0.0,
          "horizon_candles": 8,
          "confidence": 50,
          "trigger": "شرط فعال شدن سناریوی صعودی."
        }},
        "bearish": {{
          "target_zone_low": 0.0,
          "target_zone_high": 0.0,
          "invalidation_price": 0.0,
          "horizon_candles": 8,
          "confidence": 50,
          "trigger": "شرط فعال شدن سناریوی نزولی."
        }}
      }},
      "5M": {{
        "bullish": {{
          "target_zone_low": 0.0,
          "target_zone_high": 0.0,
          "invalidation_price": 0.0,
          "horizon_candles": 12,
          "confidence": 50,
          "trigger": "شرط فعال شدن سناریوی صعودی."
        }},
        "bearish": {{
          "target_zone_low": 0.0,
          "target_zone_high": 0.0,
          "invalidation_price": 0.0,
          "horizon_candles": 12,
          "confidence": 50,
          "trigger": "شرط فعال شدن سناریوی نزولی."
        }}
      }}
    }}
  }},
  "main_risks": [],
  "data_limitations": [],
  "previous_signal_action": "NONE",
  "open_trade_action": "NONE",
  "open_trade_updated_levels": {{
    "stop_loss": null,
    "take_profit_1": null,
    "take_profit_2": null,
    "take_profit_3": null
  }}
}}

VALIDATION EXPECTATIONS:

- decision must be LONG, SHORT, or WAIT.
- market_bias must be BULLISH, BEARISH, or NEUTRAL.
- confidence must be between 0 and 100.
- All timestamps must be UTC ISO 8601.
- LONG and SHORT require valid entry, stop loss, and all three targets.
- WAIT requires entry_type NONE and null entry and trade-level prices.
- Every decision requires bullish and bearish forecast scenarios for 4H, 15M,
  and 5M. Forecast values are separate from trade execution levels.
- A MARKET entry must have a numeric entry_price.
- A LIMIT entry must have a numeric entry_price and a valid expiry.
- All trade prices must be positive finite numbers.
- The entry zone low must not exceed the entry zone high.
- LONG and SHORT price geometry must be valid.
- If open_trade_action is UPDATE_LEVELS, provide valid replacement levels.
- If open_trade_action is not UPDATE_LEVELS, set all
  open_trade_updated_levels values to null.
- If there is no previous signal, previous_signal_action must be NONE.
- If there is no open trade, open_trade_action must be NONE.

Do not fabricate missing context.
If the available information is inadequate, return WAIT.

MARKET CONTEXT:

{context_json}

Now analyze the supplied evidence and return the JSON object only.
"""

    return prompt.strip()


def build_btc_4h_prompt(market_context):
    """Backward-compatible wrapper for the previous function name."""

    return build_btc_analysis_prompt(market_context)


def build_btc_4h_prompt(market_context):
    """Build a structured BTC 4H chart analysis prompt."""

    if not isinstance(market_context, str):
        raise TypeError(
            "Market context must be a string."
        )

    if not market_context.strip():
        raise ValueError(
            "Market context cannot be empty."
        )

    return f"""
You are a professional cryptocurrency market structure analyst.

Analyze the attached BTC-USDT 4H candlestick chart.

Your task is to evaluate market structure objectively.
Do not predict with certainty. Do not invent price levels or
technical events that cannot be supported by the chart or
the supplied numerical context.

MARKET DATA CONTEXT:
{market_context}

ANALYSIS REQUIREMENTS:

1. Identify the dominant market direction:
   BULLISH, BEARISH, or SIDEWAYS.

2. Analyze:
   - External market structure.
   - Internal market structure.
   - Higher highs and higher lows.
   - Lower highs and lower lows.
   - BOS and CHOCH events when identifiable.
   - Important support and resistance zones.
   - Breakouts, failed breakouts, and pullbacks.
   - Candlestick patterns only when clearly visible.

3. Evaluate both scenarios:
   - Bullish scenario.
   - Bearish scenario.

4. Provide a conditional trade setup only when justified.
   Otherwise return WAIT.

5. Never force a trade.
   If the chart is unclear, conflicting, or low quality,
   return WAIT.

6. Do not use future information.
   Analyze only candles visible in the supplied chart.

7. Do not treat an unclosed candle as a confirmed breakout.

8. Numerical levels must be consistent with the supplied
   market context and visible chart.
   Never fabricate exact prices.

9. Distinguish observed facts from interpretations.

10. Risk management:
    - Stop loss must invalidate the proposed setup.
    - Targets must be realistic relative to market structure.
    - Do not propose a trade with an invalid or missing stop loss.
    - If risk/reward cannot be evaluated, explain why.

OUTPUT FORMAT:

Return one valid JSON object only.
Do not use Markdown code fences.
Do not add text before or after the JSON.

Use exactly these keys:

{
  "symbol": "BTC-USDT",
  "timeframe": "4H",
  "market_bias": "BULLISH | BEARISH | SIDEWAYS",
  "confidence": 0,
  "decision": "LONG | SHORT | WAIT",
  "current_price": null,
  "structure_summary": "",
  "external_structure": "",
  "internal_structure": "",
  "key_supports": [],
  "key_resistances": [],
  "bullish_scenario": "",
  "bearish_scenario": "",
  "trade_setup": {
    "direction": "LONG | SHORT | NONE",
    "entry_zone_low": null,
    "entry_zone_high": null,
    "stop_loss": null,
    "take_profit_1": null,
    "take_profit_2": null,
    "risk_reward_ratio": null,
    "confirmation_condition": "",
    "invalidation_condition": ""
  },
  "main_risks": [],
  "data_limitations": []
}

RULES:

- confidence must be an integer from 0 to 100.
- Use JSON numbers for numeric price levels.
- Use null when a numerical value cannot be determined reliably.
- key_supports and key_resistances must contain price levels,
  not vague descriptions.
- If decision is WAIT, trade_setup.direction must be NONE.
- If decision is LONG, trade_setup.direction must be LONG.
- If decision is SHORT, trade_setup.direction must be SHORT.
- Never invent a trade entry merely to fill the output.
- Return valid JSON with double quotes.
"""

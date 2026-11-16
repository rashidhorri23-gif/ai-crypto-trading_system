
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

    prompt = """
You are a professional cryptocurrency market structure analyst.

Analyze the attached BTC-USDT 4H candlestick chart.

Do not predict with certainty. Do not invent price levels
or technical events unsupported by the chart or market data.

MARKET DATA CONTEXT:
"""

    prompt += market_context.strip()

    prompt += """

ANALYSIS REQUIREMENTS:

1. Identify the dominant direction:
   BULLISH, BEARISH, or SIDEWAYS.

2. Analyze external and internal market structure.
3. Identify HH, HL, LH, LL, BOS and CHOCH when visible.
4. Identify important support and resistance levels.
5. Evaluate breakouts, failed breakouts and pullbacks.
6. Identify candlestick patterns only when clearly visible.
7. Evaluate both bullish and bearish scenarios.
8. Recommend LONG, SHORT or WAIT.
9. Never force a trade.
10. Do not treat an unclosed candle as confirmed.
11. Do not use future information.
12. Do not fabricate exact prices.
13. Distinguish observations from interpretations.
14. A proposed stop loss must invalidate the trade setup.
15. If the chart is unclear, return WAIT.

OUTPUT REQUIREMENTS:

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

VALIDATION RULES:

- confidence must be an integer from 0 to 100.
- Numeric price levels must be JSON numbers.
- Use null when a value cannot be determined reliably.
- Support and resistance arrays must contain numeric prices.
- WAIT requires trade_setup.direction = NONE.
- LONG requires trade_setup.direction = LONG.
- SHORT requires trade_setup.direction = SHORT.
- Never invent an entry just to fill the output.
- Return valid JSON with double quotes.
"""

    return prompt

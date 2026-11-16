
def build_btc_4h_prompt(market_context):
    """Build a structured BTC 4H chart analysis prompt."""

    if not isinstance(market_context, str):
        raise TypeError("Market context must be a string.")

    if not market_context.strip():
        raise ValueError("Market context cannot be empty.")

    prompt = """
You are a professional cryptocurrency market structure analyst.

Analyze the attached BTC-USDT 4H candlestick chart using the
provided market data context.

Do not predict with certainty.
Do not invent price levels or technical events.
Do not force trades.
Do not treat an unclosed candle as confirmed.

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

7. Evaluate both bullish and bearish scenarios independently.

8. Prepare a complete conditional LONG plan and a complete
   conditional SHORT plan, even when the final decision is WAIT.

9. Each scenario must include:
   - Breakout entry zone.
   - Retest entry zone.
   - Stop loss.
   - Take profit 1.
   - Take profit 2.
   - Take profit 3.
   - Risk/reward ratio for each target.
   - Activation condition.
   - Invalidation condition.

10. Breakout entries must require a confirmed 4H candle close
    beyond the relevant support or resistance.

11. Retest entries must require price to revisit the broken
    level and show confirmation in the intended direction.

12. Do not consider a simple wick beyond a level as a confirmed
    breakout.

13. Entry zones must be based on visible price structure.
    Do not create arbitrary percentage-based entry zones.

14. Stop loss must be placed beyond a technically meaningful
    invalidation level.

15. Take profit levels must be based on visible support,
    resistance, liquidity areas, or other defensible chart levels.

16. Target prices must be ordered correctly:
    LONG: TP1 < TP2 < TP3.
    SHORT: TP1 > TP2 > TP3.

17. Risk/reward ratios must be calculated from the proposed
    entry and stop loss. Return null if they cannot be calculated.

18. If a reliable entry, stop loss, or target cannot be identified,
    use null for that value. Explain the limitation.

19. Keep both scenario plans CONDITIONAL unless their activation
    conditions have already been confirmed.

20. The final decision must reflect current market conditions:
    LONG, SHORT, or WAIT.

21. WAIT means there is no currently confirmed trade.
    It does not mean bullish_plan and bearish_plan should be empty.

22. Never use future information.

23. Distinguish chart observations from interpretations.

24. If the chart is unclear or data is insufficient, reduce
    confidence and return WAIT.

FINAL DECISION RULES:

- LONG only when a bullish setup is currently confirmed.
- SHORT only when a bearish setup is currently confirmed.
- Otherwise return WAIT.
- Never activate both directions simultaneously.
- When decision is WAIT, trade_setup.direction must be NONE.
- Conditional scenario plans must remain available during WAIT.
- When a trade is active, trade_setup must contain the selected
  active setup.

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
  "bullish_plan": {
    "status": "CONDITIONAL",
    "breakout_entry": {
      "entry_zone_low": null,
      "entry_zone_high": null,
      "confirmation_condition": ""
    },
    "retest_entry": {
      "entry_zone_low": null,
      "entry_zone_high": null,
      "confirmation_condition": ""
    },
    "stop_loss": null,
    "take_profit_1": null,
    "take_profit_2": null,
    "take_profit_3": null,
    "risk_reward_to_tp1": null,
    "risk_reward_to_tp2": null,
    "risk_reward_to_tp3": null,
    "activation_condition": "",
    "invalidation_condition": ""
  },
  "bearish_plan": {
    "status": "CONDITIONAL",
    "breakout_entry": {
      "entry_zone_low": null,
      "entry_zone_high": null,
      "confirmation_condition": ""
    },
    "retest_entry": {
      "entry_zone_low": null,
      "entry_zone_high": null,
      "confirmation_condition": ""
    },
    "stop_loss": null,
    "take_profit_1": null,
    "take_profit_2": null,
    "take_profit_3": null,
    "risk_reward_to_tp1": null,
    "risk_reward_to_tp2": null,
    "risk_reward_to_tp3": null,
    "activation_condition": "",
    "invalidation_condition": ""
  },
  "trade_setup": {
    "direction": "LONG | SHORT | NONE",
    "entry_zone_low": null,
    "entry_zone_high": null,
    "stop_loss": null,
    "take_profit_1": null,
    "take_profit_2": null,
    "take_profit_3": null,
    "risk_reward_ratio": null,
    "confirmation_condition": "",
    "invalidation_condition": ""
  },
  "main_risks": [],
  "data_limitations": []
}

VALIDATION RULES:

- confidence must be an integer from 0 to 100.
- Numeric price levels must be JSON numbers or null.
- Use null when a value cannot be determined reliably.
- Support and resistance arrays must contain numeric prices.
- All three take profit levels must be logically ordered.
- Risk/reward values must be numeric or null.
- Do not fabricate prices to fill missing values.
- Both bullish_plan and bearish_plan must always be present.
- Both scenario plan statuses must be CONDITIONAL unless the
  corresponding scenario is currently confirmed.
- WAIT requires trade_setup.direction = NONE.
- LONG requires trade_setup.direction = LONG.
- SHORT requires trade_setup.direction = SHORT.
- For LONG, trade_setup must reflect the active bullish plan.
- For SHORT, trade_setup must reflect the active bearish plan.
- Return valid JSON with double quotes.
"""

    return prompt

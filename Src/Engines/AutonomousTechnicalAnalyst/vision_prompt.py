
import json


def build_btc_4h_prompt(market_context):
    """
    Build the BTC 4H vision analysis prompt.
    """

    if isinstance(market_context, str):
        try:
            context = json.loads(market_context)
        except json.JSONDecodeError:
            context = {"raw_context": market_context}
    elif isinstance(market_context, dict):
        context = market_context
    else:
        context = {"raw_context": str(market_context)}

    schema = {
        "symbol": "BTC-USDT",
        "timeframe": "4H",
        "market_bias": "BULLISH | BEARISH | SIDEWAYS",
        "confidence": 0,
        "decision": "LONG | SHORT | WAIT",
        "current_price": None,
        "structure_summary": "",
        "external_structure": "",
        "internal_structure": "",
        "key_supports": [],
        "key_resistances": [],
        "bullish_scenario": "",
        "bearish_scenario": "",
        "bullish_plan": {
            "setup_type": "BREAKOUT | RETEST | NONE",
            "entry_zone_low": None,
            "entry_zone_high": None,
            "stop_loss": None,
            "take_profit_1": None,
            "take_profit_2": None,
            "take_profit_3": None,
            "risk_reward_tp1": None,
            "risk_reward_tp2": None,
            "risk_reward_tp3": None,
            "activation_condition": "",
            "invalidation_condition": ""
        },
        "bearish_plan": {
            "setup_type": "BREAKDOWN | RETEST | NONE",
            "entry_zone_low": None,
            "entry_zone_high": None,
            "stop_loss": None,
            "take_profit_1": None,
            "take_profit_2": None,
            "take_profit_3": None,
            "risk_reward_tp1": None,
            "risk_reward_tp2": None,
            "risk_reward_tp3": None,
            "activation_condition": "",
            "invalidation_condition": ""
        },
        "trade_setup": {
            "direction": "LONG | SHORT | NONE",
            "entry_zone_low": None,
            "entry_zone_high": None,
            "stop_loss": None,
            "take_profit_1": None,
            "take_profit_2": None,
            "take_profit_3": None,
            "risk_reward_ratio": None,
            "confirmation_condition": "",
            "invalidation_condition": ""
        },
        "main_risks": [],
        "data_limitations": []
    }

    prompt = f"""
You are a professional BTC cryptocurrency market analyst
specializing in Smart Money Concepts, market structure,
price action and risk management.

Analyze the supplied BTC-USDT 4H candlestick chart
and numerical market context.

MARKET CONTEXT:
{json.dumps(context, indent=2, ensure_ascii=False)}

ANALYSIS INSTRUCTIONS:

1. Analyze external and internal market structure separately.
2. Identify confirmed BOS and CHOCH only.
3. Identify meaningful support and resistance zones.
4. Evaluate bullish and bearish scenarios independently.
5. Do not allow the current bearish or bullish bias to suppress
   the opposite conditional scenario.
6. Evaluate breakout and retest entries.
7. Provide entry zones, stop loss and three targets when
   technically justified.
8. Calculate risk/reward ratios using the proposed entry
   and stop loss.
9. Use null for prices that cannot be reliably established.
10. Never invent prices or claim that an unfinished candle
    is confirmed.
11. Oversold RSI alone is not a bullish reversal signal.
12. EMA alignment alone is not a confirmed trade entry.

BULLISH PLAN:
Evaluate a possible bullish breakout or retest.
Specify the activation and invalidation conditions.
If no valid setup exists, use setup_type NONE.

BEARISH PLAN:
Evaluate a possible bearish breakdown or retest.
Specify the activation and invalidation conditions.
If no valid setup exists, use setup_type NONE.

FINAL DECISION:
LONG means a sufficiently confirmed bullish trade.
SHORT means a sufficiently confirmed bearish trade.
WAIT means neither setup is sufficiently confirmed.

If decision is WAIT, trade_setup.direction MUST be NONE.

The bullish_plan and bearish_plan are conditional scenarios.
They are not active trades unless their activation conditions
are satisfied.

OUTPUT REQUIREMENTS:
- Return one complete JSON object only.
- No Markdown or code fences.
- Include all keys in the schema.
- Use BULLISH, BEARISH or SIDEWAYS for market_bias.
- Use LONG, SHORT or WAIT for decision.
- Use LONG, SHORT or NONE for trade_setup.direction.
- Confidence must be between 0 and 100.
- Numeric prices must be numbers or null.
- Keep descriptions concise.
- Do not omit closing braces or quotation marks.
- Do not add text outside the JSON object.

REQUIRED JSON SCHEMA:
{json.dumps(schema, indent=2, ensure_ascii=False)}
"""

    return prompt.strip()

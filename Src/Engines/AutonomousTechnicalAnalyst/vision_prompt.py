
import json


def build_btc_4h_prompt(market_context):
    """
    Build a structured vision-analysis prompt for BTC USDT 4H.
    """

    if isinstance(market_context, dict):
        context_text = json.dumps(
            market_context,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    else:
        context_text = str(market_context)

    output_schema = {
        "symbol": "BTC-USDT",
        "timeframe": "4H",
        "market_bias": "BULLISH | BEARISH | NEUTRAL",
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
You are a professional cryptocurrency market structure analyst
specialized in Smart Money Concepts (SMC) and price action.

Your task is to analyze the supplied BTC-USDT 4H candlestick chart
together with the market context below.

MARKET CONTEXT:
{context_text}

ANALYSIS REQUIREMENTS:

1. MARKET STRUCTURE
- Identify the overall bullish, bearish, or neutral bias.
- Analyze external and internal market structure separately.
- Identify higher highs, higher lows, lower highs and lower lows.
- Identify confirmed BOS and CHOCH only.
- Do not invent structure breaks.

2. SUPPORT AND RESISTANCE
- Identify important visible support and resistance zones.
- Use actual chart prices and supplied market context.
- Do not create false precision.

3. BULLISH SCENARIO
- Independently evaluate a possible bullish setup.
- Consider breakout and retest opportunities.
- Define conditional entry, stop loss and three targets.
- Specify activation and invalidation conditions.
- If no valid bullish setup exists, use NONE and null prices.

4. BEARISH SCENARIO
- Independently evaluate a possible bearish setup.
- Consider breakdown and retest opportunities.
- Define conditional entry, stop loss and three targets.
- Specify activation and invalidation conditions.
- If no valid bearish setup exists, use NONE and null prices.

5. TRADE DECISION
- Choose LONG only if a bullish setup is sufficiently confirmed.
- Choose SHORT only if a bearish setup is sufficiently confirmed.
- Otherwise choose WAIT.
- Never force a trade.
- WAIT requires trade_setup.direction = NONE.
- Do not confuse a conditional scenario with an active trade.

6. RISK MANAGEMENT
- Entry must be a realistic price zone.
- Stop loss must invalidate the setup.
- Targets must follow visible market structure.
- Calculate risk/reward ratios from the proposed entry and stop.
- Use null when a value cannot be reliably determined.

7. ACCURACY
- Never invent chart values.
- Do not assume an unfinished candle is confirmed.
- Distinguish observed facts from possible scenarios.
- Do not treat oversold RSI alone as a bullish reversal.
- Do not treat an EMA crossover alone as confirmation.

OUTPUT RULES:
- Return exactly one valid JSON object.
- Do not use Markdown or code fences.
- Include every key in the required schema.
- Use numeric values for prices and confidence.
- Confidence must be between 0 and 100.
- Use null for unavailable numeric values.
- Keep descriptions concise to avoid response truncation.
- Do not include additional text outside the JSON object.

REQUIRED JSON STRUCTURE:
{json.dumps(output_schema, ensure_ascii=False, indent=2)}
"""

    return prompt.strip()

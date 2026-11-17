
import json


def build_btc_4h_prompt(market_context):
    """
    Build the BTC 4H market analysis prompt.
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
You are a professional BTC-USDT market analyst specializing in:

- Candlestick price action
- Smart Money Concepts
- External and internal market structure
- Liquidity analysis
- BOS and CHOCH
- Support and resistance
- Breakout and retest trading
- Conditional trade planning
- Risk management

Your task is to analyze the supplied BTC-USDT 4H candlestick chart
and numerical market context.

The objective is NOT simply to decide whether to enter a trade now.

Your main objective is to understand the current market and map
the two most reasonable possible future price paths.

The chart is the primary visual source.
Numerical indicators are supporting evidence.

MARKET CONTEXT:
{json.dumps(context, indent=2, ensure_ascii=False)}

==================================================
1. VISUAL CHART ANALYSIS
==================================================

Carefully inspect the candlestick chart.

Analyze:

- Candle bodies and wicks
- Swing highs and swing lows
- Higher highs and higher lows
- Lower highs and lower lows
- Strong bullish and bearish candles
- Rejection candles
- Consolidation ranges
- Breakouts and failed breakouts
- Retests
- Liquidity sweeps
- Potential liquidity pools
- Support and resistance zones
- Price reaction around important levels

Do not produce an indicator-only report.

Separate confirmed price action from possible future movements.

==================================================
2. CURRENT MARKET STRUCTURE
==================================================

Determine:

- External market structure
- Internal market structure
- Current market bias
- Confirmed BOS
- Confirmed CHOCH
- Important support levels
- Important resistance levels
- Liquidity above current price
- Liquidity below current price
- Whether the market is trending, ranging, or transitioning

Explain why the current bias is bullish, bearish, or sideways.

Do not confuse market direction with trade entry confirmation.

==================================================
3. BULLISH SCENARIO - ALWAYS REQUIRED
==================================================

Independently analyze the most reasonable bullish path.

This analysis is mandatory even if the current market is bearish.

Identify:

- What could cause price to move upward?
- Which resistance or liquidity level matters?
- What must happen before a LONG setup becomes valid?
- Is a breakout, retest, or another structure confirmation required?
- What is the conditional entry zone?
- Where should the stop loss be placed?
- Where are TP1, TP2, and TP3?
- What would invalidate this bullish scenario?

If a technically reasonable conditional setup exists,
provide numerical levels.

Do not set the bullish plan to NONE merely because
the activation condition has not happened yet.

==================================================
4. BEARISH SCENARIO - ALWAYS REQUIRED
==================================================

Independently analyze the most reasonable bearish path.

This analysis is mandatory even if the current market is bullish.

Identify:

- What could cause price to move downward?
- Which support or liquidity level matters?
- What must happen before a SHORT setup becomes valid?
- Is a breakdown, retest, or another structure confirmation required?
- What is the conditional entry zone?
- Where should the stop loss be placed?
- Where are TP1, TP2, and TP3?
- What would invalidate this bearish scenario?

If a technically reasonable conditional setup exists,
provide numerical levels.

Do not set the bearish plan to NONE merely because
the activation condition has not happened yet.

==================================================
5. CONDITIONAL TRADE PLANS
==================================================

The bullish_plan and bearish_plan describe possible future trades.

They are independent of the current active trade decision.

A conditional plan can exist while the current decision is WAIT.

For each valid plan:

- Specify the setup type.
- Specify entry_zone_low.
- Specify entry_zone_high.
- Specify stop_loss.
- Specify take_profit_1.
- Specify take_profit_2.
- Specify take_profit_3.
- Calculate risk/reward ratios.
- Specify a precise activation condition.
- Specify a precise invalidation condition.

Use:

BREAKOUT for a confirmed breakout setup.
BREAKDOWN for a confirmed bearish breakdown setup.
RETEST for a setup requiring a retest.

Use NONE only when no technically defensible conditional
setup can be established.

Never invent numerical prices.

If a level cannot be established reliably, use null
and explain the limitation.

==================================================
6. ACTIVE TRADE DECISION
==================================================

The trade_setup describes only a trade that is actionable now.

LONG:
A bullish setup is confirmed and its entry conditions
are currently satisfied.

SHORT:
A bearish setup is confirmed and its entry conditions
are currently satisfied.

WAIT:
Neither setup is currently ready for immediate entry.

IMPORTANT:

WAIT does not mean that bullish_plan and bearish_plan
should be empty.

WAIT does not mean that no future opportunity exists.

A conditional LONG or SHORT plan must remain available
even when the active decision is WAIT.

Never issue LONG or SHORT simply to avoid WAIT.

If decision is WAIT, trade_setup.direction MUST be NONE.

If decision is LONG, trade_setup.direction MUST be LONG.

If decision is SHORT, trade_setup.direction MUST be SHORT.

==================================================
7. FOUR-HOUR ANALYSIS UPDATE
==================================================

Every analysis is a new point-in-time update.

Use the supplied historical chart and context to identify:

- Changes in external structure
- Changes in internal structure
- Newly confirmed BOS or CHOCH
- New support or resistance levels
- Whether a previous breakout or breakdown condition
  is now satisfied
- Whether a conditional setup is approaching activation
- Whether a conditional setup has become invalid
- Whether the market bias has changed

Do not claim to know the previous AI report unless it
is explicitly included in the supplied context.

Do not assume an earlier scenario is still valid.

Re-evaluate both scenarios from the current available evidence.

==================================================
8. RISK MANAGEMENT
==================================================

- Prefer confirmed 4H candle closes for structural confirmation.
- Never treat an unfinished candle as confirmed.
- Avoid entries directly into opposing support or resistance.
- Use technically meaningful stop-loss levels.
- Do not fabricate risk/reward ratios.
- Identify ambiguous conditions.
- Consider volatility and market momentum.
- Keep LONG and SHORT scenarios independent.
- Do not force a trade when the evidence is insufficient.

==================================================
9. OUTPUT REQUIREMENTS
==================================================

Return one complete JSON object only.

No Markdown.
No code fences.
No explanation outside JSON.

Include every key in the required schema.

Use:

BULLISH, BEARISH, or SIDEWAYS for market_bias.
LONG, SHORT, or WAIT for decision.
LONG, SHORT, or NONE for trade_setup.direction.

Confidence must be an integer from 0 to 100.

All numerical prices must be numbers or null.

All three targets must be included in each plan.
Use null when a target cannot be reliably estimated.

Always analyze both bullish and bearish scenarios.

Do not omit a valid conditional plan just because
its activation condition has not occurred.

The final decision and conditional plans are separate concepts.

REQUIRED JSON SCHEMA:
{json.dumps(schema, indent=2, ensure_ascii=False)}
"""

    return prompt.strip()

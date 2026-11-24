"""Telegram delivery for the validated BTC vision chart and report."""

import os
from pathlib import Path

import requests


TELEGRAM_API_BASE = "https://api.telegram.org"


def _format_price(value):
    if value is None:
        return "—"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _build_caption(analysis, market_context):
    decision = analysis.get("decision", "WAIT")
    bias = analysis.get("market_bias", "NEUTRAL")
    entry = analysis.get("entry") or {}
    timeframe_context = (market_context.get("timeframes") or {}).get("4H") or {}
    latest = timeframe_context.get("latest_confirmed_candle") or {}
    support = latest.get("support")
    resistance = latest.get("resistance")
    is_rule_based = (
        (analysis.get("forecast") or {}).get("source") == "RULE_BASED_TECHNICAL"
    )
    lines = [
        "📊 تحلیل BTC-USDT | OKX",
        f"تصمیم: {decision} | جهت بازار: {bias}",
        (
            "سناریوها: تحلیل تکنیکال قانون‌محور؛ امتیاز همگرایی، نه احتمال آماری."
            if is_rule_based
            else f"اطمینان تحلیل AI: {analysis.get('confidence', '—')}٪"
        ),
    ]
    if decision in ("LONG", "SHORT"):
        lines.extend([
            f"ورود ({entry.get('entry_type', '—')}): {_format_price(entry.get('entry_price'))}",
            f"ناحیه ورود: {_format_price(entry.get('entry_zone_low'))} تا {_format_price(entry.get('entry_zone_high'))}",
            f"حد ضرر: {_format_price(analysis.get('stop_loss'))}",
            f"هدف ۱: {_format_price(analysis.get('take_profit_1'))} | هدف ۲: {_format_price(analysis.get('take_profit_2'))} | هدف ۳: {_format_price(analysis.get('take_profit_3'))}",
        ])
    else:
        lines.append("فعلاً نقطه ورود تأییدشده‌ای وجود ندارد؛ سطوح معاملاتی عمداً نمایش داده نشده‌اند.")
    lines.extend([
        f"حمایت 4H: {_format_price(support)} | مقاومت 4H: {_format_price(resistance)}",
        "", "جمع‌بندی:", str(analysis.get("reasoning", "—")).strip(),
    ])
    risks = analysis.get("main_risks") or []
    if risks:
        lines.extend(["", "ریسک‌ها: " + "؛ ".join(str(item) for item in risks[:3])])
    limitations = analysis.get("data_limitations") or []
    if limitations:
        lines.append("محدودیت داده: " + "؛ ".join(str(item) for item in limitations[:2]))
    lines.extend(["", "گزارش تحلیلی است؛ هیچ سفارش واقعی در صرافی ثبت نشده است."])
    caption = "\n".join(lines)
    return caption if len(caption) <= 1024 else caption[:1021].rstrip() + "…"


def send_telegram_chart_report(image_path, analysis, market_context):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID as GitHub Actions secrets.")
    path = Path(image_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Telegram chart image is missing: {path}")
    url = f"{TELEGRAM_API_BASE}/bot{token}/sendPhoto"
    with path.open("rb") as image_file:
        response = requests.post(
            url,
            data={"chat_id": chat_id, "caption": _build_caption(analysis, market_context)},
            files={"photo": (path.name, image_file, "image/png")},
            timeout=45,
        )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram API rejected the report: {payload}")
    message = payload.get("result", {})
    return {"message_id": message.get("message_id"), "chat_id": str((message.get("chat") or {}).get("id", chat_id))}

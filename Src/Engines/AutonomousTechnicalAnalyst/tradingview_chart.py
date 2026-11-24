"""Single-image TradingView-style BTC analysis dashboard."""

import json
import math
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

from .pattern_recognition import detect_chart_patterns


LIGHTWEIGHT_CHARTS_VERSION = "5.3.0"
CHART_CANDLE_LIMIT = 120
CHART_HEIGHT = 650
TIMEFRAME_SECONDS = {"4H": 14400, "15M": 900, "5M": 300}
SWING_CONFIRMATION_CANDLES = 3


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _boolean(value):
    if value is None or pd.isna(value):
        return False
    return bool(value)


def _prepare_timeframe(timeframe, result):
    frame = result["data"].tail(CHART_CANDLE_LIMIT).copy()
    if frame.empty:
        raise ValueError(f"{timeframe}: no candles available for chart.")

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{timeframe}: missing chart columns: {sorted(missing)}")

    candles = []
    ema = {"ema_20": [], "ema_50": [], "ema_200": []}
    overlays = {"bb_upper": [], "bb_middle": [], "bb_lower": [], "vwap_utc": []}
    volume = []
    markers = []
    pivots_high = []
    pivots_low = []

    for row in frame.to_dict(orient="records"):
        timestamp = _number(row.get("timestamp"))
        if timestamp is None:
            raise ValueError(f"{timeframe}: invalid chart timestamp.")
        candle_time = int(timestamp // 1000)
        open_price = _number(row.get("open"))
        high_price = _number(row.get("high"))
        low_price = _number(row.get("low"))
        close_price = _number(row.get("close"))
        candle_volume = _number(row.get("volume"))
        if None in (open_price, high_price, low_price, close_price, candle_volume):
            raise ValueError(f"{timeframe}: invalid OHLCV chart values.")

        candles.append({
            "time": candle_time, "open": open_price, "high": high_price,
            "low": low_price, "close": close_price,
        })
        for field in ema:
            value = _number(row.get(field))
            if value is not None:
                ema[field].append({"time": candle_time, "value": value})
        for field in overlays:
            value = _number(row.get(field))
            if value is not None:
                overlays[field].append({"time": candle_time, "value": value})

        volume.append({
            "time": candle_time,
            "value": candle_volume,
            "color": "rgba(8,153,129,0.48)" if close_price >= open_price
            else "rgba(242,54,69,0.48)",
        })

        pivot_time = candle_time - SWING_CONFIRMATION_CANDLES * TIMEFRAME_SECONDS[timeframe]
        swing_high = _number(row.get("swing_high"))
        swing_low = _number(row.get("swing_low"))
        if swing_high is not None:
            pivots_high.append({"time": pivot_time, "price": swing_high})
            markers.append({
                "time": pivot_time, "position": "aboveBar",
                "color": "#f59e0b", "shape": "circle", "text": "SH",
            })
        if swing_low is not None:
            pivots_low.append({"time": pivot_time, "price": swing_low})
            markers.append({
                "time": pivot_time, "position": "belowBar",
                "color": "#38bdf8", "shape": "circle", "text": "SL",
            })
        if _boolean(row.get("bos_bullish")):
            markers.append({
                "time": candle_time, "position": "belowBar",
                "color": "#089981", "shape": "arrowUp", "text": "BOS+",
            })
        if _boolean(row.get("bos_bearish")):
            markers.append({
                "time": candle_time, "position": "aboveBar",
                "color": "#f23645", "shape": "arrowDown", "text": "BOS-",
            })
        if _boolean(row.get("choch_bullish")):
            markers.append({
                "time": candle_time, "position": "belowBar",
                "color": "#22cfa7", "shape": "arrowUp", "text": "CH+",
            })
        if _boolean(row.get("choch_bearish")):
            markers.append({
                "time": candle_time, "position": "aboveBar",
                "color": "#fb7185", "shape": "arrowDown", "text": "CH-",
            })
        if _boolean(row.get("liquidity_sweep_high")):
            markers.append({
                "time": candle_time, "position": "aboveBar",
                "color": "#f59e0b", "shape": "circle", "text": "SW-H",
            })
        if _boolean(row.get("liquidity_sweep_low")):
            markers.append({
                "time": candle_time, "position": "belowBar",
                "color": "#38bdf8", "shape": "circle", "text": "SW-L",
            })

    markers.sort(key=lambda item: item["time"])
    latest = frame.iloc[-1]
    close = _number(latest.get("close"))

    def latest_valid(column):
        if column not in frame.columns:
            return None
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        return _number(values.iloc[-1]) if not values.empty else None

    def nearest_levels(pivots, is_support):
        candidates = []
        seen = set()
        for pivot in reversed(pivots):
            price = pivot["price"]
            if (price < close if is_support else price > close) and round(price, 2) not in seen:
                candidates.append(price)
                seen.add(round(price, 2))
        return sorted(candidates, reverse=True)[:3] if is_support else sorted(candidates)[:3]

    return {
        "timeframe": timeframe,
        "candles": candles,
        "ema": ema,
        "overlays": overlays,
        "volume": volume,
        "markers": markers,
        "patterns": detect_chart_patterns(result["data"].tail(CHART_CANDLE_LIMIT)),
        "levels": {
            "supports": nearest_levels(pivots_low, True),
            "resistances": nearest_levels(pivots_high, False),
            "latest_support": _number(latest.get("support")),
            "latest_resistance": _number(latest.get("resistance")),
        },
        "smc_zones": {
            "bullish_fvg_low": latest_valid("fvg_bullish_low"),
            "bullish_fvg_high": latest_valid("fvg_bullish_high"),
            "bearish_fvg_low": latest_valid("fvg_bearish_low"),
            "bearish_fvg_high": latest_valid("fvg_bearish_high"),
            "bullish_ob_low": latest_valid("last_bullish_ob_low"),
            "bullish_ob_high": latest_valid("last_bullish_ob_high"),
            "bearish_ob_low": latest_valid("last_bearish_ob_low"),
            "bearish_ob_high": latest_valid("last_bearish_ob_high"),
        },
        "metrics": {
            "close": close,
            "ema_trend": str(latest.get("ema_trend", "N/A")),
            "market_structure": str(latest.get("market_structure", "N/A")),
            "rsi": _number(latest.get("rsi_14")),
            "adx": _number(latest.get("adx_14")),
            "relative_volume": _number(latest.get("relative_volume")),
            "macd_histogram": _number(latest.get("macd_histogram")),
            "vwap": _number(latest.get("vwap_utc")),
            "bb_upper": _number(latest.get("bb_upper")),
            "bb_lower": _number(latest.get("bb_lower")),
        },
    }


def _build_payload(results, analysis):
    missing = {"4H", "15M", "5M"}.difference(results)
    if missing:
        raise ValueError(f"Missing timeframe data for chart: {sorted(missing)}")
    prepared = {
        timeframe: _prepare_timeframe(timeframe, results[timeframe])
        for timeframe in ("4H", "15M", "5M")
    }
    return {
        "symbol": "BTC-USDT",
        "source": "OKX",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "analysis": analysis,
        "main": prepared["4H"],
        "confirmations": {
            "15M": prepared["15M"]["metrics"],
            "5M": prepared["5M"]["metrics"],
        },
    }


def _build_html(payload):
    payload_json = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    template = r"""<!doctype html>
<html lang="fa">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTC-USDT AI Market Dashboard</title>
<style>
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:#050b14;color:#e2e8f0;font-family:Arial,Tahoma,sans-serif}
body{width:1800px;padding:22px}
.header{display:flex;justify-content:space-between;align-items:center;gap:24px;padding:16px 22px;margin-bottom:16px;background:#0b1625;border:1px solid #26364c;border-radius:14px}
.brand{font-size:29px;font-weight:700}.sub{font-size:14px;color:#91a4bc;margin-top:7px}
.pills{display:flex;gap:10px}.pill{padding:10px 15px;background:#101f32;border:1px solid #34465e;border-radius:9px;font-size:15px}
.pill b{margin-left:6px;color:#f8fafc}
.main-grid{display:grid;grid-template-columns:1fr 470px;gap:16px;align-items:stretch}
.panel{background:#091422;border:1px solid #26364c;border-radius:13px;overflow:hidden}
.panel-head{display:flex;justify-content:space-between;align-items:center;padding:13px 17px;background:#101e30;border-bottom:1px solid #26364c}
.panel-title{font-size:20px;font-weight:700}.panel-sub{font-size:13px;color:#91a4bc;margin-top:5px}
.chart{height:690px;width:100%}
.sidebar{display:flex;flex-direction:column;gap:14px}
.side-section{padding:16px 17px;background:#0b1727;border:1px solid #26364c;border-radius:12px}
.side-title{font-size:19px;font-weight:700;margin-bottom:12px;color:#f1f5f9}
.decision{display:inline-block;padding:8px 12px;border-radius:20px;border:1px solid #64748b;font-weight:700;margin-bottom:12px}
.reason{direction:rtl;text-align:right;line-height:1.9;color:#dbeafe;font-size:15px}
.kv{display:flex;justify-content:space-between;gap:12px;padding:9px 0;border-bottom:1px solid #1e3045;font-size:15px}
.kv:last-child{border-bottom:0}.muted{color:#9aabc0}.value{font-weight:700;text-align:left}
.green{color:#22cfa7}.red{color:#fb7185}.amber{color:#fbbf24}.blue{color:#60a5fa}
.trade-note{font-size:13px;color:#9aabc0;line-height:1.7;direction:rtl;text-align:right;margin-top:10px}
.bottom-grid{display:grid;grid-template-columns:1fr 1fr 470px;gap:16px;margin-top:16px}
.scenario{padding:17px;background:#0b1727;border:1px solid #26364c;border-radius:12px;direction:rtl;text-align:right}
.scenario h3{margin:0 0 12px;font-size:20px}.scenario p{margin:8px 0;line-height:1.8;font-size:14px}
.scenario.bull h3{color:#22cfa7}.scenario.bear h3{color:#fb7185}
.confirm{padding:16px;background:#0b1727;border:1px solid #26364c;border-radius:12px}
.confirm h3{margin:0 0 10px;font-size:18px}
.confirm-row{padding:10px 0;border-bottom:1px solid #1e3045}
.confirm-row:last-child{border:0}.confirm-row strong{font-size:16px}
.footer{text-align:right;color:#71839a;font-size:12px;padding:13px 4px 0}
</style>
</head>
<body>
<div class="header">
 <div><div class="brand">₿ BTC / USDT <span style="color:#38bdf8">| 4H</span></div>
 <div class="sub">OKX confirmed market data · TradingView Lightweight Charts™ · UTC</div></div>
 <div class="pills">
  <div class="pill">تصمیم AI <b id="decision">—</b></div>
  <div class="pill">جهت بازار <b id="bias">—</b></div>
  <div class="pill">اطمینان <b id="confidence">—</b></div>
 </div>
</div>
<div class="main-grid">
 <section class="panel">
  <div class="panel-head"><div><div class="panel-title">نمودار اصلی BTC-USDT | 4H</div>
  <div class="panel-sub" id="chart-sub">کندل‌ها · EMA 20 / 50 / 200 · حجم · ساختار بازار</div></div>
  <div class="panel-sub">سبز: حمایت و سناریوی صعودی | قرمز: مقاومت و سناریوی نزولی</div></div>
  <div id="main-chart" class="chart"></div>
 </section>
 <aside class="sidebar">
  <section class="side-section">
   <div class="side-title">تحلیل هوش مصنوعی</div>
   <div class="decision" id="decision-badge">در انتظار تحلیل</div>
   <div class="reason" id="reason">خلاصه تحلیل پس از دریافت پاسخ معتبر هوش مصنوعی نمایش داده می‌شود.</div>
   <div class="kv"><span class="muted">قیمت آخرین کندل 4H</span><strong class="value" id="last-price">—</strong></div>
   <div class="kv"><span class="muted">ساختار بازار</span><strong class="value" id="structure">—</strong></div>
   <div class="kv"><span class="muted">روند EMA</span><strong class="value" id="ema-trend">—</strong></div>
   <div id="technical-metrics"></div>
  </section>
  <section class="side-section">
   <div class="side-title">برنامه معاملاتی</div>
   <div id="trade-plan"></div>
   <div class="trade-note" id="trade-note">در حالت WAIT نقطه ورود، حد ضرر و اهداف معاملاتی ساخته یا حدس زده نمی‌شوند.</div>
  </section>
  <section class="side-section">
   <div class="side-title">سطوح کلیدی 4H</div>
   <div id="levels"></div>
   <div class="trade-note">سطوح از پیوت‌های تأییدشده استخراج می‌شوند؛ در صورت نبود سطح معتبر، مقدار نمایش داده نمی‌شود.</div>
  </section>
  <section class="side-section">
   <div class="side-title">الگوهای شناسایی‌شده</div>
   <div id="chart-patterns"></div>
   <div class="trade-note">الگوها با الگوریتم هندسی روی کندل‌ها و سوئینگ‌های تأییدشده شناسایی می‌شوند؛ امتیاز ساختاری، احتمال آماری برد نیست.</div>
  </section>
 </aside>
</div>
<div class="bottom-grid">
 <section class="scenario bull"><h3>↗ سناریوی صعودی</h3><div id="bull-scenario"><p>پیش‌بینی مشروط پس از تحلیل نمایش داده می‌شود.</p></div></section>
 <section class="scenario bear"><h3>↘ سناریوی نزولی</h3><div id="bear-scenario"><p>پیش‌بینی مشروط پس از تحلیل نمایش داده می‌شود.</p></div></section>
 <section class="confirm"><h3>تأیید چندتایم‌فریمی</h3><div id="confirmations"></div></section>
</div>
<div class="footer">تحلیل و سناریوها تخمینی و مشروط هستند، نه تضمین حرکت قیمت یا دستور معامله. فقط در صورت LONG/SHORT معتبر، سطوح ورود و مدیریت معامله نمایش داده می‌شوند. نمودار بر پایه داده OKX و کتابخانه TradingView Lightweight Charts ساخته شده است.</div>
<script src="https://unpkg.com/lightweight-charts@5.3.0/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function(){
 const p=__PAYLOAD__,a=p.analysis||{},m=p.main,root=document;
 const fmt=v=>v==null||!Number.isFinite(Number(v))?"—":Number(v).toLocaleString("en-US",{maximumFractionDigits:2});
 const set=(id,value)=>{const el=root.getElementById(id);if(el)el.textContent=value==null?"—":String(value)};
 const tfName=v=>({BULLISH:"صعودی",BEARISH:"نزولی",NEUTRAL:"خنثی",LONG:"خرید",SHORT:"فروش",WAIT:"انتظار"})[v]||v||"—";
 set("decision",tfName(a.decision));set("bias",tfName(a.market_bias));
 set("confidence",a.confidence==null?"—":a.confidence+"٪");
 set("decision-badge","وضعیت: "+tfName(a.decision));
 set("reason",a.reasoning||"جمع‌بندی تحلیل در دسترس نیست.");
 set("last-price",fmt(m.metrics.close));
 set("structure",tfName(m.metrics.market_structure));
 set("ema-trend",tfName(m.metrics.ema_trend));
 const technicalMetrics=root.getElementById("technical-metrics");
 [["RSI 14",fmt(m.metrics.rsi)],["ADX 14",fmt(m.metrics.adx)],
  ["Relative Volume",fmt(m.metrics.relative_volume)],["MACD Histogram",fmt(m.metrics.macd_histogram)],
  ["UTC VWAP",fmt(m.metrics.vwap)],["Bollinger Upper",fmt(m.metrics.bb_upper)],
  ["Bollinger Lower",fmt(m.metrics.bb_lower)]].forEach(metric=>{
   const row=root.createElement("div");row.className="kv";
   const label=root.createElement("span");label.className="muted";label.textContent=metric[0];
   const value=root.createElement("strong");value.className="value";value.textContent=metric[1];
   row.append(label,value);technicalMetrics.appendChild(row);
  });
 const decision=a.decision||"WAIT";
 const badge=root.getElementById("decision-badge");
 badge.style.color=decision==="LONG"?"#22cfa7":decision==="SHORT"?"#fb7185":"#fbbf24";
 badge.style.borderColor=decision==="LONG"?"#22cfa7":decision==="SHORT"?"#fb7185":"#fbbf24";
 function addLine(series,price,color,title,style){
  if(Number.isFinite(Number(price))&&Number(price)>0)series.createPriceLine({
   price:Number(price),color,lineWidth:2,lineStyle:style,axisLabelVisible:true,title
  });
 }
 const chart=LightweightCharts.createChart(root.getElementById("main-chart"),{
  width:root.getElementById("main-chart").clientWidth,height:690,
  layout:{background:{type:"solid",color:"#091422"},textColor:"#cbd5e1",fontFamily:"Arial,Tahoma,sans-serif",attributionLogo:true},
  grid:{vertLines:{color:"#1b2b40"},horzLines:{color:"#1b2b40"}},
  rightPriceScale:{borderColor:"#334155",scaleMargins:{top:.08,bottom:.2}},
  timeScale:{borderColor:"#334155",timeVisible:true,secondsVisible:false},
  localization:{timeFormatter:t=>new Date(Number(t)*1000).toISOString().slice(0,16).replace("T"," ")},
  crosshair:{mode:LightweightCharts.CrosshairMode.Normal},handleScroll:false,handleScale:false
 });
 const candles=chart.addSeries(LightweightCharts.CandlestickSeries,{
  upColor:"#089981",downColor:"#f23645",borderUpColor:"#089981",borderDownColor:"#f23645",
  wickUpColor:"#089981",wickDownColor:"#f23645",lastValueVisible:true,priceLineVisible:true
 });
 candles.setData(m.candles);
 [["ema_20","#f59e0b","EMA 20"],["ema_50","#38bdf8","EMA 50"],["ema_200","#c084fc","EMA 200"]].forEach(s=>{
  const points=m.ema[s[0]]||[];if(points.length){const line=chart.addSeries(LightweightCharts.LineSeries,{
   color:s[1],lineWidth:2,title:s[2],priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false
  });line.setData(points);}
 });
 [["bb_upper","#64748b","BB Upper"],["bb_middle","#475569","BB Mid"],["bb_lower","#64748b","BB Lower"],["vwap_utc","#facc15","UTC VWAP"]].forEach(s=>{
  const points=m.overlays[s[0]]||[];if(points.length){const line=chart.addSeries(LightweightCharts.LineSeries,{
   color:s[1],lineWidth:1,lineStyle:2,title:s[2],priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false
  });line.setData(points);}
 });
 const volume=chart.addSeries(LightweightCharts.HistogramSeries,{priceScaleId:"volume",
  priceFormat:{type:"volume"},lastValueVisible:false,priceLineVisible:false});
 volume.setData(m.volume);chart.priceScale("volume").applyOptions({scaleMargins:{top:.84,bottom:0},visible:false});
 (m.levels.supports||[]).forEach((v,i)=>addLine(candles,v,"#089981","حمایت "+(i+1),2));
 (m.levels.resistances||[]).forEach((v,i)=>addLine(candles,v,"#f59e0b","مقاومت "+(i+1),2));
 const z=m.smc_zones||{};
 [["bullish_fvg_low","#22cfa7","FVG صعودی پایین"],["bullish_fvg_high","#22cfa7","FVG صعودی بالا"],
  ["bearish_fvg_low","#fb7185","FVG نزولی پایین"],["bearish_fvg_high","#fb7185","FVG نزولی بالا"],
  ["bullish_ob_low","#38bdf8","Order Block صعودی پایین"],["bullish_ob_high","#38bdf8","Order Block صعودی بالا"],
  ["bearish_ob_low","#f59e0b","Order Block نزولی پایین"],["bearish_ob_high","#f59e0b","Order Block نزولی بالا"]]
  .forEach(item=>addLine(candles,z[item[0]],item[1],item[2],2));
 const f=a.forecast&&a.forecast.timeframes?a.forecast.timeframes["4H"]:null;
 const bull=f&&f.bullish,bear=f&&f.bearish;
 if(bull){
  addLine(candles,bull.target_zone_low,"#22c55e","هدف صعودی پایین",2);
  addLine(candles,bull.target_zone_high,"#22c55e","هدف صعودی بالا",2);
  addLine(candles,bull.invalidation_price,"#16a34a","ابطال صعود",1);
 }
 if(bear){
  addLine(candles,bear.target_zone_low,"#fb7185","هدف نزولی پایین",2);
  addLine(candles,bear.target_zone_high,"#fb7185","هدف نزولی بالا",2);
  addLine(candles,bear.invalidation_price,"#e11d48","ابطال نزول",1);
 }
 if(decision==="LONG"||decision==="SHORT"){
  const e=a.entry||{};
  addLine(candles,e.entry_price,"#60a5fa","نقطه ورود",0);
  if(e.entry_zone_low!=null&&e.entry_zone_high!=null&&Number(e.entry_zone_low)!==Number(e.entry_zone_high)){
   addLine(candles,e.entry_zone_low,"#60a5fa","کف محدوده ورود",1);
   addLine(candles,e.entry_zone_high,"#60a5fa","سقف محدوده ورود",1);
  }
  addLine(candles,a.stop_loss,"#f23645","حد ضرر",0);
  addLine(candles,a.take_profit_1,"#22c55e","هدف ۱",0);
  addLine(candles,a.take_profit_2,"#22c55e","هدف ۲",1);
  addLine(candles,a.take_profit_3,"#22c55e","هدف ۳",1);
 }
 const patternMarkers=[];
 (m.patterns||[]).forEach((pattern,index)=>{
  const color=pattern.direction==="BULLISH"?"#22cfa7":pattern.direction==="BEARISH"?"#fb7185":"#fbbf24";
  (pattern.lines||[]).forEach(boundary=>{
   const points=(boundary.points||[]).filter(point=>Number.isFinite(Number(point.time))&&Number.isFinite(Number(point.value)));
   if(points.length>=2){
    const line=chart.addSeries(LightweightCharts.LineSeries,{
     color,lineWidth:2,lineStyle:pattern.status==="CONFIRMED"?0:2,
     title:pattern.name+" "+boundary.name,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false
    });
    line.setData(points.map(point=>({time:Number(point.time),value:Number(point.value)})));
   }
  });
  (pattern.points||[]).forEach(point=>{
   if(Number.isFinite(Number(point.time))&&Number.isFinite(Number(point.price))){
    patternMarkers.push({time:Number(point.time),position:pattern.direction==="BEARISH"?"aboveBar":"belowBar",color,shape:"circle",text:String(point.label||pattern.name).slice(0,8)});
   }
  });
  if(index===0){addLine(candles,pattern.target,color,"هدف الگو",2);addLine(candles,pattern.invalidation,"#fbbf24","ابطال الگو",2);}
 });
 if(m.markers.length||patternMarkers.length){
  LightweightCharts.createSeriesMarkers(candles,m.markers.concat(patternMarkers).sort((x,y)=>x.time-y.time));
 }
 chart.timeScale().fitContent();
 const plan=root.getElementById("trade-plan");
 function kv(label,value,cls){const row=root.createElement("div");row.className="kv";
  const l=root.createElement("span");l.className="muted";l.textContent=label;
  const v=root.createElement("strong");v.className="value "+(cls||"");v.textContent=value;
  row.append(l,v);plan.appendChild(row);}
 if(decision==="LONG"||decision==="SHORT"){
  const e=a.entry||{};kv("ورود",fmt(e.entry_price),"blue");
  kv("محدوده ورود",fmt(e.entry_zone_low)+" تا "+fmt(e.entry_zone_high),"blue");
  kv("حد ضرر",fmt(a.stop_loss),"red");kv("هدف ۱",fmt(a.take_profit_1),"green");
  kv("هدف ۲",fmt(a.take_profit_2),"green");kv("هدف ۳",fmt(a.take_profit_3),"green");
  set("trade-note","این سطوح پیشنهاد تحلیل AI هستند؛ ثبت سفارش واقعی انجام نمی‌شود.");
 }else{
  kv("وضعیت","بدون ورود تأییدشده","amber");
  set("trade-note","تا زمان تأیید شرایط ورود، نقطه ورود و حد ضرر معاملاتی تعریف نمی‌شود. خطوط سناریو، پیش‌بینی مشروط‌اند.");
 }
 const patternPanel=root.getElementById("chart-patterns");
 (m.patterns||[]).forEach(pattern=>{
  const item=root.createElement("div");item.className="kv";
  const label=root.createElement("span");label.className="muted";
  label.textContent=pattern.name+" · "+(pattern.direction==="BULLISH"?"صعودی":pattern.direction==="BEARISH"?"نزولی":"خنثی");
  const status=root.createElement("strong");status.className="value "+(pattern.status==="CONFIRMED"?"green":"amber");
  status.textContent=pattern.status==="CONFIRMED"?"تأییدشده":"در حال شکل‌گیری";
  item.append(label,status);patternPanel.appendChild(item);
  const detail=root.createElement("div");detail.className="trade-note";
  detail.textContent="هدف هندسی: "+fmt(pattern.target)+" · ابطال: "+fmt(pattern.invalidation)+" · شرط: "+(pattern.trigger||"—");
  patternPanel.appendChild(detail);
 });
 if(!(m.patterns||[]).length)patternPanel.textContent="در داده فعلی الگوی معتبر شناسایی نشد.";
 const levels=root.getElementById("levels");
 function levelRow(label,value,cls){const row=root.createElement("div");row.className="kv";
  const l=root.createElement("span");l.className="muted";l.textContent=label;
  const v=root.createElement("strong");v.className="value "+(cls||"");v.textContent=fmt(value);
  row.append(l,v);levels.appendChild(row);}
 (m.levels.resistances||[]).forEach((v,i)=>levelRow("مقاومت "+(i+1),v,"amber"));
 (m.levels.supports||[]).forEach((v,i)=>levelRow("حمایت "+(i+1),v,"green"));
 if(!(m.levels.resistances||[]).length&&!(m.levels.supports||[]).length)levelRow("سطوح معتبر","در داده فعلی موجود نیست");
 function scenario(id,item,direction){
  const el=root.getElementById(id);el.innerHTML="";
  if(!item){el.textContent="سناریوی معتبر در پاسخ AI موجود نیست.";return;}
  const rows=[
   ["محدوده هدف",fmt(item.target_zone_low)+" تا "+fmt(item.target_zone_high)],
   ["قیمت ابطال",fmt(item.invalidation_price)],
   ["افق زمانی",String(item.horizon_candles)+" کندل"],
   ["اطمینان سناریو",String(item.confidence)+"٪"],
   ["شرط فعال‌سازی",item.trigger||"—"]
  ];
  rows.forEach(r=>{const p=root.createElement("p");p.textContent=r[0]+": "+r[1];el.appendChild(p);});
 }
 scenario("bull-scenario",bull,"bull");scenario("bear-scenario",bear,"bear");
 const conf=root.getElementById("confirmations");
 ["15M","5M"].forEach(tf=>{
  const d=p.confirmations[tf],box=root.createElement("div");box.className="confirm-row";
  const title=root.createElement("strong");title.textContent=tf;
  const detail=root.createElement("div");detail.className="sub";
  detail.textContent="ساختار: "+tfName(d.market_structure)+" · روند EMA: "+tfName(d.ema_trend)+" · RSI: "+fmt(d.rsi)+" · ADX: "+fmt(d.adx)+" · MACD hist: "+fmt(d.macd_histogram);
  box.append(title,detail);conf.appendChild(box);
 });
 requestAnimationFrame(()=>requestAnimationFrame(()=>{window.__CHART_READY__=true;}));
})();
</script>
</body>
</html>"""
    return template.replace("__PAYLOAD__", payload_json)


def render_tradingview_chart(results, output_path, analysis=None):
    """Render one consolidated 4H dashboard with 15M and 5M confirmation."""

    payload = _build_payload(results, analysis)
    html = _build_html(payload)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.stem + ".tmp" + output.suffix)
    page_errors = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        try:
            page = browser.new_page(
                viewport={"width": 1856, "height": 1500}, device_scale_factor=1
            )
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.set_content(html, wait_until="load", timeout=60000)
            page.wait_for_function("window.__CHART_READY__ === true", timeout=60000)
            page.evaluate("document.fonts.ready")
            page.wait_for_timeout(350)
            page.screenshot(
                path=str(temporary_output), full_page=True, animations="disabled"
            )
        except Exception as error:
            details = "; ".join(page_errors)
            suffix = f" Browser errors: {details}" if details else ""
            raise RuntimeError(
                f"TradingView chart rendering failed: {error}.{suffix}"
            ) from error
        finally:
            browser.close()

    if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
        raise RuntimeError("Chart renderer produced an empty image.")
    temporary_output.replace(output)
    return output

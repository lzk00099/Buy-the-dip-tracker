import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime
import io
import os
import tempfile
import time
from urllib.parse import quote
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# -----------------------------------------------------------------------------
# 1. 页面基本配置与全局样式
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentinel 2.0: 大盘资金底层逻辑（抄底与逃顶）双向风控系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式优化视觉体验
st.markdown("""
<style>
    .reportview-container { background: #fdfbf7; }
    .metric-box {
        padding: 10px 12px;
        border-radius: 8px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 6px;
        border-left: 5px solid #cccccc;
        font-size: 9pt;
        line-height: 1.35;
    }
    .status-bottom-active { border-left-color: #2ecc71; background-color: #f4fbf7; }
    .status-top-active { border-left-color: #e74c3c; background-color: #fdf5f5; }
    .status-neutral { border-left-color: #3498db; background-color: #f0f7fc; }
    .switch-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }
    .switch-title { font-size: 9.4pt; font-weight: 700; color: #2c3e50; line-height: 1.25; }
    .switch-value { margin: 4px 0 4px 0; font-size: 8.7pt; line-height: 1.35; }
    .switch-status { margin: 2px 0 0 0; font-size: 8.6pt; line-height: 1.35; color: #34495e; }
    .switch-status div, .switch-status p, .switch-status span { font-size: 8.6pt !important; line-height: 1.35 !important; }
    .switch-meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 8px; margin: 6px 0; padding: 6px; background: #f8fafb; border-radius: 5px; color: #34495e; }
    .switch-meta-grid div { font-size: 8.1pt; line-height: 1.25; }
    .switch-strategy { margin-top: 6px; padding: 6px 7px; border-radius: 5px; background: #fffdf4; border: 1px solid #f2e7b8; color: #5d4b00; font-size: 8.3pt; line-height: 1.35; }
    .switch-footer { margin: 4px 0 10px 0; color: #7f8c8d; font-size: 8pt; line-height: 1.3; }
    .switch-boundary-panel { padding: 8px 10px; border: 1px solid #e0e0e0; border-radius: 6px; background-color: #ffffff; }
    .switch-boundary-panel p { margin: 0 0 6px 0; font-size: 8.6pt; line-height: 1.38; }
    .switch-boundary-panel p:last-child { margin-bottom: 0; }
    div[data-testid="stExpander"] { margin: 0 0 6px 0; }
    div[data-testid="stExpander"] details { border-radius: 6px; }
    div[data-testid="stExpander"] summary p { font-size: 8.8pt !important; line-height: 1.25 !important; }
    
    .badge-bottom { background-color: #2ecc71; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 9px; white-space: nowrap; }
    .badge-top { background-color: #e74c3c; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 9px; white-space: nowrap; }
    .badge-info { background-color: #3498db; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 9px; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 数据获取与处理模块 (Data Pipeline & Timestamp Injection)
# -----------------------------------------------------------------------------

def filter_leveraged_etfs(ticker_list):
    """
    内置杠杆ETF及反向ETF特殊过滤逻辑，
    清洗诊断列队中的高损耗及反向衍生品，确保底层指标纯净度。
    """
    known_lev_etfs = {'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'SOXL', 'SOXS', 'FAS', 'FAZ', 'YINN', 'YANG', 'UVXY', 'VIXY', 'SPXL'}
    return [ticker for ticker in ticker_list if str(ticker).upper() not in known_lev_etfs]

# Yahoo 对共享云 IP 的限流会影响同一次 Streamlit 重跑中的所有 yf 调用。
# 将仍需 Yahoo 的品种收口到一个串行批次；波动率指数与 BTC 在下方改走官方源。
YAHOO_CORE_TICKERS = ['QQQ', 'SPY', 'IWM', 'RSP', 'HYG', 'LQD', '^MOVE', '^NDX']
YAHOO_REQUIRED_TICKERS = ['QQQ', 'SPY', 'IWM', 'RSP']
INTRADAY_ETF_TICKERS = ['QQQ', 'SPY', 'IWM', 'RSP', 'HYG', 'LQD']
# 仅一个低频 Yahoo 批次：能拿到即覆盖 CBOE/Yahoo 日线，缺失或过期则自动保留日线。
INTRADAY_VOL_TICKERS = ['^VIX', '^VIX3M', '^VXN', '^VVIX', '^MOVE']
INTRADAY_TTL_SECONDS = 600
# Massive（原 Polygon）指数 REST 快照可作为 CBOE 波动率指数的可选实时源。
# 真实指数数据受许可约束；默认代码表仅在用户账号实际有相应 entitlement 时生效。
MASSIVE_VOLATILITY_DEFAULT_TICKERS = {
    '^VIX': 'I:VIX',
    '^VIX3M': 'I:VIX3M',
    '^VXN': 'I:VXN',
    '^VVIX': 'I:VVIX',
}
MASSIVE_VOLATILITY_TTL_SECONDS = 60
US_EASTERN = ZoneInfo("America/New_York")
MARKET_CACHE_DIR = os.path.join(tempfile.gettempdir(), "sentinel2_market_cache")
YAHOO_ATTEMPTED_THIS_RUN = False
YAHOO_LAST_ERROR = ""
INTRADAY_ATTEMPTED_THIS_RUN = False
INTRADAY_SNAPSHOT_THIS_RUN = None
INTRADAY_LAST_ERROR = ""
VOLATILITY_LAST_ERROR = ""

def _market_cache_path(name):
    os.makedirs(MARKET_CACHE_DIR, exist_ok=True)
    return os.path.join(MARKET_CACHE_DIR, name)

def _read_frame_cache(name, max_age_days=10):
    path = _market_cache_path(name)
    if not os.path.exists(path):
        return pd.DataFrame()
    age_seconds = time.time() - os.path.getmtime(path)
    if age_seconds > max_age_days * 86400:
        return pd.DataFrame()
    try:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        cached.index = pd.to_datetime(cached.index, errors='coerce')
        cached = cached.loc[~cached.index.isna()].sort_index()
        cached.attrs["data_source"] = "last_success_cache"
        cached.attrs["cache_age_hours"] = round(age_seconds / 3600, 1)
        return cached
    except Exception:
        return pd.DataFrame()

def _write_frame_cache(frame, name):
    if frame is None or frame.empty:
        return
    path = _market_cache_path(name)
    temp_path = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    frame.to_csv(temp_path)
    os.replace(temp_path, path)

MODEL_SCORE_HISTORY_FILE = "model_score_history.csv"
MODEL_SCORE_HISTORY_DAYS = 180

def record_model_score_snapshot(risk_score, opportunity_score, macro_score, net_risk_score):
    """
    记录模型输出本身（不是伪造的历史回测）。同一个美东交易日只保留最新值，
    以便在不增加任何外部行情请求的前提下绘制清晰的日线级别模型趋势。
    """
    path = _market_cache_path(MODEL_SCORE_HISTORY_FILE)
    columns = [
        "timestamp", "weighted_risk", "weighted_opportunity",
        "macro_adjustment", "net_risk"
    ]
    try:
        if os.path.exists(path):
            history = pd.read_csv(path)
        else:
            history = pd.DataFrame(columns=columns)
        history = history.reindex(columns=columns)
        history["timestamp"] = pd.to_datetime(
            history["timestamp"], errors="coerce", utc=True
        )
        history = history.dropna(subset=["timestamp"])
        # 兼容早期 15 分钟采样文件：迁移为每个美东自然日一条记录。
        history["timestamp"] = (
            history["timestamp"].dt.tz_convert(US_EASTERN).dt.normalize()
            .dt.tz_convert("UTC")
        )

        timestamp = (
            pd.Timestamp.now(tz=US_EASTERN).normalize().tz_convert("UTC")
        )
        row = pd.DataFrame([{
            "timestamp": timestamp,
            "weighted_risk": round(float(risk_score), 3),
            "weighted_opportunity": round(float(opportunity_score), 3),
            "macro_adjustment": round(float(macro_score), 3),
            "net_risk": round(float(net_risk_score), 3)
        }])
        history = pd.concat([history, row], ignore_index=True)
        history = (
            history.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
        )
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(
            days=MODEL_SCORE_HISTORY_DAYS
        )
        history = history.loc[history["timestamp"] >= cutoff].copy()
        temp_path = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
        history.to_csv(temp_path, index=False)
        os.replace(temp_path, path)
        return history
    except Exception:
        return pd.DataFrame(columns=columns)

def _extract_yahoo_close(raw, requested_tickers):
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        level_0 = raw.columns.get_level_values(0)
        level_1 = raw.columns.get_level_values(1)
        if 'Close' in level_0:
            close = raw['Close'].copy()
        elif 'Close' in level_1:
            close = raw.xs('Close', axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    elif 'Close' in raw.columns and len(requested_tickers) == 1:
        close = raw[['Close']].rename(columns={'Close': requested_tickers[0]})
    else:
        return pd.DataFrame()

    if isinstance(close, pd.Series):
        close = close.to_frame(name=requested_tickers[0])
    close.columns = [str(col).upper() for col in close.columns]
    close.index = (
        pd.to_datetime(close.index, errors='coerce', utc=True)
        .tz_convert(None)
        .normalize()
    )
    close = close.loc[~close.index.isna()]
    close = close[~close.index.duplicated(keep='last')].sort_index()
    return close.apply(pd.to_numeric, errors='coerce').dropna(how='all')

def _download_yahoo_chart_symbol(symbol):
    encoded_symbol = quote(symbol, safe='')
    params = {"range": "1y", "interval": "1d", "events": "history"}
    last_error = None
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        try:
            response = requests.get(
                f"https://{host}/v8/finance/chart/{encoded_symbol}",
                params=params,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=8
            )
            if response.status_code == 429:
                raise RuntimeError("Yahoo HTTP 429")
            response.raise_for_status()
            result = response.json().get("chart", {}).get("result")
            if not result:
                raise RuntimeError("Yahoo chart 返回为空")
            result = result[0]
            timestamps = result.get("timestamp") or []
            quotes = result.get("indicators", {}).get("quote") or []
            closes = quotes[0].get("close") if quotes else []
            if not timestamps or not closes:
                raise RuntimeError("Yahoo chart 缺少日期或收盘价")
            series = pd.Series(
                closes,
                index=(
                    pd.to_datetime(timestamps, unit='s', utc=True)
                    .tz_convert(None)
                    .normalize()
                ),
                name=symbol
            )
            return pd.to_numeric(series, errors='coerce').dropna()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"{symbol} 直接行情失败: {last_error}")

def _download_nasdaq_history_symbol(symbol):
    """
    NASDAQ 官方历史行情作为非 Yahoo 冷启动降级源。
    支持本应用所需 ETF 以及 NDX；MOVE 不在该接口的代码表中。
    """
    if symbol == '^NDX':
        api_symbol, asset_class = 'NDX', 'index'
    elif symbol in {'QQQ', 'SPY', 'IWM', 'RSP', 'HYG', 'LQD'}:
        api_symbol, asset_class = symbol, 'etf'
    else:
        raise RuntimeError(f"NASDAQ 降级源不支持 {symbol}")

    today = datetime.date.today()
    params = {
        "assetclass": asset_class,
        "fromdate": (today - datetime.timedelta(days=370)).isoformat(),
        "todate": today.isoformat(),
        "limit": "5000"
    }
    response = requests.get(
        f"https://api.nasdaq.com/api/quote/{quote(api_symbol, safe='')}/historical",
        params=params,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/"
        },
        timeout=12
    )
    if response.status_code == 429:
        raise RuntimeError("NASDAQ HTTP 429")
    response.raise_for_status()
    rows = (
        response.json().get("data", {})
        .get("tradesTable", {})
        .get("rows")
    )
    if not rows:
        raise RuntimeError(f"NASDAQ {symbol} 历史行情为空")

    frame = pd.DataFrame(rows)
    if 'date' not in frame.columns or 'close' not in frame.columns:
        raise RuntimeError(f"NASDAQ {symbol} 返回列结构变化")
    values = pd.to_numeric(
        frame['close'].astype(str).str.replace(r'[$,]', '', regex=True),
        errors='coerce'
    )
    series = pd.Series(
        values.values,
        index=pd.to_datetime(frame['date'], errors='coerce'),
        name=symbol
    )
    series = series.loc[~series.index.isna()].dropna().sort_index()
    if series.shape[0] < 126:
        raise RuntimeError(f"NASDAQ {symbol} 有效样本不足")
    return series

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_yahoo_core_close():
    """
    仅保留一个 Yahoo 批次，并关闭 yfinance 多线程，避免冷启动时请求风暴。
    yfinance 失败时依次尝试 NASDAQ 官方历史行情与 Yahoo chart 端点；
    仍失败则读取最近成功缓存。
    此函数抛出的异常不会被 st.cache_data 缓存。
    """
    global YAHOO_ATTEMPTED_THIS_RUN, YAHOO_LAST_ERROR
    if YAHOO_ATTEMPTED_THIS_RUN:
        raise RuntimeError(
            YAHOO_LAST_ERROR or "本轮页面渲染已尝试 Yahoo，停止重复请求。"
        )
    YAHOO_ATTEMPTED_THIS_RUN = True

    errors = []
    close = pd.DataFrame()
    cached = _read_frame_cache("yahoo_core_close.csv")
    cached_has_required = all(
        symbol in cached.columns and cached[symbol].dropna().shape[0] >= 126
        for symbol in YAHOO_REQUIRED_TICKERS
    )

    try:
        raw = yf.download(
            YAHOO_CORE_TICKERS,
            period='1y',
            interval='1d',
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
            timeout=10,
            group_by='column'
        )
        close = _extract_yahoo_close(raw, YAHOO_CORE_TICKERS)
    except Exception as exc:
        errors.append(f"yfinance: {exc}")

    missing = [
        symbol for symbol in YAHOO_CORE_TICKERS
        if symbol != '^MOVE'
        and (symbol not in close.columns or close[symbol].dropna().shape[0] < 30)
    ]
    for symbol in missing:
        try:
            close[symbol] = _download_nasdaq_history_symbol(symbol)
            time.sleep(0.15)
        except Exception as exc:
            errors.append(str(exc))

    missing = [
        symbol for symbol in YAHOO_CORE_TICKERS
        if symbol != '^MOVE'
        and (symbol not in close.columns or close[symbol].dropna().shape[0] < 30)
    ]
    live_has_required = all(
        symbol in close.columns and close[symbol].dropna().shape[0] >= 126
        for symbol in YAHOO_REQUIRED_TICKERS
    )
    if not live_has_required and cached_has_required:
        # 已有最后成功数据时不要在 NASDAQ 降级也失败后继续轰炸 Yahoo。
        return cached

    consecutive_failures = 0
    for symbol in missing:
        try:
            close[symbol] = _download_yahoo_chart_symbol(symbol)
            consecutive_failures = 0
            time.sleep(0.35)
        except Exception as exc:
            errors.append(str(exc))
            consecutive_failures += 1
            if consecutive_failures >= 2:
                # 同一出口连续失败通常是 IP 级限流，继续逐代码重试只会加重封锁。
                break

    has_required = all(
        symbol in close.columns and close[symbol].dropna().shape[0] >= 126
        for symbol in YAHOO_REQUIRED_TICKERS
    )
    if has_required:
        close = close.sort_index().ffill().dropna(how='all')
        close.attrs["data_source"] = "live_market_sources"
        close.attrs["cache_age_hours"] = 0.0
        _write_frame_cache(close, "yahoo_core_close.csv")
        return close

    if cached_has_required:
        return cached

    detail = " | ".join(errors[-4:]) if errors else "Yahoo 返回数据不完整"
    YAHOO_LAST_ERROR = (
        "Yahoo 核心行情不可用且没有最近成功缓存。"
        f"请确认 yfinance 已升级（建议 >=0.2.66）；详情: {detail}"
    )
    raise RuntimeError(YAHOO_LAST_ERROR)

def get_market_session(now_et=None):
    now_et = now_et or datetime.datetime.now(US_EASTERN)
    current_time = now_et.time()
    if now_et.weekday() >= 5:
        session = "closed"
    elif datetime.time(4, 0) <= current_time < datetime.time(9, 30):
        session = "premarket"
    elif datetime.time(9, 30) <= current_time < datetime.time(16, 0):
        session = "regular"
    elif datetime.time(16, 0) <= current_time < datetime.time(20, 0):
        session = "postmarket"
    else:
        session = "closed"
    labels = {
        "premarket": "盘前",
        "regular": "盘中",
        "postmarket": "盘后",
        "closed": "休市"
    }
    return {
        "session": session,
        "label": labels[session],
        "active": session != "closed",
        "now_et": now_et
    }

def _get_config_value(name, default=None):
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)

def _paid_volatility_source_enabled():
    """
    默认完全不用付费指数源：CBOE 日线是事实基准，Yahoo 盘中仅作可选覆盖。
    即使 Secrets 中遗留 MASSIVE_API_KEY，也不会发起请求；只有显式设为 true 才启用。
    """
    value = _get_config_value("ENABLE_PAID_VOLATILITY_SOURCE", "false")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def _massive_volatility_ticker_map():
    """
    默认使用 Massive 的 CBOE/ICE 指数代码；若账号中代码不同，可在
    MASSIVE_VOL_TICKERS 中用 "内部代码=供应商代码" 逗号分隔覆盖。
    例如：^VIX=I:VIX,^VXN=I:VXN,^VVIX=I:VVIX
    """
    mapping = dict(MASSIVE_VOLATILITY_DEFAULT_TICKERS)
    overrides = _get_config_value("MASSIVE_VOL_TICKERS", "")
    for item in str(overrides).split(','):
        if '=' not in item:
            continue
        internal, provider = (part.strip() for part in item.split('=', 1))
        if internal and provider:
            mapping[internal.upper()] = provider.upper()
    return mapping

def _massive_timestamp(value):
    """兼容 Massive REST 的纳秒时间戳与少数接口返回的毫秒时间戳。"""
    numeric = int(float(value))
    if numeric >= 10 ** 17:
        unit = 'ns'
    elif numeric >= 10 ** 14:
        unit = 'us'
    elif numeric >= 10 ** 11:
        unit = 'ms'
    else:
        unit = 's'
    return pd.to_datetime(numeric, unit=unit, utc=True)

@st.cache_data(ttl=MASSIVE_VOLATILITY_TTL_SECONDS, show_spinner=False)
def _fetch_massive_volatility_snapshot_cached(trading_date):
    """
    仅接受标记为 REAL-TIME 的指数快照，绝不把 15 分钟延迟数据伪装成实时。
    不抛出单一代码的失败，方便 VIX/VXN/VVIX/MOVE 独立降级到日线。
    """
    api_key = _get_config_value("MASSIVE_API_KEY")
    if not api_key:
        return pd.DataFrame()

    rows = []
    errors = []
    for internal_symbol, provider_symbol in _massive_volatility_ticker_map().items():
        try:
            response = requests.get(
                "https://api.massive.com/v3/snapshot/indices",
                params={"ticker": provider_symbol, "apiKey": api_key},
                timeout=10
            )
            if response.status_code == 429:
                raise RuntimeError("Massive HTTP 429")
            response.raise_for_status()
            results = response.json().get("results") or []
            item = next(
                (entry for entry in results
                 if str(entry.get("ticker", "")).upper() == provider_symbol),
                results[0] if results else None
            )
            if not item:
                raise RuntimeError("快照结果为空")
            if item.get("error"):
                raise RuntimeError(
                    f"{item.get('error')}: {item.get('message', '无权限或代码不存在')}"
                )
            timeframe = str(item.get("timeframe", "")).upper()
            if timeframe != "REAL-TIME":
                raise RuntimeError(
                    f"返回 {timeframe or '未知'} 数据，未获得 REAL-TIME 权限"
                )
            price = item.get("value")
            updated_at = item.get("last_updated")
            if price is None or updated_at is None:
                raise RuntimeError("快照缺少 value 或 last_updated")
            rows.append({
                "symbol": internal_symbol,
                "price": float(price),
                "timestamp": _massive_timestamp(updated_at),
                "source": "Massive Indices REAL-TIME"
            })
        except Exception as exc:
            errors.append(f"Massive {internal_symbol}: {exc}")

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.set_index("symbol")
    frame.attrs["provider_errors"] = errors
    return frame

def fetch_realtime_volatility_snapshot():
    """盘中读取独立的授权波动率快照；无授权或休市时安静回退日线。"""
    global VOLATILITY_LAST_ERROR
    session_info = get_market_session()
    if session_info["session"] != "regular":
        return pd.DataFrame()
    if not _paid_volatility_source_enabled():
        return pd.DataFrame()
    if not _get_config_value("MASSIVE_API_KEY"):
        return pd.DataFrame()
    try:
        frame = _fetch_massive_volatility_snapshot_cached(
            session_info["now_et"].date().isoformat()
        )
        errors = frame.attrs.get("provider_errors", [])
        VOLATILITY_LAST_ERROR = " | ".join(errors[-3:]) if errors else ""
        return frame
    except Exception as exc:
        VOLATILITY_LAST_ERROR = f"Massive 波动率快照失败: {exc}"
        return pd.DataFrame()

def _read_intraday_cache(max_age_minutes=90):
    path = _market_cache_path("intraday_snapshot.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    age_minutes = (time.time() - os.path.getmtime(path)) / 60
    if age_minutes > max_age_minutes:
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, index_col=0)
        frame.index = frame.index.astype(str)
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], errors="coerce", utc=True
        )
        frame = frame.dropna(subset=["price", "timestamp"])
        if frame.empty:
            return frame
        today_et = datetime.datetime.now(US_EASTERN).date()
        latest_date_et = frame["timestamp"].max().tz_convert(US_EASTERN).date()
        if latest_date_et != today_et:
            return pd.DataFrame()
        if "source" not in frame.columns:
            frame["source"] = "last_intraday_cache"
        frame["source"] = frame["source"].astype(str) + " (缓存)"
        return frame
    except Exception:
        return pd.DataFrame()

def _write_intraday_cache(frame):
    if frame is None or frame.empty:
        return
    path = _market_cache_path("intraday_snapshot.csv")
    temp_path = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    frame.to_csv(temp_path)
    os.replace(temp_path, path)

def _fetch_alpaca_etf_snapshot():
    api_key = _get_config_value("ALPACA_API_KEY")
    api_secret = _get_config_value("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("未配置 ALPACA_API_KEY / ALPACA_API_SECRET")

    feed = _get_config_value("ALPACA_FEED", "iex")
    response = requests.get(
        "https://data.alpaca.markets/v2/stocks/bars/latest",
        params={
            "symbols": ",".join(INTRADAY_ETF_TICKERS),
            "feed": feed
        },
        headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret
        },
        timeout=12
    )
    if response.status_code == 429:
        raise RuntimeError("Alpaca HTTP 429")
    response.raise_for_status()
    bars = response.json().get("bars") or {}
    rows = []
    for symbol, bar in bars.items():
        price = bar.get("c")
        timestamp = bar.get("t")
        if price is not None and timestamp:
            rows.append({
                "symbol": symbol,
                "price": float(price),
                "timestamp": pd.to_datetime(timestamp, utc=True),
                "source": f"Alpaca {feed}"
            })
    if not rows:
        raise RuntimeError("Alpaca 最新分钟线为空")
    return pd.DataFrame(rows).set_index("symbol")

def _extract_intraday_close(raw, requested_tickers):
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        level_0 = raw.columns.get_level_values(0)
        level_1 = raw.columns.get_level_values(1)
        if "Close" in level_0:
            close = raw["Close"].copy()
        elif "Close" in level_1:
            close = raw.xs("Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    elif "Close" in raw.columns and len(requested_tickers) == 1:
        close = raw[["Close"]].rename(columns={"Close": requested_tickers[0]})
    else:
        return pd.DataFrame()
    if isinstance(close, pd.Series):
        close = close.to_frame(name=requested_tickers[0])
    close.columns = [str(col).upper() for col in close.columns]
    close.index = pd.to_datetime(close.index, errors="coerce", utc=True)
    close = close.loc[~close.index.isna()].sort_index()
    return close.apply(pd.to_numeric, errors="coerce").dropna(how="all")

def _probe_yahoo_intraday_status():
    try:
        response = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/SPY",
            params={
                "range": "1d",
                "interval": "15m",
                "includePrePost": "true",
                "events": "history"
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8
        )
        if response.status_code == 429:
            return "Yahoo 诊断请求确认 HTTP 429（出口 IP 被限流）"
        if response.status_code != 200:
            return f"Yahoo 诊断请求 HTTP {response.status_code}"
        chart = response.json().get("chart", {})
        if chart.get("error"):
            return f"Yahoo chart error: {chart['error']}"
        result = chart.get("result") or []
        if not result or not result[0].get("timestamp"):
            return "Yahoo 诊断请求 HTTP 200，但分钟线内容为空"
        latest = pd.to_datetime(
            result[0]["timestamp"][-1], unit="s", utc=True
        )
        return (
            "Yahoo 诊断请求 HTTP 200；最新 SPY 时间 "
            + latest.tz_convert(US_EASTERN).strftime("%Y-%m-%d %H:%M ET")
            + "，更可能是 yfinance 解析/代码级数据缺失"
        )
    except Exception as exc:
        return f"Yahoo 诊断请求连接失败: {exc}"

def _fetch_yahoo_intraday_snapshot(tickers):
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(
        tickers,
        period="5d",
        interval="15m",
        prepost=True,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        timeout=8,
        group_by="column"
    )
    close = _extract_intraday_close(raw, tickers)
    rows = []
    for symbol in tickers:
        if symbol not in close.columns:
            continue
        series = close[symbol].dropna()
        if series.empty:
            continue
        rows.append({
            "symbol": symbol,
            "price": float(series.iloc[-1]),
            "timestamp": series.index[-1],
            "source": "Yahoo 15m"
        })
    if not rows:
        raise RuntimeError(
            "Yahoo 盘前/日内快照为空；" + _probe_yahoo_intraday_status()
        )
    return pd.DataFrame(rows).set_index("symbol")

@st.cache_data(ttl=INTRADAY_TTL_SECONDS, show_spinner=False)
def _fetch_intraday_snapshot_cached(session_name, trading_date):
    frames = []
    errors = []

    try:
        frames.append(_fetch_alpaca_etf_snapshot())
    except Exception as exc:
        errors.append(f"Alpaca: {exc}")

    covered = set()
    if frames:
        covered.update(frames[0].index)
    yahoo_tickers = [
        symbol for symbol in INTRADAY_ETF_TICKERS
        if symbol not in covered
    ]
    # 配置了授权指数源后，VIX/VXN/VVIX/MOVE 由独立 60 秒快照处理，
    # 不再每十分钟把这些指数重新打到 Yahoo，避免其一次失败拖累所有开关。
    if not _paid_volatility_source_enabled():
        yahoo_tickers.extend(INTRADAY_VOL_TICKERS)
    if session_name == "regular":
        yahoo_tickers.append("^NDX")

    try:
        frames.append(_fetch_yahoo_intraday_snapshot(yahoo_tickers))
    except Exception as exc:
        errors.append(f"Yahoo intraday: {exc}")

    if frames:
        snapshot = pd.concat(frames, axis=0)
        snapshot = snapshot[~snapshot.index.duplicated(keep="first")]
        snapshot["timestamp"] = pd.to_datetime(
            snapshot["timestamp"], errors="coerce", utc=True
        )
        snapshot["price"] = pd.to_numeric(snapshot["price"], errors="coerce")
        snapshot = snapshot.dropna(subset=["price", "timestamp"])

        now_utc = pd.Timestamp.now(tz="UTC")
        max_age_minutes = 45 if session_name == "regular" else 120
        age = (now_utc - snapshot["timestamp"]).dt.total_seconds() / 60
        newest_age = float(age.min()) if not age.empty else None
        snapshot = snapshot.loc[
            (age >= -5) & (age <= max_age_minutes)
        ].copy()
        if not snapshot.empty:
            snapshot.attrs["provider_errors"] = errors
            _write_intraday_cache(snapshot)
            return snapshot
        if newest_age is not None:
            errors.append(
                f"所有返回行情均过期；最新一条也已滞后 {newest_age:.0f} 分钟，"
                f"当前阈值为 {max_age_minutes} 分钟"
            )

    cached = _read_intraday_cache(max_age_minutes=90)
    if not cached.empty:
        cached.attrs["provider_errors"] = errors
        return cached
    raise RuntimeError(
        "盘前/日内快照不可用: " + " | ".join(errors[-3:])
    )

def fetch_intraday_snapshot():
    global INTRADAY_ATTEMPTED_THIS_RUN
    global INTRADAY_SNAPSHOT_THIS_RUN
    global INTRADAY_LAST_ERROR

    session_info = get_market_session()
    if not session_info["active"]:
        return pd.DataFrame()
    if INTRADAY_SNAPSHOT_THIS_RUN is not None:
        return INTRADAY_SNAPSHOT_THIS_RUN
    if INTRADAY_ATTEMPTED_THIS_RUN:
        return pd.DataFrame()

    INTRADAY_ATTEMPTED_THIS_RUN = True
    base_snapshot = pd.DataFrame()
    provider_errors = []
    try:
        base_snapshot = _fetch_intraday_snapshot_cached(
            session_info["session"],
            session_info["now_et"].date().isoformat()
        )
    except Exception as exc:
        provider_errors.append(str(exc))

    # 授权指数快照独立于 Yahoo/Alpaca 的十分钟 ETF 快照，可每分钟更新，
    # 并且在 ETF 通道临时故障时仍保留 VIX/VXN/VVIX/MOVE 的盘中数据。
    volatility_snapshot = fetch_realtime_volatility_snapshot()
    provider_errors.extend(base_snapshot.attrs.get("provider_errors", []))
    provider_errors.extend(volatility_snapshot.attrs.get("provider_errors", []))
    if VOLATILITY_LAST_ERROR:
        provider_errors.append(VOLATILITY_LAST_ERROR)

    frames = [frame for frame in (volatility_snapshot, base_snapshot) if not frame.empty]
    if not frames:
        INTRADAY_LAST_ERROR = "盘前/日内快照不可用: " + " | ".join(
            provider_errors[-4:]
        )
        return pd.DataFrame()

    # 授权波动率源优先于 Yahoo 同名指数，避免低质量备用报价覆盖授权实时值。
    snapshot = pd.concat(frames, axis=0)
    snapshot = snapshot[~snapshot.index.duplicated(keep="first")]
    snapshot.attrs["provider_errors"] = list(dict.fromkeys(provider_errors))
    INTRADAY_SNAPSHOT_THIS_RUN = snapshot
    INTRADAY_LAST_ERROR = ""
    return snapshot

def overlay_intraday_prices(daily_frame, symbols, require_all=False):
    if daily_frame is None or daily_frame.empty:
        return daily_frame
    snapshot = fetch_intraday_snapshot()
    available = [symbol for symbol in symbols if symbol in snapshot.index]
    if require_all and len(available) != len(symbols):
        result = daily_frame.copy()
        result.attrs.update(daily_frame.attrs)
        result.attrs["quote_mode"] = "daily_fallback"
        return result
    if not available:
        result = daily_frame.copy()
        result.attrs.update(daily_frame.attrs)
        result.attrs["quote_mode"] = "daily_fallback"
        return result

    result = daily_frame.copy()
    original_attrs = dict(daily_frame.attrs)
    for symbol in available:
        timestamp = pd.Timestamp(snapshot.loc[symbol, "timestamp"])
        trading_date = pd.Timestamp(
            timestamp.tz_convert(US_EASTERN).date()
        )
        if symbol not in result.columns:
            result[symbol] = np.nan
        result.loc[trading_date, symbol] = float(snapshot.loc[symbol, "price"])

    result = result.sort_index().ffill()
    result.attrs.update(original_attrs)
    latest_timestamp = snapshot.loc[available, "timestamp"].max()
    age_minutes = (
        pd.Timestamp.now(tz="UTC") - latest_timestamp
    ).total_seconds() / 60
    result.attrs.update({
        "quote_mode": get_market_session()["session"],
        "intraday_asof": latest_timestamp.isoformat(),
        "quote_age_minutes": max(0, round(age_minutes, 1)),
        "quote_sources": ", ".join(
            sorted(set(snapshot.loc[available, "source"].astype(str)))
        ),
        "intraday_symbols": available
    })
    return result

def market_data_timestamp(frame):
    if frame is not None and frame.attrs.get("intraday_asof"):
        timestamp = pd.Timestamp(frame.attrs["intraday_asof"])
        return timestamp.tz_convert(US_EASTERN).strftime(
            "%Y-%m-%d %H:%M ET"
        )
    if frame is not None and not frame.empty:
        return pd.Timestamp(frame.index[-1]).strftime("%Y-%m-%d") + " 日线"
    return "无可用时间戳"

def _request_json_with_backoff(url, params=None, attempts=3):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36"
    }
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=12
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else 2 ** (attempt + 1)
                raise RuntimeError(f"HTTP 429，建议等待 {wait_seconds:.0f} 秒")
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") not in (None, "0"):
                raise RuntimeError(payload.get("msg") or f"API code={payload.get('code')}")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"API 请求失败: {last_error}")

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_okx_signal_inputs():
    """一次成功后缓存 30 分钟；异常直接抛出，因此临时失败不会被缓存。"""
    price = _request_json_with_backoff(
        "https://www.okx.com/api/v5/market/history-candles",
        params={"instId": "BTC-USDT", "bar": "1Dutc", "limit": "60"}
    )
    oi = _request_json_with_backoff(
        "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume",
        params={"ccy": "BTC", "period": "1D"}
    )
    funding = _request_json_with_backoff(
        "https://www.okx.com/api/v5/public/funding-rate-history",
        params={"instId": "BTC-USDT-SWAP", "limit": "100"}
    )
    return price, oi, funding

RISK_SCORE_MAP = {
    "极高风险": 100,
    "高风险": 82,
    "中高风险": 66,
    "中风险": 48,
    "中性偏稳": 30,
    "中性": 28,
    "低风险": 18,
    "低风险/机会": 12,
    "数据风险": 35,
    "数据缺失": 40,
}

OPPORTUNITY_SCORE_MAP = {
    "强抄底": 92,
    "抄底": 78,
    "机会": 64,
    "低风险/机会": 58,
    "观察": 28,
    "无": 0,
}

CYCLE_META = {
    "intraday": ("盘中实时", 1.00, "对情绪拐点和风控触发最敏感，适合做当天仓位和止盈线调整。"),
    "daily": ("日级别", 0.82, "用于确认资金结构和趋势状态，不适合被单日噪音反复打脸。"),
    "post_close": ("盘后日更", 0.74, "适合判断底层资金方向，盘中价格冲击时需配合实时波动指标确认。"),
    "hybrid": ("日线+盘中", 0.92, "兼顾趋势确认和实时杠杆情绪，权重保留但避免过度交易。"),
}

def level_to_score(level, score_map, default=35):
    return score_map.get(level, default)

def infer_badge_from_scores(risk_score, opportunity_score):
    if risk_score >= 72:
        return "风险预警"
    if opportunity_score >= 64 and risk_score < 66:
        return "抄底/修复"
    return "中性观察"

def build_strategy_text(risk_level, opportunity_level, cycle_key):
    if risk_level in ("极高风险", "高风险"):
        return "策略：降低净多头敞口，收紧止盈线；只保留强趋势或高EV仓位，禁止加杠杆。"
    if risk_level == "中高风险":
        return "策略：不追涨，优先做仓位体检；等实时指标确认降温后再恢复进攻。"
    if opportunity_level in ("强抄底", "抄底"):
        return "策略：允许分批进攻，但需要至少一个实时指标不再恶化；优先选择错杀高质量标的。"
    if opportunity_level in ("机会", "低风险/机会"):
        return "策略：可维持或小幅增加核心仓位，避免一次性满仓，等待更多开关共振。"
    if cycle_key == "post_close":
        return "策略：作为方向底稿，不单独决定盘中交易；盘中必须看 VIX/VXN 或价格行为确认。"
    return "策略：维持中性仓位，等待风险或机会分数突破阈值后再行动。"

def enrich_switch(s):
    cycle_label, cycle_weight, cycle_note = CYCLE_META.get(s.get("cycle_key", "daily"), CYCLE_META["daily"])
    risk_level = s.get("risk_level", "高风险" if s.get("top_active") else "中性")
    opportunity_level = s.get("opportunity_level", "抄底" if s.get("bottom_active") else "无")
    risk_score = s.get("risk_score", level_to_score(risk_level, RISK_SCORE_MAP))
    opportunity_score = s.get("opportunity_score", level_to_score(opportunity_level, OPPORTUNITY_SCORE_MAP, 0))
    effective_weight = s.get("weight", 1.0) * cycle_weight
    net_score = (risk_score - opportunity_score) * effective_weight
    badge_label = infer_badge_from_scores(risk_score, opportunity_score)
    strategy = s.get("strategy", build_strategy_text(risk_level, opportunity_level, s.get("cycle_key", "daily")))
    enriched = dict(s)
    enriched.update({
        "cycle_label": cycle_label,
        "cycle_weight": cycle_weight,
        "cycle_note": cycle_note,
        "risk_level": risk_level,
        "opportunity_level": opportunity_level,
        "risk_score": risk_score,
        "opportunity_score": opportunity_score,
        "effective_weight": effective_weight,
        "net_score": net_score,
        "badge_label": badge_label,
        "strategy": strategy,
        "top_active": s.get("top_active", False) or risk_score >= 72,
        "bottom_active": s.get("bottom_active", False) or (opportunity_score >= 76 and risk_score < 72),
    })
    return enriched

def fetch_vix_data():
    try:
        close_data = pd.concat(
            {
                '^VIX': fetch_cboe_official_history('VIX'),
                '^VIX3M': fetch_cboe_official_history('VIX3M')
            },
            axis=1
        ).sort_index().ffill().dropna().tail(90)
        close_data = overlay_intraday_prices(
            close_data, ['^VIX', '^VIX3M'], require_all=True
        ).tail(90)
        if not close_data.empty:
            close_data['Ratio'] = close_data['^VIX3M'] / close_data['^VIX']
            
            # 【新增】引入微观快线(EMA5)与趋势慢线(EMA21)作为期限结构比率的动能依据
            close_data['Ratio_Fast'] = close_data['Ratio'].ewm(span=5, adjust=False).mean()
            close_data['Ratio_Slow'] = close_data['Ratio'].ewm(span=21, adjust=False).mean()
        
            vix = close_data['^VIX'].iloc[-1]
            vix3m = close_data['^VIX3M'].iloc[-1]
            
            if np.isnan(vix): vix = close_data['^VIX'].dropna().iloc[-1]
            if np.isnan(vix3m): vix3m = close_data['^VIX3M'].dropna().iloc[-1]
                
            ratio = vix3m / vix
            
            prev_ratio = 1.0
            valid_ratios = close_data['Ratio'].dropna()
            if len(valid_ratios) >= 2:
                prev_ratio = valid_ratios.iloc[-2]
                
            # 【新增】提取当前与前一日的 EMA 状态
            fast_curr = close_data['Ratio_Fast'].iloc[-1]
            slow_curr = close_data['Ratio_Slow'].iloc[-1]
            fast_prev = close_data['Ratio_Fast'].iloc[-2] if len(close_data) >= 2 else fast_curr
            slow_prev = close_data['Ratio_Slow'].iloc[-2] if len(close_data) >= 2 else slow_curr
            
            is_death_cross = (fast_prev >= slow_prev) and (fast_curr < slow_curr)
            is_golden_cross = (fast_prev <= slow_prev) and (fast_curr > slow_curr)
          
            # -----------------------------------------------------------------
            # 【重构】修改后的抄底、逃顶开关触发方式
            # -----------------------------------------------------------------
            # 抄底触发：比率由倒挂向上突破 1.0 平衡线，或者在低位区(ratio <= 1.05)发生了 EMA 动能金叉修复
            bottom_active = ((prev_ratio <= 1.0) and (ratio > 1.0)) or (is_golden_cross and ratio <= 1.05)
            
            # 逃顶触发：比率冲破 1.25 绝对高位线，或者高位跌破 1.0 平衡线，或者在高位自满警戒带(ratio >= 1.15)发生了均线死叉
            top_active = (ratio >= 1.25) or ((prev_ratio >= 1.0) and (ratio < 1.0)) or (is_death_cross and ratio >= 1.15)
            
            # -----------------------------------------------------------------
            # 【重构】补充开关2每种细分情况的深入动态决策文字描述
            # -----------------------------------------------------------------
            ema_info = f" [EMA5:{fast_curr:.3f}, EMA21:{slow_curr:.3f}]"
            
            if ratio >= 1.25:
                if fast_curr < slow_curr or is_death_cross:
                    vix_ratio_diag = f"【比率极限见顶死叉】当前比率({ratio:.3f})冲破1.25绝对高线且EMA线死叉{ema_info}。做空拥挤盘遭遇情绪拐点反噬，多杀多风险极高，执行最高级别战略撤退。"
                else:
                    vix_ratio_diag = f"【比率极限超载发散】当前比率({ratio:.3f})冲破1.25绝对高线{ema_info}。市场极度懈怠自满，高位做空波动率策略严重过载，需严防高位突发踩踏闪崩。"
            elif bottom_active:
                vix_ratio_diag = f"【比率均线跨线/金叉修复】当前比率({ratio:.3f}){ema_info}。期限结构摆脱深度倒挂或低位达成共振修复，意味着恐慌衰竭，右侧安全抄底黄金点激活。"
            elif (prev_ratio >= 1.0) and (ratio < 1.0):
                vix_ratio_diag = f"【比率跌破平衡临界】今日比率跌破平衡至({ratio:.3f}){ema_info}。期限结构常态Contango基石全面瓦解，向倒挂过渡，大盘防线松动风险激增。"
            elif ratio <= 1.0:
                if fast_curr > slow_curr or is_golden_cross:
                    vix_ratio_diag = f"【比率倒挂带微观金叉】当前比率({ratio:.3f})&lt;=1持续倒挂，但EMA出现微观金叉回暖{ema_info}。提示非理性无差别抛售最恐慌期已过，左侧洗盘进入尾声。"
                else:
                    vix_ratio_diag = f"【比率持续深度倒挂】当前比率({ratio:.3f})&lt;=1且均线空头排列{ema_info}。全市场系统流动性仍处于冰点宣泄期，需保持严格现金观望，克制接飞刀冲动。"
            elif 1.15 <= ratio < 1.25:
                if fast_curr < slow_curr:
                    vix_ratio_diag = f"【高位自满动能死叉】当前比率({ratio:.3f})处于高位警戒带且出现均线转弱死叉{ema_info}。多头买盘边际枯竭，情绪转弱，建议分批减仓或收紧止盈。"
                else:
                    vix_ratio_diag = f"【比率高位常规自满】当前比率({ratio:.3f})处于1.15-1.25敏感带{ema_info}，快线维持在慢线上方。市场多头乐观情绪正常化积压，可持股但禁止加杠杆。"
            else:
                vix_ratio_diag = f"【比率常态健康中轴】当前比率({ratio:.3f})在1.0-1.15 Contango区间稳健运行{ema_info}。期限结构和情绪动能健康，大盘暂无宏观性异动异变风险。"

            # VIX现货分项诊断
            if vix >= 24.0:
                vix_spot_diag = f"【现货恐慌爆发】当前VIX现货飙升至 {vix:.2f}，突破24.0恐慌红线，全市场做空期权对冲踩踏剧烈，抛压处于高位宣泄状态。"
            elif vix < 13.5:
                vix_spot_diag = f"【现货极低自满】当前VIX现货极低为 {vix:.2f} (&lt;13.5)，市场对潜在黑天鹅尾部风险毫无对冲防备，极易被动洗盘。"
            else:
                vix_spot_diag = f"【现货常态理性】当前VIX现货为 {vix:.2f}，处于合理宽幅震荡区间，情绪中性，大盘系统性踩踏概率较低。"

            # 综合诊断状态标识
            if bottom_active:
                if vix >= 24.0:
                    vix_diag_status = "🚀 黄金抄底：现货极端恐慌 ✖ 期限比率动能完美金叉修复"
                else:
                    vix_diag_status = "🟢 抄底激活：期限比率率先跨线上破或低位动能金叉"
            elif top_active:
                if ratio >= 1.25 and vix < 13.5:
                    vix_diag_status = "🚨 极度逃顶：现货极限自满 ✖ 比率>1.25极端过热超载"
                elif fast_curr < slow_curr and ratio >= 1.15:
                    vix_diag_status = "🚨 逃顶激活：高位敏感带均线死叉，多头做多动能确认转弱"
                else:
                    vix_diag_status = "🚨 风控激活：比率结构破位跌破1.0平衡线"
            else:
                if ratio <= 1.0 and vix >= 24.0:
                    vix_diag_status = "🔴 严重防御：期限结构持续倒挂 ✖ 现货强恐慌快速抛售"
                elif ratio <= 1.0:
                    vix_diag_status = "🟡 风险提示：期限结构深陷倒挂（微观EMA呈现低位金叉转机）" if fast_curr > slow_curr else "🟡 风险提示：期限结构深陷持续倒挂冰点期"
                elif 1.15 <= ratio < 1.25:
                    vix_diag_status = "🟡 风险提示：高位自满情绪规律性压制（建议提高保护性止盈）"
                else:
                    if vix < 13.5:
                        vix_diag_status = "🟡 风险提示：现货波动率处于绝对低位，保留左侧防御风险"
                    else:
                        vix_diag_status = "🟢 状态中性：健康常态化牛市状态"
            
            return {
                "vix": round(vix, 2), "vix3m": round(vix3m, 2), "ratio": round(ratio, 3), "prev_ratio": round(prev_ratio, 3),
                "bottom_active": bottom_active, "top_active": top_active, "error": False,
                "vix_diag_status": vix_diag_status,
                "vix_ratio_diag": vix_ratio_diag,
                "vix_spot_diag": vix_spot_diag,
                "df": close_data.tail(60),
                "fetched_at": market_data_timestamp(close_data),
                "quote_mode": close_data.attrs.get("quote_mode", "daily_fallback"),
                "quote_age_minutes": close_data.attrs.get("quote_age_minutes"),
                "quote_sources": close_data.attrs.get("quote_sources", "CBOE EOD")
            }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False, "fetched_at": "异常断流"}
    return {"error": True, "msg": "No data", "bottom_active": False, "top_active": False, "fetched_at": "空数据"}

def fetch_crypto_signals():
    try:
        # BTC 价格、OI 与资金费率全部走 OKX 官方 API，彻底移除该开关的 Yahoo 依赖。
        price_res, r_res, fr_res = fetch_okx_signal_inputs()

        if not price_res.get("data"):
            raise Exception("OKX BTC 日线接口无返回数据")
        price_data = []
        for row in price_res['data']:
            ts = pd.to_datetime(int(row[0]), unit='ms', utc=True).normalize()
            price_data.append({'timestamp': ts, 'close': float(row[4])})
        df_price = (
            pd.DataFrame(price_data)
            .drop_duplicates('timestamp', keep='last')
            .set_index('timestamp')
            .sort_index()
        )

        # 2. 持仓量 (OI)：使用 OKX Rubik 历史接口
        if not r_res.get("data"):
            raise Exception(f"OKX OI 接口异常: {r_res.get('msg', '无返回数据')}")
            
        oi_data = []
        for row in r_res['data']:
            ts = pd.to_datetime(int(row[0]), unit='ms', utc=True).normalize()
            oi_btc = float(row[1])
            oi_data.append({'timestamp': ts, 'oi': oi_btc})
        df_oi = pd.DataFrame(oi_data).set_index('timestamp')

        # 3. 资金费率 (FR)：使用 OKX 历史资金费率接口
        if not fr_res.get("data"):
            raise Exception(f"OKX FR 接口异常: {fr_res.get('msg', '无返回数据')}")
            
        fr_data = []
        for row in fr_res['data']:
            ts = pd.to_datetime(int(row['fundingTime']), unit='ms', utc=True).normalize()
            fr_rate = float(row['fundingRate']) * 100
            fr_data.append({'timestamp': ts, 'funding_rate': fr_rate})
        df_fr = pd.DataFrame(fr_data)
        # 每天可能有3个费率(8小时一次结算)，按天取平均值平滑处理
        df_fr = df_fr.groupby('timestamp')['funding_rate'].mean().to_frame()

        # 4. 数据合并：强制时间轴绝对对齐
        df_merged = df_price.join(df_oi, how='inner').join(df_fr, how='inner')
        df_merged = df_merged.sort_index().ffill()

        if df_merged.empty or len(df_merged) < 7:
            raise Exception(f"数据源合并失败或样本过少 (当前成功对齐天数: {len(df_merged)})")

        # 5. 引入均线计算(MA7)，抹平日内噪音，让判断更精准
        df_merged['oi_ma7'] = df_merged['oi'].rolling(7).mean()
        df_merged['price_ma7'] = df_merged['close'].rolling(7).mean()
        df_merged = df_merged.dropna()

        current_row = df_merged.iloc[-1]
        prev_row = df_merged.iloc[-2]

        current_price = current_row['close']
        prev_price = prev_row['close']
        current_oi = current_row['oi']
        oi_ma7 = current_row['oi_ma7']
        current_fr = current_row['funding_rate']

        # 趋势判定文字
        price_up = current_price > prev_price
        price_trend_str = "上涨" if price_up else "下跌"
        oi_trend_str = "显著扩张" if current_oi > oi_ma7 * 1.05 else ("温和扩张" if current_oi >= oi_ma7 else "萎缩清算")

        # 6. 核心逻辑矩阵
        bottom_active = False
        top_active = False
        
        # 逃顶：多头过载（将绝对阈值 0.01% 优化为 0.025% 真过热线，并配合均线偏离度）
        if current_price > current_row['price_ma7'] and current_oi > oi_ma7 * 1.03 and current_fr >= 0.025:
            diag_status = "🚨 【极度危险/逃顶】价格多头 ✖ OI显著膨胀 ✖ 费率过热(>=0.025%)。杠杆极度拥挤，随时引发多头连环踩踏。"
            top_active = True
            
        # 博弈预警：价格弱势但 OI 逆势飙升
        elif current_price < current_row['price_ma7'] and current_oi > oi_ma7 * 1.05 and current_fr < -0.01:
            diag_status = "🟡 【轧空预警/博弈】价格弱势 ✖ OI逆势飙升 ✖ 费率深度转负。空头大军集结，需严防主力无预警暴力“逼空(Squeeze)”。"
            
        # 抄底：黄金右侧结构
        elif current_price < current_row['price_ma7'] and current_oi < oi_ma7 * 0.95 and current_fr <= 0.005:
            diag_status = "🟢 【黄金右侧/抄底】价格回落 ✖ OI深度清算 ✖ 费率降温触底。杠杆泡沫出清完毕，具备极佳的右侧筑底赔率。"
            bottom_active = True
            
        # 假突破：现货无买盘
        elif price_up and current_oi < oi_ma7 * 0.92:
            diag_status = "⚠️ 【假突破预警/缩量】价格反弹 ✖ OI显著萎缩。缺乏现货真实买盘，上涨大概率为“空头平仓(踏空回补)”推动，动能难以为继。"
            top_active = True
            
        # 常规健康持仓
        elif current_price >= current_row['price_ma7'] and current_oi >= oi_ma7 * 0.95 and 0.005 <= current_fr < 0.025:
            diag_status = "📈 【健康延续/持仓】价格企稳 ✖ OI支撑 ✖ 费率常态。真金白银良性流入，多头趋势稳健延续。"
            
        else:
            diag_status = f"⚪ 【震荡博弈/观望】价格均线缠绕，OI变动平缓({oi_trend_str})，系统处于风险真空期。"

        return {
            "btc_price": f"${current_price:,.2f}",
            "price_trend": price_trend_str,
            "oi": f"{current_oi:,.0f} BTC", 
            "oi_trend": oi_trend_str,
            "funding_rate": f"{current_fr:.4f}%", 
            "diag_status": diag_status,
            "bottom_active": bottom_active, 
            "top_active": top_active, 
            "error": False,
            "hist_df": df_merged.tail(30), # 提供最近30天数据供图标渲染
            "fetched_at": datetime.datetime.now(US_EASTERN).strftime(
                '%Y-%m-%d %H:%M:%S ET'
            )
        }
    except Exception as e:
        # 捕获真实报错抛给前端
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False, "fetched_at": "异常拦截"}

@st.cache_data(ttl=3600)
def fetch_squeezemetrics_data():
    url = "https://squeezemetrics.com/monitor/static/DIX.csv"
    cache_file = "dix_cache.csv"
    cooldown_seconds = 4 * 60 * 60
    should_download = True
    
    if os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < cooldown_seconds:
            should_download = False
            
    if should_download:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            }
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200 and "dix" in response.text.lower():
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(response.text)
        except Exception:
            pass
            
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            if not df.empty:
                df.columns = df.columns.str.lower()
                latest = df.iloc[-1]
                dix_val = float(latest['dix'])
                if dix_val < 1.0: dix_val = dix_val * 100
                gex_val = float(latest['gex'])
                
                file_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file)).strftime('%Y-%m-%d %H:%M:%S')
                
                return {
                    "dix": round(dix_val, 2), "gex": int(gex_val),
                    "error": False, "df": df.tail(100), "is_mock": False,
                    "fetched_at": file_time
                }
        except Exception:
            pass

    dates = pd.date_range(end=datetime.date.today(), periods=100)
    mock_df = pd.DataFrame({
        'date': dates, 'dix': np.sin(np.linspace(0, 10, 100)) * 3 + 44,
        'gex': np.random.normal(loc=500000000, scale=1000000000, size=100)
    })
    latest = mock_df.iloc[-1]
    return {
        "dix": round(latest['dix'], 2), "gex": int(latest['gex']),
        "error": False, "df": mock_df, "is_mock": True,
        "fetched_at": datetime.datetime.now(US_EASTERN).strftime(
            '%Y-%m-%d %H:%M:%S ET'
        ) + " (兜底)"
    }

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_cboe_official_history_cached(symbol):
    """
    CBOE 官方 CSV 是 VIX/VIX3M/VXN/VVIX/COR1M/DSPX 的首选源。
    只缓存成功数据；网络临时失败时使用最近 10 天内的最后成功快照。
    """
    cache_name = f"cboe_{symbol.upper()}.csv"
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
    last_error = None

    for attempt in range(2):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15
            )
            if response.status_code == 429:
                raise RuntimeError("CBOE HTTP 429")
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            df.columns = df.columns.str.strip().str.lower()
            date_col = 'trade_date' if 'trade_date' in df.columns else 'date'
            value_col = 'close' if 'close' in df.columns else symbol.lower()
            if date_col not in df.columns or value_col not in df.columns:
                raise RuntimeError(f"CBOE {symbol} CSV 列结构变化")
            result = pd.DataFrame({
                symbol.upper(): pd.to_numeric(df[value_col], errors='coerce').values
            }, index=pd.to_datetime(df[date_col], errors='coerce'))
            result = (
                result.loc[~result.index.isna()]
                .dropna()
                .sort_index()
                .tail(800)
            )
            if result.shape[0] < 30:
                raise RuntimeError(f"CBOE {symbol} 有效样本不足")
            _write_frame_cache(result, cache_name)
            return result.iloc[:, 0].rename(symbol.upper())
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(2)

    cached = _read_frame_cache(cache_name)
    if not cached.empty:
        return cached.iloc[:, 0].rename(symbol.upper())
    raise RuntimeError(f"CBOE {symbol} 数据不可用: {last_error}")

def fetch_cboe_official_history(symbol):
    try:
        return _fetch_cboe_official_history_cached(symbol)
    except Exception:
        # 包装层不缓存失败结果，下次 Streamlit 重跑仍可立即恢复。
        return pd.Series(dtype=float, name=symbol.upper())

def calculate_quant_and_breadth_signals():
    try:
        yf_data = fetch_yahoo_core_close()
        required = ['QQQ', 'SPY', 'IWM', 'RSP']
        yf_data = overlay_intraday_prices(
            yf_data, required, require_all=True
        )
        missing = [
            symbol for symbol in required
            if symbol not in yf_data.columns or yf_data[symbol].dropna().shape[0] < 126
        ]
        if missing:
            raise Exception(f"CTA 核心行情缺失: {', '.join(missing)}")
        data = yf_data[required].dropna().tail(260).copy()
        latest = data.iloc[-1]
        
        corr_official = fetch_cboe_official_history('COR1M')
        dspx_official = fetch_cboe_official_history('DSPX')
        
        if not corr_official.empty:
            corr_series = corr_official.reindex(data.index)
            if np.isnan(corr_series.iloc[-1]) or corr_series.tail(10).max() == corr_series.tail(10).min():
                corr_series.iloc[-1] = corr_series.dropna().iloc[-1] if not corr_series.dropna().empty else 17.8
            corr_series = corr_series.ffill()
        else:
            # 保留 CTA 开关运行，相关性开关会因常数序列自动标记为数据断层。
            corr_series = pd.Series(17.8, index=data.index, name='COR1M')
            
        if not dspx_official.empty:
            dspx_series = dspx_official.reindex(data.index)
            if np.isnan(dspx_series.iloc[-1]):
                dspx_series.iloc[-1] = dspx_series.dropna().iloc[-1] if not dspx_series.dropna().empty else 40.0
            dspx_series = dspx_series.ffill()
        else:
            dspx_series = pd.Series(40.0, index=data.index, name='DSPX')

        # CTA 动向追踪
        cta_shorts_series = pd.Series(0, index=data.index)
        cta_longs_series = pd.Series(0, index=data.index)
        for idx_name in ['QQQ', 'SPY', 'IWM']:
            price = data[idx_name]
            ma21 = price.rolling(21).mean()
            ma63 = price.rolling(63).mean()
            ma126 = price.rolling(126).mean()
            short_mask = (price < ma21) & (price < ma63) & (price < ma126) & ((price - ma21) / ma21 < -0.04)
            long_mask = (price > ma21) & (price > ma63) & (price > ma126) & ((price - ma21) / ma21 > 0.04)
            cta_shorts_series += short_mask.astype(int)
            cta_longs_series += long_mask.astype(int)

        cta_shorts_exhausted = cta_shorts_series.iloc[-1]
        cta_longs_exhausted = cta_longs_series.iloc[-1]
        cta_bottom_active = cta_shorts_exhausted >= 2
        cta_top_active = cta_longs_exhausted >= 2
        
        cta_status_text = "多头趋势/系统性买入中"
        if cta_bottom_active: cta_status_text = "系统性空头抛压耗尽"
        elif cta_top_active: cta_status_text = "系统性多头买盘枯竭"
        elif cta_shorts_exhausted > 0 or cta_longs_exhausted > 0: cta_status_text = "CTA 动量分化调仓期"

        # CBOE 交叉盘及动态分情况推演
        corr_is_broken = corr_series.tail(10).max() == corr_series.tail(10).min()
        corr_fast = corr_series.ewm(span=5, adjust=False).mean()
        corr_slow = corr_series.ewm(span=21, adjust=False).mean()
        corr_q75 = corr_series.rolling(126).quantile(0.75) 
        corr_q25 = corr_series.rolling(126).quantile(0.25)
         
        dsp_fast = dspx_series.ewm(span=5, adjust=False).mean()
        dsp_slow = dspx_series.ewm(span=21, adjust=False).mean()
        
        c_spot, c_f, c_s = corr_series.iloc[-1], corr_fast.iloc[-1], corr_slow.iloc[-1]
        d_spot, d_f, d_s = dspx_series.iloc[-1], dsp_fast.iloc[-1], dsp_slow.iloc[-1]
        
        corr_mean = corr_series.rolling(60).mean().iloc[-1]
        corr_std = corr_series.rolling(60).std().iloc[-1]
        dspx_mean = dspx_series.rolling(60).mean().iloc[-1]
        dspx_std = dspx_series.rolling(60).std().iloc[-1]
        
        c_z = (c_spot - corr_mean) / corr_std if corr_std > 0 else 0
        d_z = (d_spot - dspx_mean) / dspx_std if dspx_std > 0 else 0

        c_is_high = c_z > 1.0 or c_s > corr_q75.iloc[-1]
        c_is_low = c_z < -1.0 or c_s < corr_q25.iloc[-1]
        c_dead_cross = c_f < c_s
        c_golden_cross = c_f > c_s
        
        d_is_high = d_z > 1.0 or d_f > d_s
        d_is_low = d_z < -1.0 or d_f < d_s
        
        market_high = data['SPY'].iloc[-1] > data['SPY'].rolling(50).mean().iloc[-1]
        
        corr_risk_level = "数据风险"
        corr_risk_diag = "CBOE 数据断层，暂停象限判断。"
        corr_high_risk_active = False

        if corr_is_broken:
            corr_diag, disp_diag, combined_diag = "流断裂", "流断裂", "数据断层"
            corr_bottom_active = breadth_top_active = False
        else:
            if c_golden_cross: corr_diag = f"【相关性升温(Z:{c_z:.1f})】全市场同涨同跌共振加剧"
            else: corr_diag = f"【相关性退潮(Z:{c_z:.1f})】市场共振消退，逐步回归理性"
            
            if d_is_high: disp_diag = f"【离散度分化发散(Z:{d_z:.1f})】两极分化加剧，抱团失血效应显著"
            else: disp_diag = f"【离散度收敛(Z:{d_z:.1f})】板块轮动均衡，非极端撕裂期"

            corr_bottom_active = c_is_high and c_dead_cross and not d_is_high
            breadth_top_active = market_high and c_is_low and d_is_high
            
            if c_golden_cross and d_is_high:
                combined_diag = "⚡ 【象限 I: 双高危机】宏观剧震引发系统流动性冲击与内部结构剧烈撕裂并发（观望，严防无差别闪崩）"
                corr_risk_level = "高风险"
                corr_risk_diag = "相关性升温与离散度发散同步出现，说明指数层面与内部结构同时承压，容易从局部抱团扩散为系统波动。"
            elif c_golden_cross and not d_is_high:
                combined_diag = "🔥 【象限 II: 泥沙俱下】纯粹的同频无差别恐慌抛售，相关性极高（等待 CBOE 快慢线死叉即可抄底）"
                corr_risk_level = "中高风险"
                corr_risk_diag = "相关性升温但离散度未发散，更多是同频杀跌，需等相关性死叉后再从防守转抄底。"
            elif c_dead_cross and d_is_high:
                combined_diag = "🚨 【象限 III: 极致撕裂】大盘失真，资金极致抱团超级权重，掩护中小盘出货（触发终极广度逃顶线）"
                corr_risk_level = "极高风险" if market_high or c_is_low else "高风险"
                corr_risk_diag = "相关性退潮但离散度爆发，代表指数表面稳定、内部广度崩塌；若指数仍在高位，属于典型抱团掩护出货。"
            else:
                combined_diag = "⏳ 【象限 IV: 均衡收敛】常态低波运行，系统性风险真空期，个股特异性健康回归"
                if c_is_low and market_high:
                    corr_risk_level = "中风险"
                    corr_risk_diag = "相关性偏低且指数处于高位，市场缺少同涨支撑，但离散度尚未确认爆发。"
                elif d_is_low and not c_is_low:
                    corr_risk_level = "低风险"
                    corr_risk_diag = "离散度收敛，板块内部结构较均衡，系统性尾部风险暂未抬头。"
                else:
                    corr_risk_level = "中性"
                    corr_risk_diag = "相关性与离散度未形成极端共振，维持常规观察。"

            if breadth_top_active:
                corr_risk_level = "极高风险"
            corr_high_risk_active = corr_risk_level in ("高风险", "极高风险")
            breadth_top_active = breadth_top_active or corr_high_risk_active

        cboe_corr_text = f"相关性:{c_spot:.2f}(Z:{c_z:.1f})"
        cboe_disp_text = f"离散度:{d_spot:.2f}(Z:{d_z:.1f})"
        spy_rsp_ratio = latest['SPY'] / latest['RSP']
        
        df_hist = pd.DataFrame(index=data.index)
        df_hist['corr'] = corr_series
        df_hist['corr_fast'] = corr_fast
        df_hist['corr_slow'] = corr_slow
        df_hist['dspx'] = dspx_series
        df_hist['dsp_fast'] = dsp_fast
        df_hist['dsp_slow'] = dsp_slow
        df_hist['cta_shorts'] = cta_shorts_series
        df_hist['cta_longs'] = cta_longs_series
        
        return {
            "error": False,
            "cta_status": cta_status_text,
            "cboe_corr": cboe_corr_text,
            "cboe_disp": cboe_disp_text,
            "spy_rsp_ratio": round(spy_rsp_ratio, 4),
            "cta_bottom_active": cta_bottom_active,
            "cta_top_active": cta_top_active,
            "corr_bottom_active": corr_bottom_active,
            "breadth_top_active": breadth_top_active,
            "corr_is_broken": corr_is_broken,
            "corr_diag": corr_diag,          
            "disp_diag": disp_diag,      
            "combined_diag": combined_diag,  
            "corr_risk_level": corr_risk_level,
            "corr_risk_diag": corr_risk_diag,
            "df_hist": df_hist.tail(60),
            "data_source": yf_data.attrs.get("data_source", "live_market_sources"),
            "cache_age_hours": yf_data.attrs.get("cache_age_hours", 0.0),
            "fetched_at": market_data_timestamp(yf_data),
            "quote_mode": yf_data.attrs.get("quote_mode", "daily_fallback"),
            "quote_age_minutes": yf_data.attrs.get("quote_age_minutes"),
            "quote_sources": yf_data.attrs.get("quote_sources", "日线历史源")
        }
    except Exception as e:
        return {
            "error": True, "msg": str(e), 
            "cta_bottom_active": False, "cta_top_active": False,
            "corr_bottom_active": False, "breadth_top_active": False,
            "corr_diag": "诊断异常", "disp_diag": "诊断异常", "combined_diag": "诊断异常"
        }
        
def fetch_vxn_vix_data():
    try:
        df = pd.concat(
            {
                '^VXN': fetch_cboe_official_history('VXN'),
                '^VIX': fetch_cboe_official_history('VIX')
            },
            axis=1
        ).sort_index().ffill().dropna().tail(90)
        df = overlay_intraday_prices(
            df, ['^VXN', '^VIX'], require_all=True
        ).tail(90)
        
        if not df.empty:
            
            df['Spread'] = df['^VXN'] - df['^VIX']
            df['Ratio'] = df['^VXN'] / df['^VIX']
            
            df['Spread_Fast'] = df['Spread'].ewm(span=5, adjust=False).mean()
            df['Spread_Slow'] = df['Spread'].ewm(span=21, adjust=False).mean()
            
            current_spread = df['Spread'].iloc[-1]
            current_ratio = df['Ratio'].iloc[-1]
            fast_curr = df['Spread_Fast'].iloc[-1]
            slow_curr = df['Spread_Slow'].iloc[-1]
            
            fast_prev = df['Spread_Fast'].iloc[-2] if len(df) >= 2 else fast_curr
            slow_prev = df['Spread_Slow'].iloc[-2] if len(df) >= 2 else slow_curr
            vix_spot = df['^VIX'].iloc[-1]
            
            is_death_cross = (fast_prev >= slow_prev) and (fast_curr < slow_curr)
            is_golden_cross = (fast_prev <= slow_prev) and (fast_curr > slow_curr)
            had_high_panic = df['Spread'].tail(5).max() > 8.0
            
            bottom_active = is_death_cross and had_high_panic and (vix_spot < 35.0)
            volcano_active = (current_spread > 7.5 or current_ratio > 1.35) and (fast_curr > slow_curr)
            storm_prep_active = (current_spread < 3.0 or current_ratio < 1.10) and is_golden_cross
            
            top_active = volcano_active or storm_prep_active

            if is_death_cross:
                spread_diag = f"【高位死叉】快线({fast_curr:.2f})下穿慢线({slow_curr:.2f})。波动率溢价回落，恐慌出清。"
            elif is_golden_cross:
                spread_diag = f"【低位金叉】快线({fast_curr:.2f})上穿慢线({slow_curr:.2f})。波动率动能放大，警惕分化。"
            elif fast_curr < slow_curr:
                spread_diag = f"【动能收敛】快线运行于慢线下方，科技股溢价风险维持常态化修复。"
            else:
                spread_diag = f"【动能发散】快线运行于慢线上方，科技股情绪溢价处于风险积聚期。"
                
            if current_ratio > 1.35:
                ratio_diag = f"【极端过热】比率({current_ratio:.2f})突破1.35。纳指多头对冲严重踩踏，极度拥挤。"
            elif current_ratio < 1.10:
                ratio_diag = f"【过度自满】比率({current_ratio:.2f})跌破1.10。市场极度懈怠，隐性筑顶风险剧增。"
            else:
                ratio_diag = f"【常态均衡】比率({current_ratio:.2f})在健康区间，风格资产未出现单边撕裂。"

            if bottom_active:
                combined_diag = "🚀 【黄金右侧】科技股恐慌见顶！现货全面进场，严禁做空！"
            elif volcano_active:
                combined_diag = "🌋 【火山口】独立踩踏发散中！触发多头硬熔断，严禁接飞刀！"
            elif storm_prep_active:
                combined_diag = "🌀 【前哨预警】极度自满打破，低位动能金叉！开始战略减仓科技多头！"
            else:
                if current_ratio < 1.10:
                    combined_diag = "🟡 【隐性风险】极度自满，期权对冲完全懈怠，隐含被动洗盘风险。"
                else:
                    combined_diag = "🟢 【常态牛市】情绪均衡。多头 EV 模型正常运转，拥抱趋势。"
            
            return {
                "current_spread": round(current_spread, 2),
                "current_ratio": round(current_ratio, 2),
                "fast_curr": round(fast_curr, 2),
                "slow_curr": round(slow_curr, 2),
                "bottom_active": bottom_active,
                "top_active": top_active,
                "error": False,
                "combined_diag": combined_diag,
                "spread_diag": spread_diag,
                "ratio_diag": ratio_diag,
                "df_hist": df,
                "fetched_at": market_data_timestamp(df),
                "quote_mode": df.attrs.get("quote_mode", "daily_fallback"),
                "quote_age_minutes": df.attrs.get("quote_age_minutes"),
                "quote_sources": df.attrs.get("quote_sources", "CBOE EOD")
            }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False, "fetched_at": "异常断流"}
    return {"error": True, "msg": "No data", "bottom_active": False, "top_active": False, "fetched_at": "空数据"}

def fetch_macro_liquidity_overlay():
    try:
        frames = []
        source_notes = []
        macro_quote_meta = {}
        try:
            yahoo_data = fetch_yahoo_core_close()
            yahoo_data = overlay_intraday_prices(
                yahoo_data, ['HYG', 'LQD'], require_all=True
            )
            macro_quote_meta = {
                "quote_mode": yahoo_data.attrs.get("quote_mode", "daily_fallback"),
                "quote_age_minutes": yahoo_data.attrs.get("quote_age_minutes"),
                "quote_sources": yahoo_data.attrs.get("quote_sources", "日线历史源"),
                "fetched_at": market_data_timestamp(yahoo_data)
            }
            if macro_quote_meta["quote_mode"] != "daily_fallback":
                source_notes.append(
                    "HYG/LQD使用盘前/日内快照；MOVE/VVIX仍按官方盘后日线"
                )
            if yahoo_data.attrs.get("data_source") == "last_success_cache":
                source_notes.append(
                    f"Yahoo实时限流，使用最近成功缓存 "
                    f"({yahoo_data.attrs.get('cache_age_hours', '?')}小时前)"
                )
            yahoo_cols = [
                symbol for symbol in ['^MOVE', 'HYG', 'LQD']
                if symbol in yahoo_data.columns
            ]
            if yahoo_cols:
                frames.append(yahoo_data[yahoo_cols].tail(130))
        except Exception as exc:
            source_notes.append(f"Yahoo子集暂不可用: {exc}")

        vvix = fetch_cboe_official_history('VVIX')
        if not vvix.empty:
            frames.append(vvix.rename('^VVIX').to_frame().tail(130))

        raw = pd.concat(frames, axis=1).sort_index().ffill() if frames else pd.DataFrame()
        # MOVE/VVIX 的日线骨架仅在授权指数源确实给出 REAL-TIME 值时覆盖。
        # 无授权、无权限或休市时 overlay 会保持官方最近日线，不会伪造盘中更新。
        raw = overlay_intraday_prices(
            raw, ['^MOVE', '^VVIX'], require_all=False
        )
        realtime_vol_symbols = raw.attrs.get("intraday_symbols", [])
        if realtime_vol_symbols:
            source_notes.append(
                "授权波动率盘中快照: "
                + ", ".join(realtime_vol_symbols)
                + f"（{raw.attrs.get('quote_sources', '实时指数源')}）"
            )
            macro_quote_meta = {
                "quote_mode": raw.attrs.get("quote_mode", "daily_fallback"),
                "quote_age_minutes": raw.attrs.get("quote_age_minutes"),
                "quote_sources": raw.attrs.get("quote_sources", "实时指数源"),
                "fetched_at": market_data_timestamp(raw)
            }
        if raw.empty:
            raise Exception("宏观补充数据为空")

        available_cols = set(raw.columns)
        risk_points = 0
        opportunity_points = 0
        details = source_notes

        if '^MOVE' in available_cols and raw['^MOVE'].dropna().shape[0] >= 30:
            move = raw['^MOVE'].dropna()
            move_z = (move.iloc[-1] - move.tail(60).mean()) / move.tail(60).std() if move.tail(60).std() > 0 else 0
            if move_z >= 1.25:
                risk_points += 9
                details.append(f"MOVE债券波动Z:{move_z:.1f}，利率尾部风险抬升")
            elif move_z <= -1.0:
                opportunity_points += 4
                details.append(f"MOVE债券波动Z:{move_z:.1f}，利率冲击降温")

        if '^VVIX' in available_cols and raw['^VVIX'].dropna().shape[0] >= 30:
            vvix = raw['^VVIX'].dropna()
            vvix_z = (vvix.iloc[-1] - vvix.tail(60).mean()) / vvix.tail(60).std() if vvix.tail(60).std() > 0 else 0
            if vvix_z >= 1.25:
                risk_points += 8
                details.append(f"VVIX波动尾部Z:{vvix_z:.1f}，VIX凸性保护需求升温")
            elif vvix_z <= -1.0:
                opportunity_points += 3
                details.append(f"VVIX波动尾部Z:{vvix_z:.1f}，期权尾部恐慌缓和")

        if {'HYG', 'LQD'}.issubset(available_cols):
            credit_ratio = (raw['HYG'] / raw['LQD']).dropna()
            if credit_ratio.shape[0] >= 50:
                credit_fast = credit_ratio.ewm(span=5, adjust=False).mean()
                credit_slow = credit_ratio.ewm(span=21, adjust=False).mean()
                credit_z = (credit_ratio.iloc[-1] - credit_ratio.tail(60).mean()) / credit_ratio.tail(60).std() if credit_ratio.tail(60).std() > 0 else 0
                if credit_fast.iloc[-1] < credit_slow.iloc[-1] and credit_z < -0.7:
                    risk_points += 7
                    details.append(f"HYG/LQD信用比率走弱Z:{credit_z:.1f}，信用风险偏好退潮")
                elif credit_fast.iloc[-1] > credit_slow.iloc[-1] and credit_z > 0:
                    opportunity_points += 4
                    details.append(f"HYG/LQD信用比率修复Z:{credit_z:.1f}，信用风险偏好回暖")

        net_adjustment = risk_points - opportunity_points
        if net_adjustment >= 12:
            status = "🚨 宏观补充雷达：外部流动性明显恶化，综合策略需额外降风险。"
            level = "高风险"
        elif net_adjustment >= 6:
            status = "🟠 宏观补充雷达：债券/波动/信用有边际压力，进攻信号需要打折。"
            level = "中高风险"
        elif net_adjustment <= -5:
            status = "🟢 宏观补充雷达：外部流动性压力缓和，可提升抄底信号可信度。"
            level = "机会"
        else:
            status = "⚪ 宏观补充雷达：外部流动性中性，六开关主模型占主导。"
            level = "中性"

        if not details:
            details.append("可用补充指标不足，暂不修正主模型。")

        return {
            "error": False,
            "level": level,
            "status": status,
            "details": "；".join(details),
            "risk_points": risk_points,
            "opportunity_points": opportunity_points,
            "net_adjustment": net_adjustment,
            "fetched_at": macro_quote_meta.get(
                "fetched_at", market_data_timestamp(raw)
            ),
            "quote_mode": macro_quote_meta.get("quote_mode", "daily_fallback"),
            "quote_age_minutes": macro_quote_meta.get("quote_age_minutes"),
            "quote_sources": macro_quote_meta.get("quote_sources", "日线历史源")
        }
    except Exception as e:
        return {
            "error": True,
            "level": "数据缺失",
            "status": "宏观补充雷达数据抓取失败",
            "details": str(e),
            "risk_points": 0,
            "opportunity_points": 0,
            "net_adjustment": 0,
            "fetched_at": "异常断流"
        }
    
# -----------------------------------------------------------------------------
# 3. 业务决策逻辑组装与元数据解析
# -----------------------------------------------------------------------------
market_session = get_market_session()
alpaca_configured = bool(
    _get_config_value("ALPACA_API_KEY")
    and _get_config_value("ALPACA_API_SECRET")
)
paid_volatility_enabled = _paid_volatility_source_enabled()
auto_refresh_seconds = (
    MASSIVE_VOLATILITY_TTL_SECONDS
    if paid_volatility_enabled and market_session["session"] == "regular"
    else (INTRADAY_TTL_SECONDS if alpaca_configured else 15 * 60)
)
st.sidebar.markdown("### ⏱️ 盘前 / 日内行情")
st.sidebar.caption(
    f"当前：{market_session['label']} · "
    f"{market_session['now_et'].strftime('%Y-%m-%d %H:%M ET')}"
)
auto_refresh = st.sidebar.checkbox(
    f"交易时段每 {auto_refresh_seconds // 60} 分钟自动刷新",
    value=True,
    help="只在美东 04:00–20:00 的工作日启用；历史日线仍使用长缓存。"
)
if st.sidebar.button("🔄 立即刷新最新快照"):
    _fetch_intraday_snapshot_cached.clear()
    _fetch_massive_volatility_snapshot_cached.clear()

if market_session["active"] and auto_refresh:
    components.html(
        f"""
        <script>
        window.setTimeout(function() {{
            window.parent.location.reload();
        }}, {auto_refresh_seconds * 1000});
        </script>
        """,
        height=0
    )

vix_data = fetch_vix_data()
crypto_data = fetch_crypto_signals()
sm_data = fetch_squeezemetrics_data()
quant_data = calculate_quant_and_breadth_signals()
macro_data = fetch_macro_liquidity_overlay()

now_str = datetime.datetime.now(US_EASTERN).strftime('%Y-%m-%d %H:%M ET')
vxn_vix_data = fetch_vxn_vix_data()

intraday_snapshot = fetch_intraday_snapshot()
if market_session["active"]:
    if not intraday_snapshot.empty:
        latest_snapshot_at = intraday_snapshot["timestamp"].max()
        latest_snapshot_et = latest_snapshot_at.tz_convert(US_EASTERN)
        snapshot_age = max(
            0,
            (
                pd.Timestamp.now(tz="UTC") - latest_snapshot_at
            ).total_seconds() / 60
        )
        source_text = ", ".join(
            sorted(set(intraday_snapshot["source"].astype(str)))
        )
        st.sidebar.success(
            f"最新快照：{latest_snapshot_et.strftime('%H:%M ET')} "
            f"({snapshot_age:.0f} 分钟前)\n\n{source_text}"
        )
        provider_errors = intraday_snapshot.attrs.get("provider_errors", [])
        if provider_errors:
            with st.sidebar.expander("部分数据源诊断"):
                for error_text in provider_errors:
                    st.caption(error_text)
    else:
        st.sidebar.info(
            "盘中快照暂不可用；CBOE/Yahoo 最近有效日线仍作为模型基准运行。"
        )
        if INTRADAY_LAST_ERROR:
            with st.sidebar.expander("查看盘中快照诊断", expanded=False):
                st.code(INTRADAY_LAST_ERROR, language=None)
else:
    st.sidebar.info("当前休市，停止轮询并使用最近有效收盘数据。")

if not alpaca_configured:
    st.sidebar.caption(
        "未配置 Alpaca：ETF 使用 Yahoo 15分钟线，并将自动刷新降为15分钟以降低限流风险。"
    )
if paid_volatility_enabled:
    st.sidebar.caption(
        "已显式启用付费指数源：仅接受 REAL-TIME 波动率快照；盘中每60秒刷新。"
    )

def classify_sm_gex_dix_risk(gex_val, dix_val):
    gex_abs = abs(gex_val)
    gex_extreme = gex_abs >= 1_000_000_000
    gex_neutral = gex_abs < 250_000_000

    if gex_val < 0 and dix_val < 40.0:
        risk_level = "极高风险"
        diag = "🚨 【负Gamma放大器 ✖ DIX派发】做市商追跌对冲与暗池主力撤退共振，容易出现流动性断层、跳水和闪崩。"
        bottom_active, top_active = False, True
    elif gex_val < 0 and dix_val < 42.5:
        risk_level = "高风险"
        diag = "🚨 【负Gamma承压 ✖ DIX偏弱】盘面下跌会被对冲流进一步放大，暗池承接不足，反弹更像减仓窗口。"
        bottom_active, top_active = False, True
    elif gex_val >= 0 and dix_val < 40.0:
        risk_level = "高风险"
        diag = "🚨 【Gamma表面护盘 ✖ 暗池派发】指数可能被期权仓位暂时托住，但主力资金在暗处流出，属于高位钝刀出货。"
        bottom_active, top_active = False, True
    elif gex_val < 0 and dix_val >= 45.0:
        risk_level = "中高风险" if not gex_extreme else "高风险"
        diag = "🟠 【负Gamma波动 ✖ DIX吸筹】机构有承接但做市商仍是波动放大器，适合等待确认，不宜重仓追涨。"
        bottom_active, top_active = False, gex_extreme
    elif gex_val >= 0 and dix_val >= 45.0:
        risk_level = "低风险/机会"
        diag = "🟢 【正Gamma缓冲 ✖ DIX吸筹】做市商对冲提供安全垫，暗池主力同步承接，左侧筑底与趋势修复概率较高。"
        bottom_active, top_active = True, False
    elif gex_val >= 0 and 40.0 <= dix_val < 45.0:
        risk_level = "中性偏稳" if not gex_neutral else "中性"
        diag = "⚪ 【正Gamma缓冲 ✖ DIX中性】波动被压制但暗池没有强吸筹证据，适合常规仓位、等待方向选择。"
        bottom_active, top_active = False, False
    elif gex_val < 0 and 42.5 <= dix_val < 45.0:
        risk_level = "中风险"
        diag = "🟡 【负Gamma扰动 ✖ DIX中性】下跌仍可能被放大，但暗池并未明显派发，保持保护性止盈和仓位克制。"
        bottom_active, top_active = False, False
    else:
        risk_level = "中性"
        diag = "⚪ 【Gamma/DIX 均衡】没有形成明确的吸筹、派发或对冲放大共振，维持观察。"
        bottom_active, top_active = False, False

    if gex_extreme and gex_val < 0 and risk_level not in ("极高风险", "高风险"):
        risk_level = "高风险"
        diag += " 叠加 GEX 绝对值极端为负，任何价格破位都可能触发机械性追跌。"
        top_active = True

    return {
        "risk_level": risk_level,
        "diag": diag,
        "bottom_active": bottom_active,
        "top_active": top_active,
    }

def classify_vix_profile(vix_data):
    if vix_data.get("error"):
        return {"risk_level": "数据缺失", "opportunity_level": "无", "risk_score": 40, "opportunity_score": 0}
    ratio = vix_data.get("ratio", 1.1)
    vix = vix_data.get("vix", 18)
    if vix_data.get("top_active") and (ratio >= 1.25 or vix < 13.5):
        return {"risk_level": "极高风险", "opportunity_level": "无", "risk_score": 96, "opportunity_score": 0}
    if vix_data.get("top_active"):
        return {"risk_level": "高风险", "opportunity_level": "无", "risk_score": 82, "opportunity_score": 0}
    if vix_data.get("bottom_active") and vix >= 24:
        return {"risk_level": "中风险", "opportunity_level": "强抄底", "risk_score": 45, "opportunity_score": 90}
    if vix_data.get("bottom_active"):
        return {"risk_level": "中性", "opportunity_level": "抄底", "risk_score": 30, "opportunity_score": 76}
    if ratio <= 1.0 or vix >= 24:
        return {"risk_level": "中高风险", "opportunity_level": "观察", "risk_score": 66, "opportunity_score": 22}
    if ratio >= 1.15 or vix < 13.5:
        return {"risk_level": "中风险", "opportunity_level": "无", "risk_score": 50, "opportunity_score": 0}
    return {"risk_level": "低风险", "opportunity_level": "观察", "risk_score": 20, "opportunity_score": 25}

def classify_crypto_profile(crypto_data):
    if crypto_data.get("error"):
        return {"risk_level": "数据缺失", "opportunity_level": "无", "risk_score": 38, "opportunity_score": 0}
    diag = crypto_data.get("diag_status", "")
    if crypto_data.get("top_active") and "极度危险" in diag:
        return {"risk_level": "高风险", "opportunity_level": "无", "risk_score": 84, "opportunity_score": 0}
    if crypto_data.get("top_active"):
        return {"risk_level": "中高风险", "opportunity_level": "无", "risk_score": 68, "opportunity_score": 0}
    if crypto_data.get("bottom_active"):
        return {"risk_level": "中性", "opportunity_level": "抄底", "risk_score": 32, "opportunity_score": 76}
    if "轧空预警" in diag:
        return {"risk_level": "中风险", "opportunity_level": "机会", "risk_score": 46, "opportunity_score": 55}
    if "健康延续" in diag:
        return {"risk_level": "低风险", "opportunity_level": "观察", "risk_score": 22, "opportunity_score": 34}
    return {"risk_level": "中性", "opportunity_level": "观察", "risk_score": 32, "opportunity_score": 20}

def classify_cta_profile(quant_data):
    if quant_data.get("error"):
        return {"risk_level": "数据缺失", "opportunity_level": "无", "risk_score": 38, "opportunity_score": 0}
    if quant_data.get("cta_top_active"):
        return {"risk_level": "高风险", "opportunity_level": "无", "risk_score": 78, "opportunity_score": 0}
    if quant_data.get("cta_bottom_active"):
        return {"risk_level": "中风险", "opportunity_level": "抄底", "risk_score": 44, "opportunity_score": 78}
    if "分化" in quant_data.get("cta_status", ""):
        return {"risk_level": "中风险", "opportunity_level": "观察", "risk_score": 45, "opportunity_score": 20}
    return {"risk_level": "中性", "opportunity_level": "观察", "risk_score": 30, "opportunity_score": 25}

def classify_corr_profile(quant_data):
    if quant_data.get("error"):
        return {"risk_level": "数据缺失", "opportunity_level": "无", "risk_score": 40, "opportunity_score": 0}
    risk_level = quant_data.get("corr_risk_level", "中性")
    risk_score = level_to_score(risk_level, RISK_SCORE_MAP)
    if quant_data.get("corr_bottom_active"):
        return {"risk_level": "中风险", "opportunity_level": "抄底", "risk_score": min(risk_score, 46), "opportunity_score": 80}
    if risk_level in ("高风险", "极高风险"):
        return {"risk_level": risk_level, "opportunity_level": "无", "risk_score": risk_score, "opportunity_score": 0}
    if risk_level == "低风险":
        return {"risk_level": "低风险", "opportunity_level": "观察", "risk_score": 18, "opportunity_score": 28}
    return {"risk_level": risk_level, "opportunity_level": "观察", "risk_score": risk_score, "opportunity_score": 18}

def classify_vxn_profile(vxn_vix_data):
    if vxn_vix_data.get("error"):
        return {"risk_level": "数据缺失", "opportunity_level": "无", "risk_score": 38, "opportunity_score": 0}
    ratio = vxn_vix_data.get("current_ratio", 1.2)
    spread = vxn_vix_data.get("current_spread", 4.0)
    if vxn_vix_data.get("top_active") and (ratio > 1.35 or spread > 7.5):
        return {"risk_level": "极高风险", "opportunity_level": "无", "risk_score": 92, "opportunity_score": 0}
    if vxn_vix_data.get("top_active"):
        return {"risk_level": "高风险", "opportunity_level": "无", "risk_score": 78, "opportunity_score": 0}
    if vxn_vix_data.get("bottom_active"):
        return {"risk_level": "中性", "opportunity_level": "抄底", "risk_score": 32, "opportunity_score": 78}
    if ratio < 1.10 or spread < 3.0:
        return {"risk_level": "中风险", "opportunity_level": "无", "risk_score": 48, "opportunity_score": 0}
    return {"risk_level": "低风险", "opportunity_level": "观察", "risk_score": 24, "opportunity_score": 24}

if not sm_data["error"]:
    gex_val = sm_data.get('gex', 0)
    dix_val = sm_data.get('dix', 44.0)
    sm_risk = classify_sm_gex_dix_risk(gex_val, dix_val)
    
    sm_bottom_active = sm_risk["bottom_active"]
    sm_top_active = sm_risk["top_active"]
    
    if sm_data.get("is_mock", False):
        sm_status = "使用兜底数据 🟡"
    elif sm_top_active:
        sm_status = f"🚨 {sm_risk['risk_level']}：{sm_risk['diag']}"
    elif sm_bottom_active:
        sm_status = f"🟢 {sm_risk['risk_level']}：{sm_risk['diag']}"
    else:
        sm_status = f"⚪ {sm_risk['risk_level']}：{sm_risk['diag']}"
else:
    sm_bottom_active = False
    sm_top_active = False
    sm_status = "数据抓取失败 🔴"
    gex_val, dix_val = 0, 0.0
    sm_risk = {"risk_level": "数据缺失", "opportunity_level": "无", "risk_score": 40, "opportunity_score": 0}

vix_profile = classify_vix_profile(vix_data)
crypto_profile = classify_crypto_profile(crypto_data)
cta_profile = classify_cta_profile(quant_data)
corr_profile = classify_corr_profile(quant_data)
vxn_profile = classify_vxn_profile(vxn_vix_data)

def quote_freshness_note(data):
    if data.get("error"):
        return ""
    mode = data.get("quote_mode", "daily_fallback")
    if mode == "daily_fallback":
        return (
            "<br><span style='font-size:8pt;color:#d68910;'>"
            "⏳ 当前未取得有效盘前/日内快照，使用最近收盘日线。"
            "</span>"
        )
    age = data.get("quote_age_minutes")
    age_text = f"{age:.0f}分钟" if isinstance(age, (int, float)) else "未知时长"
    source = data.get("quote_sources", "盘前/日内源")
    return (
        "<br><span style='font-size:8pt;color:#1e8449;'>"
        f"● {get_market_session()['label']}快照 · {age_text}前 · {source}"
        "</span>"
    )

vix_freshness_note = quote_freshness_note(vix_data)
quant_freshness_note = quote_freshness_note(quant_data)
vxn_freshness_note = quote_freshness_note(vxn_vix_data)

quant_cache_note = ""
if quant_data.get("data_source") == "last_success_cache":
    quant_cache_note = (
        f"<br><span style='font-size:8pt;color:#d68910;'>"
        f"Yahoo 实时限流，当前使用 {quant_data.get('cache_age_hours', '?')} 小时前的成功缓存。"
        f"</span>"
    )

switches = [
    {
        "id": 1,
        "name": "做市商 Gamma & 暗池 DIX 联合资产开关",
        "rank": 2,
        "weight": 1.25,
        "cycle_key": "post_close",
        "core_position": "底层资金承接与做市商对冲方向",
        "importance": "核心权重：能识别暗池吸筹/派发与Gamma机械对冲的共振。",
        "risk_level": sm_risk["risk_level"],
        "opportunity_level": "低风险/机会" if sm_bottom_active else sm_risk.get("opportunity_level", "无"),
        "risk_score": sm_risk.get("risk_score", level_to_score(sm_risk["risk_level"], RISK_SCORE_MAP)),
        "opportunity_score": 68 if sm_bottom_active else sm_risk.get("opportunity_score", 0),
        "bottom_active": sm_bottom_active,
        "top_active": sm_top_active,
        "value": f"GEX: {gex_val:,} | DIX: {dix_val}%",
        "source": "SqueezeMetrics (暗池吸筹指数 & SPX期权对冲敞口)",
        "desc_bottom": "【低风险/机会】GEX为正建立行情安全垫，且 DIX>=45 显示暗池主力承接，属于正Gamma缓冲与机构吸筹共振。",
        "desc_top": "【高风险预警】① GEX<0 且 DIX<40 为极高风险；② GEX<0 且 DIX<42.5、或 GEX>=0 但 DIX<40 为高风险；③ GEX<-10亿 即使 DIX吸筹也按高风险处理。所有高风险/极高风险均触发红色预警。",
        "fetched_status": sm_status,
        "update_cycle": "每日更新 (美东盘后)",
        "last_updated": sm_data.get("fetched_at", now_str)
    },
    {
            "id": 2,
            "name": "VIX 期限结构与趋势动能雷达",
            "rank": 1,
            "weight": 1.35,
            "cycle_key": "hybrid",
            "core_position": "系统性波动压力与期限结构拐点",
            "importance": "最高权重：期限结构破位经常领先系统性风控，盘中敏感度最高。",
            "risk_level": vix_profile["risk_level"],
            "opportunity_level": vix_profile["opportunity_level"],
            "risk_score": vix_profile["risk_score"],
            "opportunity_score": vix_profile["opportunity_score"],
            "bottom_active": vix_data["bottom_active"] if not vix_data["error"] else False,
            "top_active": vix_data["top_active"] if not vix_data["error"] else False,
            "value": f"今日比率: {vix_data.get('ratio', 'N/A')} | EMA5/21状态: {'快线上穿/多头' if vix_data['bottom_active'] else '死叉/发散'} | VIX现货: {vix_data.get('vix', 'N/A')}",
            "source": "CBOE 日线基准 ✖ Yahoo 15分钟快照（可用才覆盖）",
            "desc_bottom": "【双向修复抄底标准】当隐含波动率比率向上收复突破 1.0 平衡线（摆脱远期深度倒挂状态），或者在低位倒挂修复带(<=1.05)确立微观动能均线金叉（EMA5 > EMA21）时激活。这标志着全市场非理性非对称抛售流动性枯竭，买盘筹码右侧转折确立，转入高胜率抄底期。",
            "desc_top": "【三维立体逃顶标准】满足以下任一核心条件立即拉响风控防御：①比率冲破 1.25 绝对贪婪上限，期权空头无防备极度拥挤；②比率跌破 1.0 平衡线，长短期期限结构倒挂、牛市基石全面动摇；③比率在高位自满警戒带(>=1.15)发生了 EMA5 下穿 EMA21 死叉，显示做多边际买盘已经枯竭见顶。",
            "fetched_status": "数据抓取失败 🔴" if vix_data["error"] else (
                f"<b>当下状态：</b>{vix_data.get('vix_diag_status')}<br>"
                f"<b>⚖️ 比率动能分项：</b>{vix_data.get('vix_ratio_diag')}<br>"
                f"<b>📊 现货波动分项：</b>{vix_data.get('vix_spot_diag')}"
                f"{vix_freshness_note}"
            ),
            "update_cycle": "CBOE日线基准；Yahoo盘中可用时覆盖，否则保持日线",
            "last_updated": vix_data.get("fetched_at", now_str)
        },
    {
        "id": 3,
        "name": "加密离岸高杠杆流动性前哨 (Price+OI+FR 矩阵)",
        "rank": 6,
        "weight": 0.72,
        "cycle_key": "hybrid",
        "core_position": "离岸杠杆情绪与风险偏好前哨",
        "importance": "辅助权重：对高Beta风险偏好敏感，但对美股核心资金面需打折处理。",
        "risk_level": crypto_profile["risk_level"],
        "opportunity_level": crypto_profile["opportunity_level"],
        "risk_score": crypto_profile["risk_score"],
        "opportunity_score": crypto_profile["opportunity_score"],
        "bottom_active": crypto_data["bottom_active"] if not crypto_data["error"] else False,
        "top_active": crypto_data["top_active"] if not crypto_data["error"] else False,
        "value": f"BTC现货: {crypto_data.get('btc_price', 'N/A')} ({crypto_data.get('price_trend', '')}) | OI: {crypto_data.get('oi', 'N/A')} ({crypto_data.get('oi_trend', '')}) | 费率: {crypto_data.get('funding_rate', 'N/A')}",
        "source": "OKX 官方 API (BTC K线 ✖ OI ✖ 资金费率)",
        "desc_bottom": "【缩量爆仓抄底】当 **价格下跌 + OI显著下降 + 费率转负**。代表做多杠杆被彻底清算，市场流动性恐慌见底，是高胜率左侧或右侧建仓点。",
        "desc_top": "【拥挤过载逃顶】触发两种情况立即防御：① **价格上涨 + OI上升 + 费率极高** (多头拥挤，极易被爆)；② **价格上涨 + OI下降** (缺乏新资金的假突破)。",
        "fetched_status": f"数据抓取失败 🔴 <br><span style='font-size:8pt;color:#e74c3c;'>异常原因: {crypto_data.get('msg', '未知断流')}</span>" if crypto_data["error"] else (
            f"<div style='background-color:#f4f6f7; padding:8px; border-radius:5px; margin-bottom:5px; font-weight:bold; color:#2c3e50;'>{crypto_data.get('diag_status')}</div>"
        ),
        "update_cycle": "日线级别清洗 ✖ 盘中实时快照",
        "last_updated": crypto_data.get("fetched_at", now_str)
    },
    {
        "id": 4,
        "name": "CTA 动量矩阵 (系统性抛压/买盘极值监测)",
        "rank": 5,
        "weight": 0.88,
        "cycle_key": "daily",
        "core_position": "趋势基金系统性买盘/卖盘边际位置",
        "importance": "中高权重：适合判断趋势资金是否过饱和或抛压耗尽。",
        "risk_level": cta_profile["risk_level"],
        "opportunity_level": cta_profile["opportunity_level"],
        "risk_score": cta_profile["risk_score"],
        "opportunity_score": cta_profile["opportunity_score"],
        "bottom_active": quant_data["cta_bottom_active"] if not quant_data["error"] else False,
        "top_active": quant_data["cta_top_active"] if not quant_data["error"] else False,
        "value": f"当前状态: {quant_data.get('cta_status', 'N/A')}",
        "source": "历史日线 ✖ Alpaca/Yahoo ETF 快照 ✖ 1M/3M/6M 动量",
        "desc_bottom": "主跌浪贯穿多周期均线且负乖离达极限。量化 CTA 的约跟空抛压面临彻底耗尽。",
        "desc_top": "趋势基金无脑买入的边际力量全面满仓，正乖离达极限，市场缺乏后续增量买家。",
        "fetched_status": "数据抓取失败 🔴" if quant_data["error"] else (
            (
                "🚨 警报：系统性买盘进入衰竭点" if quant_data["cta_top_active"] else (
                    "🟢 激活：系统性空头抛压触底耗尽" if quant_data["cta_bottom_active"] else f"⚪ 运行中：{quant_data.get('cta_status')}"
                )
            ) + quant_cache_note + quant_freshness_note
        ),
        "update_cycle": "盘前/盘中10–15分钟ETF快照",
        "last_updated": quant_data.get("fetched_at", now_str)
    },
    {
        "id": 5,
        "name": "全局隐含相关性拐点与离散度爆发矩阵 (全象限版)",
        "rank": 3,
        "weight": 1.18,
        "cycle_key": "daily",
        "core_position": "市场广度、抱团程度与内部结构撕裂",
        "importance": "核心权重：能捕捉指数表面稳定但内部广度塌陷的风险。",
        "risk_level": corr_profile["risk_level"],
        "opportunity_level": corr_profile["opportunity_level"],
        "risk_score": corr_profile["risk_score"],
        "opportunity_score": corr_profile["opportunity_score"],
        "bottom_active": quant_data["corr_bottom_active"] if not quant_data["error"] else False,
        "top_active": quant_data["breadth_top_active"] if not quant_data["error"] else False,
        "value": f"{quant_data.get('cboe_corr', 'N/A')} | {quant_data.get('cboe_disp', 'N/A')}",
        "source": "CBOE COR1M/DSPX 日线 ✖ SPY/RSP 日内快照",
        "desc_bottom": "【抄底激活：恐慌死叉✖撕裂收敛】当相关性极值冲顶后向下死叉确立（恐慌抛售衰退），且离散度未出现背离爆发时激活。此时大盘无差别抛压清空，回归估值红利期。",
        "desc_top": "【风险等级预警】象限 I（相关性升温+离散度发散）按高风险预警；象限 III（相关性退潮+离散度发散）按高风险，若指数高位或相关性极低升级为极高风险；所有高风险/极高风险均触发红色预警。",
        "fetched_status": "数据抓取失败 🔴" if quant_data["error"] else (
            f"<div style='background-color:#f4f6f7; padding:8px; border-radius:5px; margin-bottom:5px; font-weight:bold; color:#d35400;'>{quant_data.get('combined_diag', '无信息')}</div>"
            f"<b>🧯 风险等级：</b>{quant_data.get('corr_risk_level', '无信息')} - {quant_data.get('corr_risk_diag', '无信息')}<br>"
            f"<b>📊 相关性微观动能：</b>{quant_data.get('corr_diag', '无信息')}<br>"
            f"<b>📉 离散度微观动能：</b>{quant_data.get('disp_diag', '无信息')}"
            f"{quant_cache_note}"
            f"{quant_freshness_note}"
        ),
        "update_cycle": "CBOE日线 + ETF 10–15分钟快照",
        "last_updated": quant_data.get("fetched_at", now_str)
    },
    {
        "id": 6,
        "name": "VXN-VIX 科技股雷达",
        "rank": 4,
        "weight": 1.05,
            "cycle_key": "hybrid",
        "core_position": "科技股相对波动溢价与纳指踩踏风险",
        "importance": "中高权重：对纳指/AI/高Beta科技仓位的即时风控很敏感。",
        "risk_level": vxn_profile["risk_level"],
        "opportunity_level": vxn_profile["opportunity_level"],
        "risk_score": vxn_profile["risk_score"],
        "opportunity_score": vxn_profile["opportunity_score"],
        "bottom_active": vxn_vix_data["bottom_active"] if not vxn_vix_data["error"] else False,
        "top_active": vxn_vix_data["top_active"] if not vxn_vix_data["error"] else False,
        "value": f"Spread: {vxn_vix_data.get('current_spread', 'N/A')} | Ratio: {vxn_vix_data.get('current_ratio', 'N/A')} | 熔断风控实时检测",
        "source": "CBOE 日线基准 ✖ Yahoo 15分钟 VXN/VIX 快照（可用才覆盖）",
        "desc_bottom": "【右侧出击】当剪刀差自高位（>8.0）回落，且微观动能死叉（EMA5 < EMA21）时激活。此时非对称踩踏结束，IV Crush 来临，是高弹性科技股胜率极高的反转买点。",
        "desc_top": "【双重风控防御】① 火山口（单边踩踏）：极高位金叉发散，无条件熔断科技股多头；② 暴风雨前夜（隐性筑顶）：低位自满区间突发金叉，主力悄然买入 Put，需立刻收紧止盈或做空保护。",
        "fetched_status": "数据抓取失败 🔴" if vxn_vix_data["error"] else (
            f"<div style='background-color:#f4f6f7; padding:8px; border-radius:5px; margin-bottom:5px; font-weight:bold; color:#d35400;'>{vxn_vix_data.get('combined_diag', '无信息')}</div>"
            f"<b>📊 微观动能：</b>{vxn_vix_data.get('spread_diag', '无信息')}<br>"
            f"<b>📉 情绪象限：</b>{vxn_vix_data.get('ratio_diag', '无信息')}"
            f"{vxn_freshness_note}"
        ),
        "update_cycle": "CBOE日线基准；Yahoo盘中可用时覆盖，否则保持日线",
        "last_updated": vxn_vix_data.get("fetched_at", now_str)
    }
]

switches = sorted([enrich_switch(s) for s in switches], key=lambda x: x["rank"])

bottom_score = sum([1 for s in switches if s["bottom_active"]])
top_score = sum([1 for s in switches if s["top_active"]])
neutral_score = len(switches) - bottom_score - top_score
total_effective_weight = sum([s["effective_weight"] for s in switches])
weighted_risk_score = sum([s["risk_score"] * s["effective_weight"] for s in switches]) / total_effective_weight
weighted_opportunity_score = sum([s["opportunity_score"] * s["effective_weight"] for s in switches]) / total_effective_weight
macro_adjustment = macro_data.get("net_adjustment", 0)
net_risk_score = weighted_risk_score - weighted_opportunity_score + macro_adjustment
model_score_history = record_model_score_snapshot(
    weighted_risk_score,
    weighted_opportunity_score,
    macro_adjustment,
    net_risk_score
)

high_risk_names = [f"#{s['rank']} {s['name']}({s['risk_level']})" for s in switches if s["risk_score"] >= 72]
opportunity_names = [f"#{s['rank']} {s['name']}({s['opportunity_level']})" for s in switches if s["opportunity_score"] >= 64 and s["risk_score"] < 72]
ranking_line = " > ".join([f"{s['rank']}.{s['name'].split(' ')[0]}(权重{s['weight']:.2f})" for s in switches])

if net_risk_score >= 38 or weighted_risk_score >= 72:
    status_color = "red"
    action_title = "🚨 【红色防御：加权风险占优，进入系统性降杠杆模式】"
    action_text = (
        f"<b>加权诊断</b>：风险分 <b>{weighted_risk_score:.1f}</b> / 机会分 <b>{weighted_opportunity_score:.1f}</b> / "
        f"宏观修正 <b>{macro_adjustment:+.1f}</b> / 净风险 <b>{net_risk_score:.1f}</b>。<br>"
        f"<b>主导风险</b>：{'；'.join(high_risk_names[:3]) if high_risk_names else '风险来自多个中等级别开关叠加'}。<br>"
        "策略：净多头降到防御仓位，优先处理高Beta、弱广度、弱现金流标的；新开仓只允许小仓试错，所有盈利仓提高保护性止盈。"
    )
elif net_risk_score <= -22 and weighted_opportunity_score >= 56:
    status_color = "green"
    action_title = "🚀 【绿色进攻：加权机会占优，允许分批抄底/加仓】"
    action_text = (
        f"<b>加权诊断</b>：风险分 <b>{weighted_risk_score:.1f}</b> / 机会分 <b>{weighted_opportunity_score:.1f}</b> / "
        f"宏观修正 <b>{macro_adjustment:+.1f}</b> / 净风险 <b>{net_risk_score:.1f}</b>。<br>"
        f"<b>主导机会</b>：{'；'.join(opportunity_names[:3]) if opportunity_names else '机会来自多个开关温和修复'}。<br>"
        "策略：允许分批抄底或提高核心仓位，但仍需避开财务/趋势双弱个股；若盘中实时开关重新转红，立即暂停加仓。"
    )
elif net_risk_score >= 16:
    status_color = "orange"
    action_title = "🟠 【橙色谨慎：风险边际占优，进入轻防御与观察模式】"
    action_text = (
        f"<b>加权诊断</b>：风险分 <b>{weighted_risk_score:.1f}</b> / 机会分 <b>{weighted_opportunity_score:.1f}</b> / "
        f"宏观修正 <b>{macro_adjustment:+.1f}</b> / 净风险 <b>{net_risk_score:.1f}</b>。<br>"
        "策略：不追涨、不加杠杆，保留核心仓但减少边缘仓；等待 VIX/VXN 盘中开关或 GEX/DIX 日更确认方向。"
    )
elif net_risk_score <= -8:
    status_color = "green"
    action_title = "🟢 【绿色修复：机会边际占优，但尚未满仓共振】"
    action_text = (
        f"<b>加权诊断</b>：风险分 <b>{weighted_risk_score:.1f}</b> / 机会分 <b>{weighted_opportunity_score:.1f}</b> / "
        f"宏观修正 <b>{macro_adjustment:+.1f}</b> / 净风险 <b>{net_risk_score:.1f}</b>。<br>"
        "策略：可以小步增加高质量核心资产或做空波动后的反弹修复，但仓位节奏必须分批，等待更多日级别开关确认。"
    )
else:
    status_color = "orange"
    action_title = "⏳ 【黄色均衡：多空证据交织，执行中性仓位与分层观察】"
    action_text = (
        f"<b>加权诊断</b>：风险分 <b>{weighted_risk_score:.1f}</b> / 机会分 <b>{weighted_opportunity_score:.1f}</b> / "
        f"宏观修正 <b>{macro_adjustment:+.1f}</b> / 净风险 <b>{net_risk_score:.1f}</b>。<br>"
        f"当前红灯:<b>{top_score}</b> / 绿灯:<b>{bottom_score}</b> / 中性:<b>{neutral_score}</b>。"
        "策略：维持均衡仓位，重点跟踪排名靠前的 VIX期限、GEX/DIX、相关性离散度三大核心开关。"
    )

# -----------------------------------------------------------------------------
# 4. Streamlit UI 界面绘制
# -----------------------------------------------------------------------------
st.title("🛡️ Sentinel 2.0 核心决策系统：大盘底层资金双向雷达")
st.subheader(
    f"看板渲染时钟: "
    f"{datetime.datetime.now(US_EASTERN).strftime('%Y-%m-%d %H:%M:%S ET')}"
)

st.markdown(f"""
<div style="padding:15px; border-radius:8px; border-left: 6px solid {status_color}; background-color:#fafafa; margin-bottom:20px;">
    <h4 style="color:{status_color}; margin:0 0 10px 0;">{action_title}</h4>
    <p style="font-size:11pt; line-height:1.6; color:#333;">{action_text}</p>
    <p style="font-size:9pt; line-height:1.45; color:#555; margin:8px 0 0 0;">
        <b>重要性排序:</b> {ranking_line}<br>
        <b>宏观补充:</b> {macro_data.get('status', '无信息')} {macro_data.get('details', '')}<br>
        <b>宏观行情时间:</b> {macro_data.get('fetched_at', '未知')} ·
        {macro_data.get('quote_sources', '日线历史源')}
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📈 模型日线净风险趋势")
if not model_score_history.empty:
    score_plot = model_score_history.copy()
    score_plot["timestamp"] = pd.to_datetime(
        score_plot["timestamp"], errors="coerce", utc=True
    ).dt.tz_convert(US_EASTERN)
    score_plot = score_plot.dropna(subset=["timestamp"]).sort_values("timestamp")
    fig_scores = go.Figure()
    marker_colors = [
        "#c0392b" if value >= 38 else
        "#e67e22" if value >= 16 else
        "#95a5a6" if value > -8 else
        "#27ae60"
        for value in score_plot["net_risk"]
    ]
    fig_scores.add_hrect(y0=38, y1=100, line_width=0, fillcolor="#e74c3c", opacity=0.08)
    fig_scores.add_hrect(y0=16, y1=38, line_width=0, fillcolor="#f39c12", opacity=0.08)
    fig_scores.add_hrect(y0=-8, y1=16, line_width=0, fillcolor="#95a5a6", opacity=0.05)
    fig_scores.add_hrect(y0=-100, y1=-8, line_width=0, fillcolor="#2ecc71", opacity=0.08)
    fig_scores.add_trace(go.Scatter(
        x=score_plot["timestamp"],
        y=score_plot["net_risk"],
        customdata=score_plot[[
            "weighted_risk", "weighted_opportunity", "macro_adjustment"
        ]].to_numpy(),
        mode="lines+markers",
        name="净风险",
        line=dict(color="#2c3e50", width=3),
        marker=dict(size=8, color=marker_colors, line=dict(color="#ffffff", width=1)),
        hovertemplate=(
            "%{x|%Y-%m-%d}<br>"
            "净风险: %{y:.1f}<br>"
            "加权风险: %{customdata[0]:.1f}<br>"
            "加权机会: %{customdata[1]:.1f}<br>"
            "宏观修正: %{customdata[2]:+.1f}<extra></extra>"
        )
    ))
    fig_scores.add_hline(y=38, line_dash="dot", line_color="#c0392b", annotation_text="红色防御", annotation_position="top left")
    fig_scores.add_hline(y=16, line_dash="dot", line_color="#e67e22", annotation_text="橙色谨慎", annotation_position="top left")
    fig_scores.add_hline(y=-8, line_dash="dot", line_color="#27ae60", annotation_text="绿色修复", annotation_position="bottom left")
    fig_scores.add_hline(y=0, line_dash="dash", line_color="#7f8c8d", annotation_text="中性线", annotation_position="bottom right")
    score_y_min = min(-35, float(np.floor(score_plot["net_risk"].min() - 8)))
    score_y_max = max(85, float(np.ceil(score_plot["net_risk"].max() + 8)))
    fig_scores.update_layout(
        template="plotly_white",
        height=330,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        yaxis=dict(title="净风险分数", range=[score_y_min, score_y_max]),
        xaxis=dict(title="交易日（ET）", rangeslider=dict(visible=False)),
        title="日线净风险：点色对应当前风险区间；悬停查看风险、机会与宏观拆分"
    )
    st.plotly_chart(fig_scores, use_container_width=True)
    st.caption(
        f"每日仅保留最新模型值；当前实例最多保留最近 {MODEL_SCORE_HISTORY_DAYS} 天。"
    )
else:
    st.info("模型日线历史将在本次运行完成后开始积累。")

st.markdown("### 🔌 双向资金逻辑开关实时追踪")
cols = st.columns(3)

for i, s in enumerate(switches):
    with cols[i % 3]:
        if s["top_active"]:
            box_class = "status-top-active"
            badge_html = f"<span class='badge-top'>🚨 {s['badge_label']}</span>"
        elif s["bottom_active"]:
            box_class = "status-bottom-active"
            badge_html = f"<span class='badge-bottom'>🟢 {s['badge_label']}</span>"
        else:
            box_class = "status-neutral"
            badge_html = f"<span class='badge-info'>⚪ {s['badge_label']}</span>"
        
        metadata_line = f'<div style="margin-top: 10px; padding-top: 6px; border-top: 1px dashed #e0e0e0; font-size: 8pt; color: #7f8c8d;"><span style="float: left;">⏱️ {s.get("update_cycle", "未知")}</span><span style="float: right; font-family: monospace;">📅 {s.get("last_updated", "实时")}</span><div style="clear: both;"></div></div>'
        
        # 【防御拦截逻辑】：将文本中的 < 和 > 转义为安全的 HTML 实体 &lt; 和 &gt;，避免吞噬后续组件
        safe_desc_bottom = s['desc_bottom'].replace('<', '&lt;').replace('>', '&gt;')
        safe_desc_top = s['desc_top'].replace('<', '&lt;').replace('>', '&gt;')
            
        st.markdown(f"""
        <div class="metric-box {box_class}">
            <div class="switch-head">
                <span class="switch-title">开关 {s['id']}: {s['name']}</span>
                {badge_html}
            </div>
            <hr style="margin: 8px 0; border: 0; border-top: 1px solid #eee;">
            <p class="switch-value"><b>核心定位:</b> <span style="color:#2c3e50; font-weight:bold;">{s['core_position']}</span></p>
            <p class="switch-value"><b>当前数值:</b> <span style="font-family: monospace; color:#2980b9; font-weight:bold;">{s['value']}</span></p>
            <div class="switch-meta-grid">
                <div><b>风险等级:</b> {s['risk_level']} ({s['risk_score']})</div>
                <div><b>机会等级:</b> {s['opportunity_level']} ({s['opportunity_score']})</div>
                <div><b>重要性:</b> #{s['rank']} / 权重 {s['weight']:.2f}</div>
                <div><b>周期:</b> {s['cycle_label']} / 影响 {s['cycle_weight']:.2f}</div>
            </div>
            <div class="switch-status"><b>📡 数据状态:</b> <span>{s['fetched_status']}</span></div>
            <div class="switch-strategy"><b>策略动作:</b> {s['strategy']}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔘 点击展开：多空防守边界逻辑", expanded=False):
            st.markdown(f"""
            <div class="switch-boundary-panel">
                <p><b>📈 多头见底边界:</b> <span style="color:#27ae60;">{safe_desc_bottom}</span></p>
                <p><b>📉 空头防守边界:</b> <span style="color:#c0392b;">{safe_desc_top}</span></p>
                <p><b>⚖️ 重要性说明:</b> {s['importance']}</p>
                <p><b>⏱️ 周期影响:</b> {s['cycle_note']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="switch-footer">
            🧭 数据来源: {s['source']}
            {metadata_line}
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. 纳指走势雷达引擎
# -----------------------------------------------------------------------------
st.markdown("### 🗺️ 纳指 100 (NDX) 承接区间与走势雷达引擎")

def fetch_ndx_chart_data():
    try:
        close = fetch_yahoo_core_close()
        if '^NDX' not in close.columns:
            return pd.DataFrame()
        close = overlay_intraday_prices(close, ['^NDX'])
        return (
            close[['^NDX']]
            .dropna()
            .tail(66)
            .rename(columns={'^NDX': 'Close'})
        )
    except Exception:
        return pd.DataFrame()

ndx_data = fetch_ndx_chart_data()
if not ndx_data.empty:
    fig_ndx = go.Figure()
    latest_ndx_close = float(ndx_data['Close'].iloc[-1])
    ndx_time_label = market_data_timestamp(ndx_data)
    
    fig_ndx.add_trace(go.Scatter(
        x=ndx_data.index, y=ndx_data['Close'], mode='lines', name='NDX 实际走势曲线', line=dict(color='#2980b9', width=2.5)
    ))
    
    fig_ndx.add_hline(
        y=latest_ndx_close, line_dash="solid", line_color="#2c3e50", 
        annotation_text=f"最新有效位 ({latest_ndx_close:,.2f}) · {ndx_time_label}",
        annotation_position="top right"
    )
    
    fig_ndx.add_hline(y=28500, line_dash="dash", line_color="#e74c3c", annotation_text="CTA 二次抛售加速位 (28,500)", annotation_position="bottom right")
    fig_ndx.add_hline(y=26500, line_dash="dash", line_color="#c0392b", annotation_text="极端下影/二次冲洗 (26,500)", annotation_position="bottom right")
    
    fig_ndx.add_hrect(
        y0=27200, y1=28000, line_width=0, fillcolor="#2ecc71", opacity=0.15,
        annotation_text="核心承接区 (27,200 - 28,000)", annotation_position="inside top right"
    )

    data_min = float(ndx_data['Close'].min())
    data_max = float(ndx_data['Close'].max())
    y_range_min = data_min * 0.97
    y_range_max = data_max * 1.03
    
    if 26500 >= data_min * 0.88 and 26500 <= data_max * 1.12:
        y_range_min = min(y_range_min, 26500 * 0.99)
    if 28500 >= data_min * 0.88 and 28500 <= data_max * 1.12:
        y_range_max = max(y_range_max, 28500 * 1.01)

    fig_ndx.update_layout(
        title="Nasdaq 100 (^NDX) 阶梯支撑与洗盘推演 (智能自适应缩放)",
        template="plotly_white",
        yaxis=dict(title="NDX Index Points", range=[y_range_min, y_range_max], autorange=False, tickformat=",.0f"),
        xaxis_rangeslider_visible=False, height=500, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_ndx, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. 近期日线级别定量监控图表选项卡
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📡 资金波段逻辑追踪：近期日线级别定量监控图表")

all_tabs = st.tabs([
    "TAB 1: 做市商 & 暗池", 
    "TAB 2: VIX 期限结构", 
    "TAB 3: 离岸高杠杆", 
    "TAB 4: CTA 动量矩阵", 
    "TAB 5: 相关性与离散度",
    "TAB 6: VXN-VIX 科技剪刀差"
])

tab1 = all_tabs[0]
tab2 = all_tabs[1]
tab3 = all_tabs[2]
tab4 = all_tabs[3]
tab5 = all_tabs[4]
tab6 = all_tabs[5]

# --- TAB 1 ---
with tab1:
    if not sm_data["error"] and "df" in sm_data:
        plot_df = sm_data["df"]
        fig_sm = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sm.add_trace(
            go.Scatter(x=plot_df['date'], y=plot_df['dix'], name="暗池 DIX (%)", line=dict(color="#3498db", width=2)),
            secondary_y=False,
        )
        fig_sm.add_trace(
            go.Scatter(x=plot_df['date'], y=plot_df['gex'], name="做市商 GEX 净敞口", line=dict(color="#e74c3c", width=1.5, dash='dot')),
            secondary_y=True,
        )
        fig_sm.update_layout(title_text="DIX 与做市商 GEX 双向变动曲线", template="plotly_white", height=400)
        fig_sm.update_yaxes(title_text="<b>DIX 比例</b>", secondary_y=False)
        fig_sm.update_yaxes(title_text="<b>Gamma 敞口绝对值</b>", secondary_y=True)
        st.plotly_chart(fig_sm, use_container_width=True)
    else:
        st.warning("数据不可用。")

# --- TAB 2 ---
with tab2:
    if not vix_data["error"] and "df" in vix_data:
        v_df = vix_data["df"]
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            fig_vix_spot = go.Figure()
            fig_vix_spot.add_trace(go.Scatter(x=v_df.index, y=v_df['^VIX'], name="VIX 现货指数", line=dict(color="#e67e22", width=2)))
            fig_vix_spot.add_hline(y=12.0, line_dash="dash", line_color="#c0392b", annotation_text="自满安全线 (12.0)")
            fig_vix_spot.update_layout(title_text="图表 A: VIX 现货恐慌指数趋势", template="plotly_white", height=400)
            st.plotly_chart(fig_vix_spot, use_container_width=True)
        with v_col2:
            fig_vix_ratio = go.Figure()
            
            # 绘制真实计算期限比率基线
            fig_vix_ratio.add_trace(go.Scatter(
                x=v_df.index, y=v_df['Ratio'], 
                name="真实期限比率 (VIX3M / VIX)", 
                line=dict(color="#bdc3c7", width=1.2, dash='solid')
            ))
            
            # 【核心补充】引入 EMA5 微观快线
            if 'Ratio_Fast' in v_df.columns:
                fig_vix_ratio.add_trace(go.Scatter(
                    x=v_df.index, y=v_df['Ratio_Fast'], 
                    name="EMA5 (微观脉冲快线)", 
                    line=dict(color="#e74c3c", width=2.2)
                ))
                
            # 【核心补充】引入 EMA21 趋势慢线
            if 'Ratio_Slow' in v_df.columns:
                fig_vix_ratio.add_trace(go.Scatter(
                    x=v_df.index, y=v_df['Ratio_Slow'], 
                    name="EMA21 (多空大趋势线)", 
                    line=dict(color="#2c3e50", width=2.2)
                ))
            
            # 绘制更精准的区间分界线锚点
            fig_vix_ratio.add_hline(y=1.0, line_dash="dash", line_color="#2ecc71", annotation_text="Contango 恐慌修复平衡线 (1.0)", annotation_position="top left")
            fig_vix_ratio.add_hline(y=1.25, line_dash="dash", line_color="#e74c3c", annotation_text="极限自满防御高压线 (1.25)", annotation_position="bottom left")
            
            # 新增一条 1.15 的自满预警警戒中线，方便前瞻性减仓决策
            fig_vix_ratio.add_hline(y=1.15, line_dash="dot", line_color="#f39c12", annotation_text="高位自满警戒线 (1.15)")
            
            fig_vix_ratio.update_layout(
                title_text="图表 B: VIX3M / VIX 期限结构动能雷达 (均线交叉 ✖ 区间风控决策模型)", 
                template="plotly_white", 
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_vix_ratio, use_container_width=True)

# --- TAB 3 ---
with tab3:
    if not crypto_data.get("error", True) and crypto_data.get("hist_df") is not None:
        c_df = crypto_data["hist_df"]
        
        # 构建主副 Y 轴双轴图表
        fig_crypto = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 主 Y 轴：全网合约持仓量 (OI) 面面积图
        fig_crypto.add_trace(
            go.Scatter(
                x=c_df.index, y=c_df['oi'], 
                name="全网持仓量 (BTC)", 
                line=dict(color="#3498db", width=2, shape='spline'),
                fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.15)'
            ),
            secondary_y=False,
        )
        
        # 副 Y 轴：资金费率虚线图
        fig_crypto.add_trace(
            go.Scatter(
                x=c_df.index, y=c_df['funding_rate'], 
                name="日均资金费率 (%)", 
                line=dict(color="#f1c40f", width=2.5, dash='dot')
            ),
            secondary_y=True,
        )
        
        # 添加水平参照线
        fig_crypto.add_hline(y=0.0, secondary_y=True, line_dash="solid", line_color="#7f8c8d", opacity=0.6)
        fig_crypto.add_hline(y=0.025, secondary_y=True, line_dash="dash", line_color="#e74c3c", annotation_text="多头极端过热线 (0.025%)")
        
        fig_crypto.update_layout(
            title_text="加密离岸雷达：BTC 持仓规模 (OKX) 与日均资金费率同步校验", 
            template="plotly_white", 
            height=400,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig_crypto.update_yaxes(title_text="<b>合约持仓量 (BTC)</b>", secondary_y=False)
        fig_crypto.update_yaxes(title_text="<b>日均资金费率 (%)</b>", secondary_y=True)
        
        st.plotly_chart(fig_crypto, use_container_width=True)
    else:
        # 把底层抓取的异常信息暴露在 Tab 面板里
        st.warning(f"⚠️ 离岸高杠杆图表渲染终止。底层数据异常：{crypto_data.get('msg', '未知环境异常')}")
    
# --- TAB 4 ---
with tab4:
    if not quant_data["error"] and "df_hist" in quant_data:
        h_df = quant_data["df_hist"]
        fig_cta = go.Figure()
        fig_cta.add_trace(go.Scatter(x=h_df.index, y=h_df['cta_shorts'], name="系统性空头得分", line=dict(color="#e74c3c", width=2.5)))
        fig_cta.add_trace(go.Scatter(x=h_df.index, y=h_df['cta_longs'], name="系统性多头得分", line=dict(color="#2ecc71", width=2.5)))
        fig_cta.add_hline(y=2, line_dash="dash", line_color="#34495e", annotation_text="极值激活线 (2)")
        fig_cta.update_layout(title_text="CTA 量化追踪：多/空头趋势耗尽历史得分", template="plotly_white", height=400)
        st.plotly_chart(fig_cta, use_container_width=True)

# --- TAB 5 ---
with tab5:
    if not quant_data["error"] and "df_hist" in quant_data:
        h_df = quant_data["df_hist"]
        c6_col1, c6_col2 = st.columns(2)
        
        with c6_col1:
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(x=h_df.index, y=h_df['corr'], name="真实值", line=dict(color="#bdc3c7", width=1)))
            fig_corr.add_trace(go.Scatter(x=h_df.index, y=h_df['corr_fast'], name="EMA5 (快线)", line=dict(color="#e74c3c", width=2)))
            fig_corr.add_trace(go.Scatter(x=h_df.index, y=h_df['corr_slow'], name="EMA21 (慢线)", line=dict(color="#2c3e50", width=2)))
            fig_corr.update_layout(title_text="CBOE COR1M 相关性快慢线 (死叉形成释放见底信号)", template="plotly_white", height=380)
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with c6_col2:
            fig_disp = go.Figure()
            fig_disp.add_trace(go.Scatter(x=h_df.index, y=h_df['dspx'], name="真实值", line=dict(color="#bdc3c7", width=1)))
            fig_disp.add_trace(go.Scatter(x=h_df.index, y=h_df['dsp_fast'], name="EMA5 (快线)", line=dict(color="#2ecc71", width=2)))
            fig_disp.add_trace(go.Scatter(x=h_df.index, y=h_df['dsp_slow'], name="EMA21 (慢线)", line=dict(color="#34495e", width=2)))
            fig_disp.update_layout(title_text="CBOE DSPX 离散度快慢线 (高位金叉发散警惕拉巨头出货)", template="plotly_white", height=380)
            st.plotly_chart(fig_disp, use_container_width=True)

# --- TAB 6 ---
with tab6:
    if not vxn_vix_data["error"] and "df_hist" in vxn_vix_data:
        vx_df = vxn_vix_data["df_hist"]
        c7_col1, c7_col2 = st.columns(2)
        
        with c7_col1:
            fig_vx_spread = go.Figure()
            fig_vx_spread.add_trace(go.Scatter(x=vx_df.index, y=vx_df['Spread'], name="真实剪刀差 (VXN - VIX)", line=dict(color='#bdc3c7', width=1)))
            fig_vx_spread.add_trace(go.Scatter(x=vx_df.index, y=vx_df['Spread_Fast'], name="EMA5 (微观快线)", line=dict(color='#e74c3c', width=2)))
            fig_vx_spread.add_trace(go.Scatter(x=vx_df.index, y=vx_df['Spread_Slow'], name="EMA21 (趋势慢线)", line=dict(color='#2c3e50', width=2)))
            fig_vx_spread.update_layout(
                title_text="VXN - VIX 波动率剪刀差收敛雷达 (高位死叉确立科技股黄金买点)", 
                template="plotly_white", 
                height=380
            )
            st.plotly_chart(fig_vx_spread, use_container_width=True)
            
        with c7_col2:
            fig_vx_ratio = go.Figure()
            fig_vx_ratio.add_trace(go.Scatter(x=vx_df.index, y=vx_df['Ratio'], name="VXN / VIX 比率", line=dict(color='#9b59b6', width=2, dash='dash')))
            fig_vx_ratio.add_hline(y=1.35, line_dash="dash", line_color="#e74c3c", annotation_text="极端过热线 (1.35)")
            fig_vx_ratio.add_hline(y=1.10, line_dash="dash", line_color="#2ecc71", annotation_text="极限自满线 (1.10)")
            fig_vx_ratio.update_layout(
                title_text="VXN / VIX 情绪乘数溢价区间 (追踪科技股相对大盘的拥挤度)", 
                template="plotly_white", 
                height=380
            )
            st.plotly_chart(fig_vx_ratio, use_container_width=True)
    else:
        st.warning("⚠️ VXN-VIX 科技前哨模块数据未激活或加载失败。")

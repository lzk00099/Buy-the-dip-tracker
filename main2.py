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
import market_runtime as mr
from pathlib import Path
import json

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
YAHOO_CORE_TICKERS = ['QQQ', 'SPY', 'IWM', 'RSP', 'HYG', 'LQD', '^MOVE', '^NDX', '^GSPC']
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
MARKET_CACHE_DIR = str(mr.data_dir() / "cache")
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
        return mr.remote_frame(name)
    age_seconds = time.time() - os.path.getmtime(path)
    if age_seconds > max_age_days * 86400:
        return mr.remote_frame(name)
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
    scores = dict(weighted_risk=float(risk_score), weighted_opportunity=float(opportunity_score),
                  macro_adjustment=float(macro_score), neutral_baseline=float(neutral_model_baseline),
                  net_risk=float(net_risk_score))
    components = [{k:s[k] for k in ('id','risk_score','opportunity_score','effective_weight')}
                  for s in switches]
    inputs = dict(dix=sm_data, vix=vix_data, crypto=crypto_data, quant=quant_data,
                  macro=macro_data, vxn=vxn_vix_data, quotes=intraday_snapshot,
                  source_dates=source_dates)
    record, local = mr.save_observation(scores, components, inputs, model_quality_ok, model_input_date)
    st.session_state['latest_observation'] = record
    remote = [] if os.getenv('SENTINEL_COLLECTOR') else (mr.remote_json('daily_history.json') or {}).get('records',[])
    st.session_state['durable_record_count'] = len(remote)
    if os.getenv('SENTINEL_COLLECTOR'):
        root = mr.data_dir()
        prior_file = root/'daily_history.json'
        prior = json.loads(prior_file.read_text(encoding='utf-8')).get('records',[]) if prior_file.exists() else []
        actual = mr.daily_records(prior + local)
        mr.atomic_json(prior_file, {'version':mr.VERSION,'updated_at':record['observed_at'],'records':actual})
        mr.atomic_json(root/'observations'/record['observed_at'][:10]/(record['id']+'.json'),record)
        mr.atomic_json(root/'latest_quotes.json', {'observed_at':record['observed_at'],
                      'quotes':intraday_snapshot.rename_axis('symbol').reset_index().to_dict('records')})
        mr.atomic_json(root/'health.json', {'observed_at':record['observed_at'], 'quality_ok':model_quality_ok,
                      'source_dates':source_dates,'missing':quality_problems, 'version':mr.VERSION})
        remote = prior
    return mr.history_frame(remote + local)


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
        "fromdate": (today - datetime.timedelta(days=760)).isoformat(),
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

@st.cache_data(ttl=900, show_spinner=False)
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
            period='2y',
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
        close = close.sort_index().dropna(how='all')
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
    return mr.market_session(now_et)


def _get_config_value(name, default=None):
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)

def _paid_volatility_source_enabled():
    return str(_get_config_value("ENABLE_PAID_VOLATILITY_SOURCE", "false")).lower() in {"true","1","yes"}

def fetch_intraday_snapshot():
    global INTRADAY_SNAPSHOT_THIS_RUN, INTRADAY_LAST_ERROR
    info = get_market_session()
    if not info["active"]:
        return pd.DataFrame(columns=["price", "timestamp", "source"])
    if INTRADAY_SNAPSHOT_THIS_RUN is None:
        frame = mr.fetch_quotes(info["session"], info["trading_date"])
        errors = frame.attrs.get("provider_errors", [])
        # Cached function outputs age too; validate again at the point of use.
        INTRADAY_SNAPSHOT_THIS_RUN = mr.validate_quotes(frame)
        INTRADAY_SNAPSHOT_THIS_RUN.attrs["provider_errors"] = errors
        INTRADAY_LAST_ERROR = " | ".join(errors[-6:])
    return INTRADAY_SNAPSHOT_THIS_RUN

def overlay_intraday_prices(daily_frame, symbols, require_all=False):
    return mr.overlay(daily_frame, fetch_intraday_snapshot(), symbols, require_all)

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
        "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history",
        params={"instId": "BTC-USDT-SWAP", "period": "1Dutc", "limit": "60"}
    )
    funding = _request_json_with_backoff(
        "https://www.okx.com/api/v5/public/funding-rate-history",
        params={"instId": "BTC-USDT-SWAP", "limit": "100"}
    )
    return price, oi, funding


@st.cache_data(ttl=300, show_spinner=False)
def fetch_okx_current_oi():
    return _request_json_with_backoff(
        "https://www.okx.com/api/v5/public/open-interest",
        params={"instType": "SWAP", "instId": "BTC-USDT-SWAP"}
    )

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
        ).sort_index().dropna().tail(180)
        close_data = overlay_intraday_prices(
            close_data, ['^VIX', '^VIX3M'], require_all=True
        ).tail(180)
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
                "df": close_data.tail(180),
                "fetched_at": market_data_timestamp(close_data),
                "quote_mode": close_data.attrs.get("quote_mode", "daily_fallback"),
                "quote_age_minutes": close_data.attrs.get("quote_age_minutes"),
                "quote_sources": close_data.attrs.get("quote_sources", "CBOE EOD")
            }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False, "fetched_at": "异常断流"}
    return {"error": True, "msg": "No data", "bottom_active": False, "top_active": False, "fetched_at": "空数据"}

def fetch_crypto_signals():
    live = {}
    live_error = ''
    try:
        # Revalidate after cache lookup: a cached response must not look current.
        live = mr.current_crypto_oi(fetch_okx_current_oi(), now=market_session['now_et'])
    except Exception as exc:
        live_error = mr.safe_error(exc)
    context = {'oi_current': live, 'oi_current_error': live_error,
               'oi_instrument': 'BTC-USDT-SWAP', 'oi_unit': 'USD'}
    try:
        price_res, r_res, fr_res = fetch_okx_signal_inputs()
        df_merged = mr.crypto_daily_frame(price_res, r_res, fr_res, now=market_session['now_et'])

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
            **context,
            "btc_price": f"${current_price:,.2f}",
            "price_trend": price_trend_str,
            "oi": f"${current_oi:,.0f} USD",
            "oi_trend": oi_trend_str,
            "funding_rate": f"{current_fr:.4f}%", 
            "diag_status": diag_status,
            "bottom_active": bottom_active, 
            "top_active": top_active, 
            "error": False,
            "hist_df": df_merged.tail(90), # OKX 可用日线窗口内尽量保留事件上下文
            "data_date": df_merged.index[-1].date().isoformat(),
            "fetched_at": df_merged.index[-1].strftime('%Y-%m-%d') + " UTC 日线（非实时BTC）"
        }
    except Exception as e:
        # 捕获真实报错抛给前端
        return {**context, "error": True, "msg": mr.safe_error(e), "bottom_active": False, "top_active": False, "fetched_at": "异常拦截"}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_squeezemetrics_data():
    path = Path(_market_cache_path('dix.csv'))
    try:
        response = requests.get('https://squeezemetrics.com/monitor/static/DIX.csv',
                                headers={'User-Agent':'Mozilla/5.0'}, timeout=15)
        response.raise_for_status()
        df = mr.normalize_dix(pd.read_csv(io.StringIO(response.text)))
        df.to_csv(path,index=False)
    except Exception:
        try:
            if path.exists():
                df = mr.normalize_dix(pd.read_csv(path))
            else:
                remote = mr.remote_frame('dix.csv').reset_index()
                df = mr.normalize_dix(remote)
        except Exception:
            return {'error':True,'is_mock':False,'df':pd.DataFrame(),'fetched_at':'数据缺失'}
    latest = df.iloc[-1]
    return {'error':False,'is_mock':False,'dix':float(latest.dix),'gex':float(latest.gex),
            'df':df.tail(260), 'data_date':latest.date.date().isoformat(),
            'fetched_at':latest.date.strftime('%Y-%m-%d')+' 收盘（SqueezeMetrics）'}


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
        breadth_prices = overlay_intraday_prices(yf_data, ['SPY','RSP'], require_all=True)
        yf_data = overlay_intraday_prices(
            yf_data, ['QQQ','SPY','IWM'], require_all=True
        )
        missing = [
            symbol for symbol in required
            if symbol not in yf_data.columns or yf_data[symbol].dropna().shape[0] < 126
        ]
        if missing:
            raise Exception(f"CTA 核心行情缺失: {', '.join(missing)}")
        data = yf_data[['QQQ','SPY','IWM']].dropna().tail(260).copy()
        latest = data.iloc[-1]
        
        corr_official = fetch_cboe_official_history('COR1M')
        dspx_official = fetch_cboe_official_history('DSPX')
        
        corr_series = corr_official.dropna().tail(260)
        dspx_series = dspx_official.dropna().tail(260)
        # NaN preserves the CTA module while explicitly invalidating breadth.
        if corr_series.empty:
            corr_series = pd.Series(np.nan, index=data.index)
        if dspx_series.empty:
            dspx_series = pd.Series(np.nan, index=data.index)

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
        corr_is_broken = (corr_official.empty or dspx_official.empty or corr_series.tail(10).nunique() <= 1 or dspx_series.tail(10).nunique() <= 1)
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
        breadth_latest = breadth_prices[['SPY','RSP']].dropna().iloc[-1]
        spy_rsp_ratio = breadth_latest['SPY'] / breadth_latest['RSP']
        
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
            "corr_date": corr_official.index[-1].date().isoformat() if not corr_official.empty else None,
            "dspx_date": dspx_official.index[-1].date().isoformat() if not dspx_official.empty else None,
            "corr_diag": corr_diag,          
            "disp_diag": disp_diag,      
            "combined_diag": combined_diag,  
            "corr_risk_level": corr_risk_level,
            "corr_risk_diag": corr_risk_diag,
            "df_hist": df_hist.tail(180),
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
        ).sort_index().dropna().tail(180)
        df = overlay_intraday_prices(
            df, ['^VXN', '^VIX'], require_all=True
        ).tail(180)
        
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

        raw = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
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
        macro_dates = {}
        for symbol in ('^MOVE','^VVIX','HYG','LQD'):
            series = raw[symbol].dropna() if symbol in raw else pd.Series(dtype=float)
            if len(series) < 50:
                raise ValueError(f'宏观指标{symbol}有效样本不足')
            macro_dates[symbol] = series.index[-1].date().isoformat()
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
            "source_dates": macro_dates,
            "hist_df": raw,
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
auto_refresh_seconds = 300
st.sidebar.markdown("### ⏱️ 盘中数据状态")
st.sidebar.caption(f"当前：{market_session['label']} · {market_session['now_et'].strftime('%Y-%m-%d %H:%M ET')}")
auto_refresh = st.sidebar.checkbox("每5分钟刷新（页面开启时）", value=True,
                                  help="休市时也检查是否开盘；交易日与半日市按交易所日历判断。")
if st.sidebar.button("🔄 立即刷新数据"):
    st.cache_data.clear()

# Streamlit's supported rerun mechanism also wakes a page left open overnight.
# A fragment timer requests one full run; normal full runs reset its timer.
if auto_refresh and not os.getenv('SENTINEL_COLLECTOR'):
    st.session_state['_last_full_run'] = time.monotonic()
    @st.fragment(run_every=auto_refresh_seconds)
    def refresh_timer():
        if time.monotonic() - st.session_state['_last_full_run'] >= auto_refresh_seconds - 1:
            st.rerun(scope="app")
    refresh_timer()

vix_data = fetch_vix_data()
crypto_data = fetch_crypto_signals()
sm_data = fetch_squeezemetrics_data()
quant_data = calculate_quant_and_breadth_signals()
macro_data = fetch_macro_liquidity_overlay()

now_str = datetime.datetime.now(US_EASTERN).strftime('%Y-%m-%d %H:%M ET')
vxn_vix_data = fetch_vxn_vix_data()

intraday_snapshot = fetch_intraday_snapshot()
quote_rows = []
for symbol in INTRADAY_ETF_TICKERS + ['^NDX', '^GSPC'] + INTRADAY_VOL_TICKERS:
    if symbol in intraday_snapshot.index:
        row = intraday_snapshot.loc[symbol]
        quote_rows.append({'标的':symbol, '价格':round(float(row.price),4), '行情时间ET':row.timestamp.tz_convert(US_EASTERN).strftime('%m-%d %H:%M'),
                           '来源':row.source, '状态':'盘中/延迟报价'})
    else:
        quote_rows.append({'标的':symbol,'行情时间ET':'见模块日线日期','来源':'日线回退','状态':'暂无有效盘中报价'})
with st.sidebar.expander('逐项行情时间与来源', expanded=True):
    st.dataframe(pd.DataFrame(quote_rows),hide_index=True,use_container_width=True)
if INTRADAY_LAST_ERROR:
    with st.sidebar.expander('接口诊断'):
        st.caption(INTRADAY_LAST_ERROR)
st.sidebar.caption('刷新频率不等于实时授权。DIX/GEX、COR1M/DSPX与OKX日线模块仍按各自发布频率更新。')
if not alpaca_configured:
    st.sidebar.info('ETF使用Yahoo/Nasdaq公开备用源；配置Alpaca可增加稳定的独立通道。')

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
    if quant_data.get("error") or quant_data.get("corr_is_broken"):
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
        "value": f"GEX: {gex_val:,.0f} | DIX: {dix_val:.2f}%",
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
            "source": "CBOE 日线基准 ✖ 盘中/延迟指数报价（按原始时间校验）",
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
        "value": f"BTC日线: {crypto_data.get('btc_price', 'N/A')} ({crypto_data.get('price_trend', '')}) | OI(日线): {crypto_data.get('oi', 'N/A')} ({crypto_data.get('oi_trend', '')}) | 费率: {crypto_data.get('funding_rate', 'N/A')}",
        "source": "OKX：BTC-USDT现货日线 ✖ BTC-USDT-SWAP永续OI(USD)/资金费率（非全市场）",
        "desc_bottom": "【缩量爆仓抄底】当 **价格下跌 + OI显著下降 + 费率转负**。代表做多杠杆被彻底清算，市场流动性恐慌见底，是高胜率左侧或右侧建仓点。",
        "desc_top": "【拥挤过载逃顶】触发两种情况立即防御：① **价格上涨 + OI上升 + 费率极高** (多头拥挤，极易被爆)；② **价格上涨 + OI下降** (缺乏新资金的假突破)。",
        "fetched_status": f"数据抓取失败 🔴 <br><span style='font-size:8pt;color:#e74c3c;'>异常原因: {crypto_data.get('msg', '未知断流')}</span>" if crypto_data["error"] else (
            f"<div style='background-color:#f4f6f7; padding:8px; border-radius:5px; margin-bottom:5px; font-weight:bold; color:#2c3e50;'>{crypto_data.get('diag_status')}</div>"
        ),
        "update_cycle": "OI现值每5分钟检查；日线模型每30分钟更新（含当日未收盘数据）",
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
        "update_cycle": "每5分钟检查ETF报价，供应商可能延迟",
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
            f"<br>COR1M日线：{quant_data.get('corr_date', '缺失')}；DSPX日线：{quant_data.get('dspx_date', '缺失')}"
        ),
        "update_cycle": "CBOE日线 + ETF盘中快照",
        "last_updated": f"COR1M {quant_data.get('corr_date','缺失')} / DSPX {quant_data.get('dspx_date','缺失')}"
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
        "source": "CBOE 日线基准 ✖ 盘中/延迟 VXN/VIX 报价（按原始时间校验）",
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
# Raw risk-minus-opportunity has a positive structural bias because each switch
# uses a different score scale.  Centre it on the same neutral state used by the
# historical replay, so zero genuinely means "no directional model pressure".
neutral_switch_scores = {
    1: (28, 0), 2: (20, 25), 3: (32, 20),
    4: (30, 25), 5: (28, 18), 6: (24, 24)
}
neutral_model_baseline = sum(
    (neutral_switch_scores[s["id"]][0] - neutral_switch_scores[s["id"]][1]) * s["effective_weight"]
    for s in switches
) / total_effective_weight
net_risk_score = (
    weighted_risk_score - weighted_opportunity_score
    + macro_adjustment - neutral_model_baseline
)
# Availability is separate from the score. Missing data cannot become a neutral signal.
def frame_date(payload, key):
    frame = payload.get(key)
    return pd.Timestamp(frame.index[-1]).date().isoformat() if isinstance(frame,pd.DataFrame) and not frame.empty else None

source_dates = {'DIX':sm_data.get('data_date'), 'VIX/VIX3M':frame_date(vix_data,'df'),
                'VXN/VIX':frame_date(vxn_vix_data,'df_hist'), 'ETF':frame_date(quant_data,'df_hist'),
                'COR1M':quant_data.get('corr_date'),'DSPX':quant_data.get('dspx_date'),
                'OKX':crypto_data.get('data_date')}
source_dates.update(macro_data.get('source_dates',{}))
quality_problems = [name for name,payload in [('DIX',sm_data),('VIX',vix_data),('VXN',vxn_vix_data),
                    ('ETF',quant_data),('OKX',crypto_data),('宏观',macro_data)] if payload.get('error')]
if quant_data.get('corr_is_broken'):
    quality_problems.append('COR1M/DSPX缺失')
expected_day = market_session['last_completed']
quality_problems += [f'{name}停留在{day or "未知日期"}' for name,day in source_dates.items()
                     if not day or day < expected_day]
model_quality_ok = not quality_problems
model_input_date = source_dates.get('ETF')
try:
    model_score_history = record_model_score_snapshot(weighted_risk_score, weighted_opportunity_score,
                                                     macro_adjustment, net_risk_score)
except Exception as exc:
    model_score_history = pd.DataFrame()
    st.error('模型实录写入失败：' + mr.safe_error(exc))
    if os.getenv('SENTINEL_COLLECTOR'):
        raise
if os.getenv('SENTINEL_COLLECTOR'):
    st.stop()

high_risk_names = [f"#{s['rank']} {s['name']}({s['risk_level']})" for s in switches if s["risk_score"] >= 72]
opportunity_names = [f"#{s['rank']} {s['name']}({s['opportunity_level']})" for s in switches if s["opportunity_score"] >= 64 and s["risk_score"] < 72]
ranking_line = " > ".join([f"{s['rank']}.{s['name'].split(' ')[0]}(权重{s['weight']:.2f})" for s in switches])

if net_risk_score >= 30 or weighted_risk_score >= 72:
    status_color = "red"
    action_title = "🚨 【红色防御：加权风险占优，进入系统性降杠杆模式】"
    action_text = (
        f"<b>加权诊断</b>：风险分 <b>{weighted_risk_score:.1f}</b> / 机会分 <b>{weighted_opportunity_score:.1f}</b> / "
        f"宏观修正 <b>{macro_adjustment:+.1f}</b> / 净风险 <b>{net_risk_score:.1f}</b>。<br>"
        f"<b>主导风险</b>：{'；'.join(high_risk_names[:3]) if high_risk_names else '风险来自多个中等级别开关叠加'}。<br>"
        "策略：净多头降到防御仓位，优先处理高Beta、弱广度、弱现金流标的；新开仓只允许小仓试错，所有盈利仓提高保护性止盈。"
    )
elif net_risk_score <= -20 and weighted_opportunity_score >= 56:
    status_color = "green"
    action_title = "🚀 【绿色进攻：加权机会占优，允许分批抄底/加仓】"
    action_text = (
        f"<b>加权诊断</b>：风险分 <b>{weighted_risk_score:.1f}</b> / 机会分 <b>{weighted_opportunity_score:.1f}</b> / "
        f"宏观修正 <b>{macro_adjustment:+.1f}</b> / 净风险 <b>{net_risk_score:.1f}</b>。<br>"
        f"<b>主导机会</b>：{'；'.join(opportunity_names[:3]) if opportunity_names else '机会来自多个开关温和修复'}。<br>"
        "策略：允许分批抄底或提高核心仓位，但仍需避开财务/趋势双弱个股；若盘中实时开关重新转红，立即暂停加仓。"
    )
elif net_risk_score >= 10:
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

quote_cols = st.columns(4)
for column, symbol in zip(quote_cols, ['QQQ','SPY','^NDX','^VIX']):
    if symbol in intraday_snapshot.index:
        observed = intraday_snapshot.loc[symbol]
        column.metric(symbol, f'{observed.price:,.2f}')
        column.caption(observed.timestamp.tz_convert(US_EASTERN).strftime('%m-%d %H:%M ET') + ' · ' + observed.source)
    else:
        column.metric(symbol, '暂无盘中报价')
        column.caption('历史图表保留最近有效日线')

if not model_quality_ok:
    status_color = '#7f8c8d'
    action_title = '⏸️ 数据不完整：综合信号暂不可用'
    action_text = '缺失/落后输入：' + '；'.join(quality_problems) + '。已保存诊断记录，未写入有效净风险趋势。'

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

@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily_equity_benchmark():
    try:
        close = fetch_yahoo_core_close()
        columns = [symbol for symbol in ('QQQ', 'SPY') if symbol in close.columns]
        return close[columns].dropna(how='all').copy() if columns else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

daily_equity_benchmark = fetch_daily_equity_benchmark()

def add_equity_benchmark_overlay(fig, chart_index, show_qqq, show_spy, has_secondary_y=False):
    """Overlay QQQ/SPY as rebased daily performance without altering an indicator's scale."""
    if daily_equity_benchmark.empty or not (show_qqq or show_spy):
        return
    dates = pd.DatetimeIndex(pd.to_datetime(chart_index)).tz_localize(None).normalize()
    benchmark = daily_equity_benchmark.reindex(dates).ffill()
    columns = [symbol for symbol, enabled in (('QQQ', show_qqq), ('SPY', show_spy)) if enabled and symbol in benchmark]
    if not columns:
        return
    benchmark = benchmark[columns].dropna(how='all')
    if benchmark.empty:
        return
    benchmark = benchmark.divide(benchmark.bfill().iloc[0]).multiply(100)
    axis = 'y3' if has_secondary_y else 'y2'
    colors = {'QQQ': '#2980b9', 'SPY': '#8e44ad'}
    for symbol in columns:
        fig.add_trace(go.Scatter(
            x=benchmark.index, y=benchmark[symbol], name=f'{symbol}（起点=100）',
            yaxis=axis, mode='lines', line=dict(color=colors[symbol], width=1.4),
            hovertemplate=f'%{{x|%Y-%m-%d}}<br>{symbol} 相对表现: %{{y:.1f}}<extra></extra>'
        ))
    axis_config = dict(
        title='QQQ / SPY（起点=100）', overlaying='y', side='right', showgrid=False
    )
    if has_secondary_y:
        axis_config.update(anchor='free', position=.96)
        fig.update_layout(yaxis3=axis_config)
    else:
        fig.update_layout(yaxis2=axis_config)

def chart_benchmark_controls(key_prefix):
    controls = st.columns(2)
    return (
        controls[0].checkbox("显示 QQQ 对照（右轴）", value=False, key=f"{key_prefix}_qqq"),
        controls[1].checkbox("显示 SPY 对照（右轴）", value=False, key=f"{key_prefix}_spy")
    )

def add_risk_opportunity_markers(fig, x, y, risk_mask, opportunity_mask, secondary_y=False,
                                 risk_label="风险点", opportunity_label="机会点", opportunity_y=None):
    """Show each switch's own historical triggers on its indicator, not on the model aggregate."""
    x_values = pd.Index(x)
    y_values = pd.Series(np.asarray(y), index=x_values)
    opportunity_values = pd.Series(
        np.asarray(y if opportunity_y is None else opportunity_y), index=x_values
    )
    risk = pd.Series(np.asarray(risk_mask), index=x_values).fillna(False).astype(bool)
    opportunity = pd.Series(np.asarray(opportunity_mask), index=x_values).fillna(False).astype(bool)
    traces = [
        (risk, risk_label, "triangle-down", "#c0392b", y_values),
        (opportunity, opportunity_label, "triangle-up", "#27ae60", opportunity_values)
    ]
    for mask, label, symbol, color, values in traces:
        if not mask.any():
            continue
        trace = go.Scatter(
            x=x_values[mask], y=values.loc[mask], mode="markers", name=label,
            marker=dict(symbol=symbol, size=10, color=color, line=dict(color="#ffffff", width=1)),
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{label}<br>指标值: %{{y:.3f}}<extra></extra>"
        )
        if secondary_y:
            fig.add_trace(trace, secondary_y=False)
        else:
            fig.add_trace(trace)

def state_entry_points(active_state, mark_initial_active=False):
    """Emit one point when a condition is entered; optionally retain an active window start."""
    if isinstance(active_state, pd.Series):
        active = active_state.copy().fillna(False).astype(bool)
    else:
        active = pd.Series(np.asarray(active_state)).fillna(False).astype(bool)
    entered = active & ~active.shift(1, fill_value=False)
    if not entered.empty:
        entered.iloc[0] = bool(active.iloc[0]) if mark_initial_active else False
    return entered

st.markdown("### 📈 模型日线净风险趋势 · 实际采集")
st.caption('净风险 = 加权风险 − 加权机会 + 宏观修正 − 中性基线。它是规则分数，不是下跌概率。'
           '每个交易日显示最后一次有效观察；保留采集时刻与输入日期，未采集日期不补造。')
if not model_quality_ok:
    st.warning('当前数据不完整或落后，暂停生成有效综合信号：' + '；'.join(quality_problems))
durable_count = st.session_state.get('durable_record_count',0)
if durable_count:
    st.caption(f'已从GitHub独立数据分支读取 {durable_count} 个交易日实录。')
else:
    st.warning('尚未读到后台持久实录。当前页面记录只是本机副本；请查看采集任务状态。')
health = mr.remote_json('health.json') or {}
if health.get('observed_at'):
    st.caption(f"后台最近采集：{health['observed_at']}；完整性：{'通过' if health.get('quality_ok') else '部分数据缺失'}")
    if pd.Timestamp.now(tz='UTC') - pd.Timestamp(health['observed_at']) > pd.Timedelta(hours=36):
        st.error('后台采集已超过36小时未更新，请检查GitHub Actions。')
if not model_score_history.empty:
    score_plot = model_score_history.sort_values('timestamp').copy()
    # Explicit missing-session rows prevent lines spanning data gaps.
    calendar = mr.mcal.get_calendar('NYSE').schedule(start_date=score_plot.timestamp.min().date(),
                                                   end_date=max(market_session['last_completed'],score_plot.timestamp.max().date().isoformat()))
    full_index = pd.DatetimeIndex(calendar.index).tz_localize(US_EASTERN)
    full_index = full_index.union(pd.DatetimeIndex(score_plot.timestamp))
    plotted = score_plot.set_index('timestamp').reindex(full_index)
    fig_scores = go.Figure(go.Scatter(x=plotted.index,y=plotted.net_risk,mode='lines+markers',
                    connectgaps=False,name='每日最后一次有效实录',
                    customdata=plotted[['weighted_risk','weighted_opportunity','macro_adjustment','neutral_baseline','observed_at']],
                    hovertemplate='%{x|%Y-%m-%d}<br>净风险 %{y:.2f}<br>风险 %{customdata[0]:.2f}'
                                  '<br>机会 %{customdata[1]:.2f}<br>宏观 %{customdata[2]:.2f}'
                                  '<br>中性基线 %{customdata[3]:.2f}<br>采集UTC %{customdata[4]}<extra></extra>'))
    for level,label,color in [(30,'防御阈值','#c0392b'),(10,'谨慎阈值','#e67e22'),(-8,'修复阈值','#27ae60'),(0,'中性基线','#7f8c8d')]:
        fig_scores.add_hline(y=level,line_dash='dot',line_color=color,annotation_text=label)
    fig_scores.update_layout(template='plotly_white',height=340,yaxis_title='净风险规则分数',xaxis_title='数据交易日（ET）')
    fig_scores.update_xaxes(tickformat='%Y-%m-%d')
    if len(plotted) == 1:
        fig_scores.update_xaxes(range=[plotted.index[0]-pd.Timedelta(days=2),plotted.index[0]+pd.Timedelta(days=2)],dtick=86400000)
    st.plotly_chart(fig_scores,use_container_width=True)
    st.download_button('下载每日实录CSV',score_plot.to_csv(index=False).encode('utf-8-sig'),
                       file_name='observed_model_history.csv',mime='text/csv')
else:
    st.info('暂无符合完整性要求的每日实录。旧版的假设性历史重算已停止显示。')

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
st.markdown("### 🗺️ 指数 CTA 枢轴与技术低位区")
index_choice = st.radio('选择指数', ['标普 500（SPX）', '纳指 100（NDX）'], horizontal=True)
index_code, index_symbol = ('SPX','^GSPC') if index_choice.startswith('标普') else ('NDX','^NDX')

def fetch_index_chart_data(symbol):
    try:
        close = fetch_yahoo_core_close()
        if symbol not in close.columns:
            return pd.DataFrame()
        close = overlay_intraday_prices(close, [symbol])
        return close[[symbol]].dropna().tail(260).rename(columns={symbol:'Close'})
    except Exception:
        return pd.DataFrame()

index_data = fetch_index_chart_data(index_symbol)
if not index_data.empty:
    fig_index = go.Figure()
    latest_index = float(index_data['Close'].iloc[-1])
    fig_index.add_trace(go.Scatter(x=index_data.index,y=index_data.Close,mode='lines',
                                  name=f'{index_code} 实际走势',line=dict(color='#2980b9',width=2.5)))
    fig_index.add_hline(y=latest_index,line_color='#2c3e50',
                       annotation_text=f"最新有效位 {latest_index:,.2f} · {market_data_timestamp(index_data)}",
                       annotation_position='top right')
    plotted_values = list(index_data.Close.tail(66))
    levels = mr.technical_levels(index_data.Close)
    if levels:
        st.caption(f"技术参考更新至 {levels['asof']} 收盘；{index_code}现货指数。")
        show_trend = st.checkbox('显示21/63/126日趋势失守参考（价格模型）',value=index_code=='NDX')
        if show_trend:
            for horizon,value in levels['trend_triggers'].items():
                fig_index.add_hline(y=value,line_dash='dot',line_color='#e67e22',
                                   annotation_text=f'{horizon}日趋势参考（模型） {value:,.0f}')
                plotted_values.append(value)
        fig_index.add_hrect(y0=levels['support_low'],y1=levels['support_high'],fillcolor='#2ecc71',
                            opacity=.12,line_width=0,annotation_text='20/63日收盘低位区（技术参考）')
        plotted_values.extend([levels['support_low'],levels['support_high']])
        st.caption('低位区由最近20/63日最低收盘价形成，每日更新；没有真实资金流证据，不能称为机构承接区。'
                   '趋势参考为最近20/62/125个已完成交易日收盘均值，不是机构CTA实际仓位模型。')
    institutional = mr.verified_levels(Path(__file__).with_name(index_code.lower()+'_levels.json'),instrument=index_code)
    if institutional:
        for level in institutional['levels']:
            fig_index.add_hline(y=level['value'],line_dash='dash',line_color='#c0392b',
                               annotation_text=level['label'],annotation_position='bottom left')
            plotted_values.append(level['value'])
        st.caption(f"报道日期：{institutional['asof']} · {institutional['publisher']}；"
                   f"看板陈旧保护到期：{institutional['expires']}（届时隐藏，非机构承诺的有效期）。")
        st.info('公开报道将这些枢轴归于高盛；已核对报道，未独立取得高盛原始报告。'
                '跌破枢轴表示趋势状态可能转弱，不代表必定触发卖单。')
        st.link_button('查看CTA点位报道与归属',institutional['source_url'])
    else:
        st.info(f'当前没有经核对且未过期的{index_code}机构CTA点位。只显示明确标注的技术参考。')
    fig_index.update_xaxes(range=[index_data.index[-min(66,len(index_data))],index_data.index[-1]])
    fig_index.update_layout(title=f'{index_code}：实际行情、报道枢轴与动态技术参考',template='plotly_white',
                            yaxis=dict(title=f'{index_code} 现货点数',range=[min(plotted_values)*.98,max(plotted_values)*1.02],
                                       tickformat=',.0f'),xaxis_rangeslider_visible=False,height=540,
                            margin=dict(l=20,r=20,t=40,b=20))
    st.plotly_chart(fig_index,use_container_width=True)
else:
    st.warning(f'{index_code}行情尚不可用，暂不绘制点位图。')

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
    tab1_qqq, tab1_spy = chart_benchmark_controls("tab1")
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
        dix = pd.to_numeric(plot_df['dix'], errors='coerce')
        gex = pd.to_numeric(plot_df['gex'], errors='coerce')
        # A marker means the state was entered, not that it simply persisted.
        sm_risk_state = ((gex < 0) & (dix < 42.5)) | ((gex >= 0) & (dix < 40)) | (gex < -1_000_000_000)
        sm_opportunity_state = (gex >= 0) & (dix >= 45)
        sm_risk_points = state_entry_points(sm_risk_state, mark_initial_active=True)
        sm_opportunity_points = state_entry_points(sm_opportunity_state, mark_initial_active=True)
        add_risk_opportunity_markers(
            fig_sm, plot_df['date'], dix, sm_risk_points, sm_opportunity_points,
            secondary_y=True, risk_label="GEX/DIX 进入风险区", opportunity_label="GEX/DIX 进入机会区"
        )
        sm_extreme_state = (gex < 0) & (dix < 40)
        sm_extreme_points = state_entry_points(sm_extreme_state, mark_initial_active=True)
        if sm_extreme_points.any():
            fig_sm.add_trace(go.Scatter(
                x=pd.Index(plot_df['date'])[sm_extreme_points.to_numpy()], y=dix.iloc[sm_extreme_points.to_numpy()],
                mode='markers', name='GEX/DIX 极端风险',
                marker=dict(symbol='x', size=11, color='#7b241c'),
                hovertemplate='%{x|%Y-%m-%d}<br>GEX/DIX 极端风险：负 Gamma + DIX&lt;40<extra></extra>'
            ), secondary_y=False)
        # A recovery is only actionable after an extreme state and a genuine
        # Gamma/DIX repair; simply leaving DIX<40 is not labelled as a dip buy.
        sm_extreme_recovery = (
            sm_extreme_state.shift(1).fillna(False)
            & ~sm_extreme_state
            & (dix > dix.shift(1))
            & (gex > gex.shift(1))
            & (gex >= 0)
        )
        sm_extreme_recovery_points = state_entry_points(sm_extreme_recovery, mark_initial_active=True)
        if sm_extreme_recovery_points.any():
            recovery_mask = sm_extreme_recovery_points.to_numpy()
            fig_sm.add_trace(go.Scatter(
                x=pd.Index(plot_df['date'])[recovery_mask], y=dix.iloc[recovery_mask],
                mode='markers', name='极端风险后修复机会',
                marker=dict(symbol='star', size=12, color='#1e8449'),
                hovertemplate='%{x|%Y-%m-%d}<br>极端风险后修复机会：GEX转正且DIX回升<extra></extra>'
            ), secondary_y=False)
        fig_sm.update_layout(title_text="DIX 与做市商 GEX 双向变动曲线", template="plotly_white", height=400)
        fig_sm.update_yaxes(title_text="<b>DIX 比例</b>", secondary_y=False)
        fig_sm.update_yaxes(title_text="<b>Gamma 敞口绝对值</b>", secondary_y=True)
        add_equity_benchmark_overlay(fig_sm, plot_df['date'], tab1_qqq, tab1_spy, has_secondary_y=True)
        st.plotly_chart(fig_sm, use_container_width=True)
    else:
        st.warning("数据不可用。")

# --- TAB 2 ---
with tab2:
    tab2_qqq, tab2_spy = chart_benchmark_controls("tab2")
    if not vix_data["error"] and "df" in vix_data:
        v_df = vix_data["df"]
        v_col1, v_col2 = st.columns(2)
        ratio_prev = v_df['Ratio'].shift(1)
        ratio_golden = (v_df['Ratio_Fast'].shift(1) <= v_df['Ratio_Slow'].shift(1)) & (v_df['Ratio_Fast'] > v_df['Ratio_Slow'])
        ratio_death = (v_df['Ratio_Fast'].shift(1) >= v_df['Ratio_Slow'].shift(1)) & (v_df['Ratio_Fast'] < v_df['Ratio_Slow'])
        vix_opportunity_state = ((ratio_prev <= 1.0) & (v_df['Ratio'] > 1.0)) | (ratio_golden & (v_df['Ratio'] <= 1.05))
        vix_risk_state = (v_df['Ratio'] >= 1.25) | ((ratio_prev >= 1.0) & (v_df['Ratio'] < 1.0)) | (ratio_death & (v_df['Ratio'] >= 1.15))
        vix_opportunity_points = state_entry_points(vix_opportunity_state, mark_initial_active=True)
        vix_risk_points = state_entry_points(vix_risk_state, mark_initial_active=True)
        with v_col1:
            fig_vix_spot = go.Figure()
            fig_vix_spot.add_trace(go.Scatter(x=v_df.index, y=v_df['^VIX'], name="VIX 现货指数", line=dict(color="#e67e22", width=2)))
            fig_vix_spot.add_hline(y=12.0, line_dash="dash", line_color="#c0392b", annotation_text="自满安全线 (12.0)")
            add_risk_opportunity_markers(
                fig_vix_spot, v_df.index, v_df['^VIX'], vix_risk_points, vix_opportunity_points,
                risk_label="VIX 期限风险点", opportunity_label="VIX 修复机会点"
            )
            fig_vix_spot.update_layout(title_text="图表 A: VIX 现货恐慌指数趋势", template="plotly_white", height=400)
            add_equity_benchmark_overlay(fig_vix_spot, v_df.index, tab2_qqq, tab2_spy)
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
            add_risk_opportunity_markers(
                fig_vix_ratio, v_df.index, v_df['Ratio'], vix_risk_points, vix_opportunity_points,
                risk_label="期限结构风险点", opportunity_label="期限结构机会点"
            )
            
            fig_vix_ratio.update_layout(
                title_text="图表 B: VIX3M / VIX 期限结构动能雷达 (均线交叉 ✖ 区间风控决策模型)", 
                template="plotly_white", 
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            add_equity_benchmark_overlay(fig_vix_ratio, v_df.index, tab2_qqq, tab2_spy)
            st.plotly_chart(fig_vix_ratio, use_container_width=True)

# --- TAB 3 ---
with tab3:
    live_oi = crypto_data.get('oi_current', {})
    if live_oi:
        st.metric('OKX BTC-USDT 永续合约 · 当前OI（USD）', f"${live_oi['value']:,.0f}")
        st.caption('行情时间：' + pd.Timestamp(live_oi['timestamp']).tz_convert(US_EASTERN).strftime('%Y-%m-%d %H:%M:%S ET')
                   + ' · 每5分钟检查；仅此合约，不是全市场总量。')
    else:
        st.info('当前OI暂不可用：' + crypto_data.get('oi_current_error', '未取得有效快照') + '；不以0代替。')
    st.caption('历史图与信号统一使用BTC-USDT永续OI（USD）及其日均资金费率，按UTC日对齐；当日可能尚未收盘。')
    tab3_qqq, tab3_spy = chart_benchmark_controls("tab3")
    if not crypto_data.get("error", True) and crypto_data.get("hist_df") is not None:
        c_df = crypto_data["hist_df"]
        
        # 构建主副 Y 轴双轴图表
        fig_crypto = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 主 Y 轴：OKX 合约持仓名义金额 (OI) 面面积图
        fig_crypto.add_trace(
            go.Scatter(
                x=c_df.index, y=c_df['oi'], 
                name="BTC-USDT永续 OI (USD)",
                line=dict(color="#3498db", width=2), connectgaps=False,
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
        crypto_risk_state = (
            (c_df['close'] > c_df['price_ma7'])
            & (c_df['oi'] > c_df['oi_ma7'] * 1.03)
            & (c_df['funding_rate'] >= 0.025)
        ) | ((c_df['close'].diff() > 0) & (c_df['oi'] < c_df['oi_ma7'] * 0.92))
        crypto_opportunity_state = (
            (c_df['close'] < c_df['price_ma7'])
            & (c_df['oi'] < c_df['oi_ma7'] * 0.95)
            & (c_df['funding_rate'] <= 0.005)
        )
        crypto_risk_points = state_entry_points(crypto_risk_state, mark_initial_active=True)
        crypto_opportunity_points = state_entry_points(crypto_opportunity_state, mark_initial_active=True)
        add_risk_opportunity_markers(
            fig_crypto, c_df.index, c_df['oi'], crypto_risk_points, crypto_opportunity_points,
            secondary_y=False, risk_label="杠杆拥挤风险点", opportunity_label="去杠杆机会点"
        )
        
        fig_crypto.update_layout(
            title_text="加密离岸雷达：BTC-USDT永续 OI (USD) 与日均资金费率", 
            template="plotly_white", 
            height=400,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig_crypto.update_yaxes(title_text="<b>BTC-USDT永续 OI (USD)</b>", secondary_y=False)
        fig_crypto.update_yaxes(title_text="<b>日均资金费率 (%)</b>", secondary_y=True)
        add_equity_benchmark_overlay(fig_crypto, c_df.index, tab3_qqq, tab3_spy, has_secondary_y=True)
        
        st.plotly_chart(fig_crypto, use_container_width=True)
    else:
        # 把底层抓取的异常信息暴露在 Tab 面板里
        st.warning(f"⚠️ 离岸高杠杆图表渲染终止。底层数据异常：{crypto_data.get('msg', '未知环境异常')}")
    
# --- TAB 4 ---
with tab4:
    tab4_qqq, tab4_spy = chart_benchmark_controls("tab4")
    if not quant_data["error"] and "df_hist" in quant_data:
        h_df = quant_data["df_hist"]
        fig_cta = go.Figure()
        fig_cta.add_trace(go.Scatter(x=h_df.index, y=h_df['cta_shorts'], name="系统性空头得分", line=dict(color="#e74c3c", width=2.5)))
        fig_cta.add_trace(go.Scatter(x=h_df.index, y=h_df['cta_longs'], name="系统性多头得分", line=dict(color="#2ecc71", width=2.5)))
        fig_cta.add_hline(y=2, line_dash="dash", line_color="#34495e", annotation_text="极值激活线 (2)")
        cta_risk_points = state_entry_points(h_df['cta_longs'] >= 2, mark_initial_active=True)
        cta_opportunity_points = state_entry_points(h_df['cta_shorts'] >= 2, mark_initial_active=True)
        add_risk_opportunity_markers(
            fig_cta, h_df.index, h_df['cta_longs'], cta_risk_points, cta_opportunity_points,
            risk_label="CTA 多头拥挤风险点", opportunity_label="CTA 空头耗尽机会点",
            opportunity_y=h_df['cta_shorts']
        )
        fig_cta.update_layout(title_text="CTA 量化追踪：多/空头趋势耗尽历史得分", template="plotly_white", height=400)
        add_equity_benchmark_overlay(fig_cta, h_df.index, tab4_qqq, tab4_spy)
        st.plotly_chart(fig_cta, use_container_width=True)

# --- TAB 5 ---
with tab5:
    tab5_qqq, tab5_spy = chart_benchmark_controls("tab5")
    if not quant_data["error"] and "df_hist" in quant_data:
        h_df = quant_data["df_hist"]
        c6_col1, c6_col2 = st.columns(2)
        
        with c6_col1:
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(x=h_df.index, y=h_df['corr'], name="真实值", line=dict(color="#bdc3c7", width=1)))
            fig_corr.add_trace(go.Scatter(x=h_df.index, y=h_df['corr_fast'], name="EMA5 (快线)", line=dict(color="#e74c3c", width=2)))
            fig_corr.add_trace(go.Scatter(x=h_df.index, y=h_df['corr_slow'], name="EMA21 (慢线)", line=dict(color="#2c3e50", width=2)))
            corr_golden = (h_df['corr_fast'].shift(1) <= h_df['corr_slow'].shift(1)) & (h_df['corr_fast'] > h_df['corr_slow'])
            corr_death = (h_df['corr_fast'].shift(1) >= h_df['corr_slow'].shift(1)) & (h_df['corr_fast'] < h_df['corr_slow'])
            add_risk_opportunity_markers(
                fig_corr, h_df.index, h_df['corr'], corr_golden, corr_death,
                risk_label="相关性金叉风险点", opportunity_label="相关性死叉机会点"
            )
            fig_corr.update_layout(title_text="CBOE COR1M 相关性快慢线 (死叉形成释放见底信号)", template="plotly_white", height=380)
            add_equity_benchmark_overlay(fig_corr, h_df.index, tab5_qqq, tab5_spy)
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with c6_col2:
            fig_disp = go.Figure()
            fig_disp.add_trace(go.Scatter(x=h_df.index, y=h_df['dspx'], name="真实值", line=dict(color="#bdc3c7", width=1)))
            fig_disp.add_trace(go.Scatter(x=h_df.index, y=h_df['dsp_fast'], name="EMA5 (快线)", line=dict(color="#2ecc71", width=2)))
            fig_disp.add_trace(go.Scatter(x=h_df.index, y=h_df['dsp_slow'], name="EMA21 (慢线)", line=dict(color="#34495e", width=2)))
            dsp_golden = (h_df['dsp_fast'].shift(1) <= h_df['dsp_slow'].shift(1)) & (h_df['dsp_fast'] > h_df['dsp_slow'])
            dsp_death = (h_df['dsp_fast'].shift(1) >= h_df['dsp_slow'].shift(1)) & (h_df['dsp_fast'] < h_df['dsp_slow'])
            add_risk_opportunity_markers(
                fig_disp, h_df.index, h_df['dspx'], dsp_golden, dsp_death,
                risk_label="离散度金叉风险点", opportunity_label="离散度死叉修复点"
            )
            fig_disp.update_layout(title_text="CBOE DSPX 离散度快慢线 (高位金叉发散警惕拉巨头出货)", template="plotly_white", height=380)
            add_equity_benchmark_overlay(fig_disp, h_df.index, tab5_qqq, tab5_spy)
            st.plotly_chart(fig_disp, use_container_width=True)

# --- TAB 6 ---
with tab6:
    tab6_qqq, tab6_spy = chart_benchmark_controls("tab6")
    if not vxn_vix_data["error"] and "df_hist" in vxn_vix_data:
        vx_df = vxn_vix_data["df_hist"]
        spread_golden = (vx_df['Spread_Fast'].shift(1) <= vx_df['Spread_Slow'].shift(1)) & (vx_df['Spread_Fast'] > vx_df['Spread_Slow'])
        spread_death = (vx_df['Spread_Fast'].shift(1) >= vx_df['Spread_Slow'].shift(1)) & (vx_df['Spread_Fast'] < vx_df['Spread_Slow'])
        vx_risk_state = (
            ((vx_df['Spread'] > 7.5) | (vx_df['Ratio'] > 1.35)) & (vx_df['Spread_Fast'] > vx_df['Spread_Slow'])
        ) | (((vx_df['Spread'] < 3.0) | (vx_df['Ratio'] < 1.10)) & spread_golden)
        # Keep this technology-volatility panel sparse: show only fresh entries
        # and a separate marker when the spread/ratio becomes truly extreme.
        vx_risk_points = state_entry_points(vx_risk_state, mark_initial_active=True)
        vx_extreme_points = state_entry_points(
            ((vx_df['Spread'] > 9.0) | (vx_df['Ratio'] > 1.40))
            & (vx_df['Spread_Fast'] > vx_df['Spread_Slow']), mark_initial_active=True
        )
        vx_opportunity_state = spread_death & (vx_df['Spread'].rolling(5).max() > 8.0) & (vx_df['^VIX'] < 35.0)
        vx_opportunity_points = state_entry_points(vx_opportunity_state, mark_initial_active=True)
        c7_col1, c7_col2 = st.columns(2)
        
        with c7_col1:
            fig_vx_spread = go.Figure()
            fig_vx_spread.add_trace(go.Scatter(x=vx_df.index, y=vx_df['Spread'], name="真实剪刀差 (VXN - VIX)", line=dict(color='#bdc3c7', width=1)))
            fig_vx_spread.add_trace(go.Scatter(x=vx_df.index, y=vx_df['Spread_Fast'], name="EMA5 (微观快线)", line=dict(color='#e74c3c', width=2)))
            fig_vx_spread.add_trace(go.Scatter(x=vx_df.index, y=vx_df['Spread_Slow'], name="EMA21 (趋势慢线)", line=dict(color='#2c3e50', width=2)))
            add_risk_opportunity_markers(
                fig_vx_spread, vx_df.index, vx_df['Spread'], vx_risk_points, vx_opportunity_points,
                risk_label="进入科技波动风险区", opportunity_label="科技波动修复机会"
            )
            if vx_extreme_points.any():
                fig_vx_spread.add_trace(go.Scatter(
                    x=vx_df.index[vx_extreme_points.to_numpy()], y=vx_df['Spread'].iloc[vx_extreme_points.to_numpy()],
                    mode='markers', name='科技波动极端风险', marker=dict(symbol='x', size=11, color='#7b241c'),
                    hovertemplate='%{x|%Y-%m-%d}<br>科技波动极端风险<extra></extra>'
                ))
            fig_vx_spread.update_layout(
                title_text="VXN - VIX 波动率剪刀差收敛雷达 (高位死叉确立科技股黄金买点)", 
                template="plotly_white", 
                height=380
            )
            add_equity_benchmark_overlay(fig_vx_spread, vx_df.index, tab6_qqq, tab6_spy)
            st.plotly_chart(fig_vx_spread, use_container_width=True)
            
        with c7_col2:
            fig_vx_ratio = go.Figure()
            fig_vx_ratio.add_trace(go.Scatter(x=vx_df.index, y=vx_df['Ratio'], name="VXN / VIX 比率", line=dict(color='#9b59b6', width=2, dash='dash')))
            fig_vx_ratio.add_hline(y=1.35, line_dash="dash", line_color="#e74c3c", annotation_text="极端过热线 (1.35)")
            fig_vx_ratio.add_hline(y=1.10, line_dash="dash", line_color="#2ecc71", annotation_text="极限自满线 (1.10)")
            add_risk_opportunity_markers(
                fig_vx_ratio, vx_df.index, vx_df['Ratio'], vx_risk_points, vx_opportunity_points,
                risk_label="进入科技波动风险区", opportunity_label="科技波动修复机会"
            )
            if vx_extreme_points.any():
                fig_vx_ratio.add_trace(go.Scatter(
                    x=vx_df.index[vx_extreme_points.to_numpy()], y=vx_df['Ratio'].iloc[vx_extreme_points.to_numpy()],
                    mode='markers', name='科技波动极端风险', marker=dict(symbol='x', size=11, color='#7b241c'),
                    hovertemplate='%{x|%Y-%m-%d}<br>科技波动极端风险<extra></extra>'
                ))
            fig_vx_ratio.update_layout(
                title_text="VXN / VIX 情绪乘数溢价区间 (追踪科技股相对大盘的拥挤度)", 
                template="plotly_white", 
                height=380
            )
            add_equity_benchmark_overlay(fig_vx_ratio, vx_df.index, tab6_qqq, tab6_spy)
            st.plotly_chart(fig_vx_ratio, use_container_width=True)
    else:
        st.warning("⚠️ VXN-VIX 科技前哨模块数据未激活或加载失败。")

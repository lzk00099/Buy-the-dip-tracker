"""Timestamp-checked market inputs and append-only model observations.

Quotes are observations, not executions. No synthetic fallback is permitted.
The web process has only read access to the durable GitHub data branch.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import requests
import streamlit as st

ET = ZoneInfo('America/New_York')
VERSION = '3.0-observed-20260930'
ETF = ['QQQ', 'SPY', 'IWM', 'RSP', 'HYG', 'LQD']
INDICES = ['^NDX', '^GSPC', '^VIX', '^VIX3M', '^VXN', '^VVIX', '^MOVE']
DATA_URL = 'https://raw.githubusercontent.com/lzk00099/Buy-the-dip-tracker/market-data'


def config(name, default=''):
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


@lru_cache(maxsize=32)
def schedule(day):
    day = pd.Timestamp(day)
    return mcal.get_calendar('NYSE').schedule(
        start_date=day - pd.Timedelta(days=15), end_date=day + pd.Timedelta(days=1))


def market_session(now_et=None):
    now = pd.Timestamp(now_et or dt.datetime.now(ET))
    now = now.tz_localize(ET) if now.tzinfo is None else now.tz_convert(ET)
    sched = schedule(now.date().isoformat())
    today = pd.Timestamp(now.date())
    row = sched.loc[today] if today in sched.index else None
    state = 'closed'
    if row is not None:
        opening, closing = row.market_open.tz_convert(ET), row.market_close.tz_convert(ET)
        if now.normalize() + pd.Timedelta(hours=4) <= now < opening:
            state = 'premarket'
        elif opening <= now < closing:
            state = 'regular'
        elif closing <= now < now.normalize() + pd.Timedelta(hours=20):
            state = 'postmarket'
    completed = sched.loc[sched.market_close <= now.tz_convert('UTC')]
    last_close = completed.index[-1].date().isoformat()
    trading_date = today.date().isoformat() if row is not None else last_close
    return {'session': state, 'active': state != 'closed', 'now_et': now,
            'label': {'closed':'休市','premarket':'盘前','regular':'盘中','postmarket':'盘后'}[state],
            'trading_date': trading_date, 'last_completed': last_close,
            'close_at': row.market_close if row is not None else None}


def safe_error(exc):
    # requests HTTPError can include credentials in its URL. Never display it.
    if isinstance(exc, requests.HTTPError):
        return f'HTTP {exc.response.status_code}'
    if isinstance(exc, requests.Timeout):
        return '请求超时'
    if isinstance(exc, requests.ConnectionError):
        return '连接失败'
    return re.sub(r'(?i)(apikey|api_key|token|secret)=([^&\s]+)', r'\1=[redacted]', str(exc))[:200]


def json_get(url, **kwargs):
    response = requests.get(url, timeout=(4, 10), **kwargs)
    response.raise_for_status()
    return response.json()


def validate_quotes(frame, now=None, max_age=35):
    """Validate EVERY row, including disk/remote caches and paid snapshots."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=['price','timestamp','source'])
    result = frame.copy()
    result['timestamp'] = pd.to_datetime(result['timestamp'], utc=True, errors='coerce')
    result['price'] = pd.to_numeric(result['price'], errors='coerce')
    now = pd.Timestamp(now or dt.datetime.now(dt.timezone.utc)).tz_convert('UTC')
    age = (now - result.timestamp).dt.total_seconds() / 60
    result = result.loc[(age >= -1) & (age <= max_age) & (result.price > 0)
                        & result.price.map(lambda x: pd.notna(x) and math.isfinite(x))
                        & result.timestamp.dt.tz_convert(ET).dt.date.eq(now.tz_convert(ET).date())]
    return result.sort_values('timestamp').loc[lambda x: ~x.index.duplicated(keep='last')]


def yahoo_quote(symbol):
    # Direct chart parsing avoids yfinance cookie/crumb/parser failures. It is
    # still the same vendor, so a 429 stops this route; it is not an SLA.
    data = json_get('https://query1.finance.yahoo.com/v8/finance/chart/' + quote(symbol, safe=''),
                    params={'range':'1d','interval':'5m','includePrePost':'true'},
                    headers={'User-Agent':'Mozilla/5.0'})
    item = (data.get('chart', {}).get('result') or [None])[0]
    if not item:
        raise ValueError('Yahoo 无报价')
    bars = item['indicators']['quote'][0]['close']
    valid = [(ts, p) for ts, p in zip(item.get('timestamp', []), bars) if p is not None]
    if not valid:
        raise ValueError('Yahoo 分钟线为空')
    timestamp, price = valid[-1]
    return {'symbol':symbol,'price':price,'timestamp':pd.to_datetime(timestamp, unit='s', utc=True),
            'source':'Yahoo 5分钟线（供应商可能延迟）'}


def nasdaq_quote(symbol):
    data = json_get('https://api.nasdaq.com/api/quote/' + ('NDX' if symbol == '^NDX' else symbol) + '/info',
                    params={'assetclass':'index' if symbol == '^NDX' else 'etf'},
                    headers={'User-Agent':'Mozilla/5.0','Accept':'application/json,text/plain,*/*'})
    item = (data.get('data') or {}).get('primaryData') or {}
    raw_time = item.get('lastTradeTimestamp', '').replace('DATA AS OF ', '')
    if not re.search(r'\b(ET|EDT|EST)\b', raw_time):
        raise ValueError('Nasdaq 缺少可验证的美东行情时间')
    stamp = pd.Timestamp(re.sub(r'\b(ET|EDT|EST)\b', '', raw_time).strip()).tz_localize(ET)
    price = float(str(item.get('lastSalePrice','')).replace('$','').replace(',',''))
    return {'symbol':symbol,'price':price,'timestamp':stamp.tz_convert('UTC'),
            'source':'Nasdaq 公开报价（可能延迟）'}


def alpaca_quotes():
    key, secret = config('ALPACA_API_KEY'), config('ALPACA_API_SECRET')
    if not key or not secret:
        return []
    headers = {'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret}
    feed = config('ALPACA_FEED', 'iex')
    try:
        data = json_get('https://data.alpaca.markets/v2/stocks/bars/latest',
                        params={'symbols':','.join(ETF),'feed':feed}, headers=headers)
        bars = data.get('bars') or {}
        return [{'symbol':s,'price':b['c'],'timestamp':b['t'],
                 'source':f'Alpaca {feed}（IEX为单交易所）' if feed == 'iex' else f'Alpaca {feed}'}
                for s,b in bars.items()]
    except requests.HTTPError as exc:
        if exc.response.status_code not in (401,403):
            raise
        raise ValueError('Alpaca 凭据或数据权限不可用；已转备用源') from None


def massive_quotes():
    key = config('MASSIVE_API_KEY')
    enabled = config('ENABLE_PAID_VOLATILITY_SOURCE', 'false').lower() in ('true','1','yes')
    if not enabled or not key:
        return []
    mapping = {'^NDX':'I:NDX','^GSPC':'I:SPX','^VIX':'I:VIX','^VIX3M':'I:VIX3M','^VXN':'I:VXN','^VVIX':'I:VVIX'}
    for pair in config('MASSIVE_VOL_TICKERS').split(','):
        if '=' in pair:
            a,b = pair.split('=',1)
            mapping[a.strip()] = b.strip()
    data = json_get('https://api.massive.com/v3/snapshot/indices',
                    params={'ticker.any_of':','.join(mapping.values()),'limit':250},
                    headers={'Authorization':f'Bearer {key}'})
    reverse = {v:k for k,v in mapping.items()}
    return [{'symbol':reverse[x['ticker']], 'price':x['value'],
             'timestamp':pd.to_datetime(int(x['last_updated']),unit='ns',utc=True),
             'source':'Massive ' + x['timeframe']}
            for x in data.get('results',[]) if x.get('ticker') in reverse
            and not x.get('error') and x.get('timeframe') in ('REAL-TIME','DELAYED')
            and x.get('value') is not None and x.get('last_updated')]


@st.cache_data(ttl=300, show_spinner=False)
def remote_json(name):
    try:
        return json_get(DATA_URL + '/' + name)
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def remote_frame(name):
    try:
        response = requests.get(DATA_URL + '/cache/' + name, timeout=(4,10))
        response.raise_for_status()
        result = pd.read_csv(io.StringIO(response.text),index_col=0,parse_dates=True)
        result.attrs['data_source'] = '后台持久缓存（以行情日期为准）'
        return result
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_quotes(session_name, trading_date):
    rows, errors = [], []
    for provider in (alpaca_quotes, massive_quotes):
        try:
            rows.extend(provider())
        except Exception as exc:
            errors.append(provider.__name__ + ': ' + safe_error(exc))
    def checked(rows):
        return validate_quotes(pd.DataFrame(rows).set_index('symbol')) if rows else validate_quotes(None)
    accepted = checked(rows)
    symbols = ETF + INDICES
    missing = [s for s in symbols if s not in accepted.index]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(yahoo_quote,s):s for s in missing}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(f'{futures[future]} Yahoo: {safe_error(exc)}')
    accepted = checked(rows)
    missing = [s for s in ETF + ['^NDX'] if s not in accepted.index]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(nasdaq_quote,s):s for s in missing}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(f'{futures[future]} Nasdaq: {safe_error(exc)}')
    # A GitHub-runner observation can bridge a Streamlit-specific outage.
    if not os.getenv('SENTINEL_COLLECTOR'):
        remote = remote_json('latest_quotes.json') or {}
        rows.extend(remote.get('quotes', []))
    result = checked(rows)
    result.attrs['provider_errors'] = errors
    return result


def overlay(daily, quotes, symbols, require_all=False):
    if daily is None or daily.empty:
        return daily
    result = daily.copy()
    result.attrs = dict(daily.attrs)
    for key in ('intraday_asof','intraday_symbols','quote_sources','quote_age_minutes'):
        result.attrs.pop(key,None)
    result.attrs['quote_mode'] = 'daily_fallback'
    valid = validate_quotes(quotes)
    available = [s for s in symbols if s in valid.index]
    if not available or (require_all and len(available) != len(symbols)):
        return result
    selected = valid.loc[available]
    # Ratios must not combine an old price and a new price.
    if require_all and (selected.timestamp.max()-selected.timestamp.min()).total_seconds() > 15*60:
        return result
    for symbol,row in selected.iterrows():
        date = pd.Timestamp(row.timestamp.tz_convert(ET).date())
        result.loc[date,symbol] = row.price
    # Do NOT ffill missing columns into today's row and relabel them as live.
    earliest = selected.timestamp.min()
    result.attrs.update(quote_mode=market_session()['session'], intraday_asof=earliest.isoformat(),
                        quote_age_minutes=round((pd.Timestamp.now(tz='UTC')-earliest).total_seconds()/60,1),
                        quote_sources=', '.join(sorted(set(selected.source))), intraday_symbols=available)
    return result.sort_index()


def normalize_dix(frame):
    result = frame.copy()
    result.columns = result.columns.str.strip().str.lower()
    if not {'date','dix','gex'}.issubset(result):
        raise ValueError('DIX CSV 缺少必需列')
    result['date'] = pd.to_datetime(result.date, errors='coerce')
    for col in ('dix','gex'):
        result[col] = pd.to_numeric(result[col], errors='coerce')
    result['dix'] = result.dix.where(result.dix.abs() > 1, result.dix * 100)
    result = result.dropna(subset=['date','dix','gex']).sort_values('date').drop_duplicates('date',keep='last')
    if result.empty or not result.dix.between(0,100).all():
        raise ValueError('DIX 有效数据不足或单位异常')
    return result


def technical_levels(close, now=None):
    """Transparent PRICE proxies, never represented as dealer/CTA positions."""
    info = market_session(now)
    series = close.dropna().sort_index()
    series = series.loc[series.index <= pd.Timestamp(info['last_completed'])]
    if len(series) < 126:
        return None
    last = float(series.iloc[-1])
    # Price at which a new close would equal its N-day SMA: mean of prior N-1 closes.
    triggers = {str(n):float(series.tail(n-1).mean()) for n in (21,63,126)}
    low20,low63 = float(series.tail(20).min()),float(series.tail(63).min())
    return {'asof':series.index[-1].date().isoformat(),'last_close':last,
            'trend_triggers':triggers,'support_low':min(low20,low63),
            'support_high':max(low20,low63),'kind':'价格模型估算；非机构CTA或实际买盘'}


def verified_levels(path, now=None, instrument='NDX'):
    """Only checked, dated reports for this exact spot instrument may overlay."""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        today = pd.Timestamp(now or dt.datetime.now(ET)).date()
        if data.get('instrument') != instrument or data.get('basis') != 'spot' or not data.get('verified'):
            return None
        if not data.get('source_url','').startswith('https://') or not data.get('publisher'):
            return None
        asof, expires = dt.date.fromisoformat(data['asof']), dt.date.fromisoformat(data['expires'])
        if not asof <= today <= expires or (expires-asof).days > 7:
            return None
        levels = data.get('levels',[])
        if not levels or any(not math.isfinite(float(x['value'])) or float(x['value']) <= 0 for x in levels):
            return None
        return data
    except (OSError,ValueError,KeyError,TypeError):
        return None


def jsonable(value):
    if isinstance(value,pd.DataFrame):
        return {'rows':jsonable(value.tail(2).reset_index().to_dict('records')), 'metadata':jsonable(value.attrs)}
    if isinstance(value,pd.Series):
        return jsonable(value.tail(2).to_dict())
    if isinstance(value,dict):
        return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value,(pd.Timestamp,dt.date,dt.datetime)):
        return value.isoformat()
    if hasattr(value,'item'):
        return jsonable(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value


def data_dir():
    root = Path(config('SENTINEL_DATA_DIR', '.sentinel-data'))
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_observation(scores, components, inputs, quality_ok, input_date, now=None):
    """SQLite serializes concurrent page sessions; inputs+outputs are append-only."""
    observed = pd.Timestamp(now).tz_convert('UTC') if now is not None else pd.Timestamp.now(tz='UTC')
    info = market_session(observed)
    # Use the observation's exchange day, NEVER backdate tomorrow's crypto or
    # revised inputs into yesterday's score. Weekend observations are audit-only.
    observed_day = observed.tz_convert(ET).date().isoformat()
    trade_date = observed_day if observed_day == info['trading_date'] else None
    payload = jsonable({'version':VERSION,'scores':scores,'components':components,'inputs':inputs})
    identity = dict(payload, observed_day=observed_day, quality_ok=bool(quality_ok))
    digest = hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    record = dict(payload, id=digest, observed_at=observed.isoformat(), trade_date=trade_date,
                  session=info['session'], quality_ok=bool(quality_ok), input_trade_date=input_date,
                  origin='scheduled_observation' if os.getenv('SENTINEL_COLLECTOR') else 'page_observation')
    with sqlite3.connect(data_dir()/'observations.sqlite',timeout=20) as db:
        db.execute('CREATE TABLE IF NOT EXISTS observations (id TEXT PRIMARY KEY, observed_at TEXT, payload TEXT)')
        db.execute('INSERT OR IGNORE INTO observations VALUES (?,?,?)',
                   (digest,record['observed_at'],json.dumps(record,ensure_ascii=False,allow_nan=False)))
        rows = db.execute('SELECT payload FROM observations ORDER BY observed_at').fetchall()
    return record,[json.loads(x[0]) for x in rows]


def daily_records(records):
    """One actual valid observation per trading day; no interpolation/replay."""
    chosen = {}
    for row in sorted(records,key=lambda x:x['observed_at']):
        if not row.get('quality_ok') or not row.get('trade_date') or row.get('version') != VERSION:
            continue
        chosen[row['trade_date']] = row
    return [chosen[k] for k in sorted(chosen)]


def history_frame(records):
    rows = []
    for record in daily_records(records):
        rows.append(dict(timestamp=pd.Timestamp(record['trade_date']).tz_localize(ET),
                         **record['scores'], origin=record['origin'], observed_at=record['observed_at'],
                         version=record['version']))
    return pd.DataFrame(rows)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(jsonable(value),ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    tmp.replace(path)

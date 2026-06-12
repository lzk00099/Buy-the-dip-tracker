import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime
import os
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        padding: 15px;
        border-radius: 8px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border-left: 5px solid #cccccc;
    }
    .status-bottom-active { border-left-color: #2ecc71; background-color: #f4fbf7; }
    .status-top-active { border-left-color: #e74c3c; background-color: #fdf5f5; }
    .status-neutral { border-left-color: #3498db; background-color: #f0f7fc; }
    
    .badge-bottom { background-color: #2ecc71; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
    .badge-top { background-color: #e74c3c; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
    .badge-info { background-color: #3498db; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 数据获取与处理模块 (Data Pipeline)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_vix_data():
    """开关2：获取VIX期限结构数据"""
    try:
        tickers = yf.Tickers('^VIX ^VIX3M')
        hist = tickers.history(period='3mo')  
        if not hist.empty and 'Close' in hist.columns:
            close_data = hist['Close'].ffill().copy()
            close_data['Ratio'] = close_data['^VIX3M'] / close_data['^VIX']
            
            vix = close_data['^VIX'].iloc[-1]
            vix3m = close_data['^VIX3M'].iloc[-1]
            
            if np.isnan(vix):
                vix = close_data['^VIX'].dropna().iloc[-1]
            if np.isnan(vix3m):
                vix3m = close_data['^VIX3M'].dropna().iloc[-1]
                
            ratio = vix3m / vix
            bottom_active = ratio > 1.0  
            top_active = vix < 12.0 or ratio > 1.25  
            
            return {
                "vix": round(vix, 2), "vix3m": round(vix3m, 2), "ratio": round(ratio, 3),
                "bottom_active": bottom_active, "top_active": top_active, "error": False,
                "df": close_data.tail(60)
            }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False}
    return {"error": True, "msg": "No data", "bottom_active": False, "top_active": False}

@st.cache_data(ttl=1800)
def fetch_crypto_signals():
    """开关3：获取加密货币资产永续合约资金费率与OI趋势"""
    try:
        fr_url = "https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP"
        fr_res = requests.get(fr_url, timeout=5).json()
        if fr_res.get("code") != "0":
            return {"error": True, "msg": "OKX API 异常", "bottom_active": False, "top_active": False}
            
        funding_rate = float(fr_res['data'][0]['fundingRate']) * 100
        
        oi_url = "https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP"
        oi_res = requests.get(oi_url, timeout=5).json()
        if oi_res.get("code") != "0":
            return {"error": True, "msg": "获取 OI 数据异常", "bottom_active": False, "top_active": False}
            
        open_interest = float(oi_res['data'][0]['oiCcy'])
        
        bottom_active = funding_rate >= 0.0  
        top_active = funding_rate >= 0.01  # 💡 低迷期过热线降至 0.01%
        
        hist_df = None
        try:
            hist_url = "https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=60"
            hist_res = requests.get(hist_url, timeout=5).json()
            if hist_res.get("code") == "0":
                hist_data = hist_res['data']
                hist_df = pd.DataFrame(hist_data)
                hist_df['fundingTime'] = pd.to_datetime(hist_df['fundingTime'].astype(float), unit='ms')
                hist_df['fundingRate'] = hist_df['fundingRate'].astype(float) * 100
                hist_df = hist_df.sort_values('fundingTime')
        except:
            pass  
        
        return {
            "funding_rate": f"{funding_rate:.4f}%", "oi": f"{open_interest:,.0f}",
            "bottom_active": bottom_active, "top_active": top_active, "error": False,
            "hist_df": hist_df
        }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False}

@st.cache_data(ttl=3600)
def fetch_squeezemetrics_data():
    """开关1 & 开关5：获取 SqueezeMetrics 的 DIX 和 GEX 数据"""
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
                
                return {
                    "dix": round(dix_val, 2), "gex": int(gex_val),
                    "dix_bottom_active": dix_val >= 45.0, "dix_top_active": dix_val < 40.0,
                    "gex_bottom_active": gex_val > 0, "gex_top_active": gex_val < 0,
                    "error": False, "df": df.tail(100), "is_mock": False
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
        "dix_bottom_active": latest['dix'] >= 45.0, "dix_top_active": latest['dix'] < 40.0,
        "gex_bottom_active": latest['gex'] > 0, "gex_top_active": latest['gex'] < 0,
        "error": False, "df": mock_df, "is_mock": True
    }

@st.cache_data(ttl=14400)
def fetch_cboe_official_history(symbol):
    """从 CBOE 官方 CDN 直接拉取历史完整序列"""
    try:
        url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        df = pd.read_csv(url)
        df.columns = df.columns.str.lower()
        date_col = 'trade_date' if 'trade_date' in df.columns else 'date'
        df['date'] = pd.to_datetime(df[date_col])
        df.set_index('date', inplace=True)
        df = df.sort_index()
        return df['close']
    except Exception:
        return pd.Series()

@st.cache_data(ttl=3600)
def calculate_quant_and_breadth_signals():
    """混合对齐引擎 —— 💡 重构开关6线序排阵及大/小级别极端状态推演"""
    try:
        yf_tickers = ['QQQ', 'SPY', 'IWM', 'RSP', '^COR1M', '^DSPX']
        yf_data = yf.download(yf_tickers, period='1y', progress=False)['Close']
        yf_data = yf_data.ffill()
        latest = yf_data.iloc[-1]
        
        data = yf_data[['QQQ', 'SPY', 'IWM', 'RSP']].copy()
        
        corr_official = fetch_cboe_official_history('COR1M')
        dspx_official = fetch_cboe_official_history('DSPX')
        
        if not corr_official.empty:
            corr_series = corr_official.reindex(data.index)
            if np.isnan(corr_series.iloc[-1]) or corr_series.tail(10).max() == corr_series.tail(10).min():
                corr_series.iloc[-1] = latest.get('^COR1M', corr_series.dropna().iloc[-1] if not corr_series.dropna().empty else 17.8)
            corr_series = corr_series.ffill()
        else:
            corr_series = yf_data['^COR1M']
            
        if not dspx_official.empty:
            dspx_series = dspx_official.reindex(data.index)
            if np.isnan(dspx_series.iloc[-1]):
                dspx_series.iloc[-1] = latest.get('^DSPX', dspx_series.dropna().iloc[-1] if not dspx_series.dropna().empty else 40.0)
            dspx_series = dspx_series.ffill()
        else:
            dspx_series = yf_data['^DSPX']

        # 开关4：CTA 动向矩阵历史序列计算
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

        # ---------------------------------------------------------------------
        # 💡 【重大更新】开关6：快/慢/现货相对位置及大级别状态计算
        # ---------------------------------------------------------------------
        corr_is_broken = corr_series.tail(10).max() == corr_series.tail(10).min()
        corr_fast = corr_series.ewm(span=5, adjust=False).mean()
        corr_slow = corr_series.ewm(span=21, adjust=False).mean()
        corr_q75 = corr_series.rolling(126).quantile(0.75) 
        
        dsp_fast = dspx_series.ewm(span=5, adjust=False).mean()
        dsp_slow = dspx_series.ewm(span=21, adjust=False).mean()
        dspx_q75 = dspx_series.rolling(126).quantile(0.75)

        c_val, c_f, c_s = corr_series.iloc[-1], corr_fast.iloc[-1], corr_slow.iloc[-1]
        d_val, d_f, d_s = dspx_series.iloc[-1], dsp_fast.iloc[-1], dsp_slow.iloc[-1]
        
        # 向量化相对排列阵法计算
        def determine_line_order(val, fast, slow):
            labels = [("现货", val), ("快线", fast), ("慢线", slow)]
            labels.sort(key=lambda x: x[1], reverse=True)
            return " > ".join([x[0] for x in labels])

        corr_pos_str = determine_line_order(c_val, c_f, c_s)
        dsp_pos_str = determine_line_order(d_val, d_f, d_s)

        # 重新根据大/小级别回调与底部定义核心驱动状态
        market_high = data['SPY'].iloc[-1] > data['SPY'].rolling(50).mean().iloc[-1]
        
        # 底部判定：隐含相关性拐点 (快线 < 慢线)
        corr_turning_down = c_f < c_s
        corr_large_bottom = corr_turning_down and (corr_slow.tail(15).max() > corr_q75.iloc[-1])
        corr_small_bottom = corr_turning_down and not corr_large_bottom
        
        # 回调风险判定：个股离散度爆发 (快线 > 慢线)
        disp_breaking_up = d_f > d_s
        disp_large_correction = disp_breaking_up and (dspx_series.tail(15).max() > dspx_q75.iloc[-1]) and market_high
        disp_small_correction = disp_breaking_up and not disp_large_correction

        if corr_large_bottom:
            corr_tier = "🚀 大级别金身底确立 (全市场高热踩踏无差别释放，黄金左侧买点)"
        elif corr_small_bottom:
            corr_tier = "🌤️ 小级别波段底形成 (常规回踩动能衰竭，个股即将开启修复)"
        else:
            corr_tier = "🔘 状态平衡 / 当前未触及明确见底拐点"
            
        if disp_large_correction:
            disp_tier = "🚨 大级别回调危机预警 (极端结构性撕裂，高位抱团股极易引发踩踏)"
        elif disp_small_correction:
            disp_tier = "⚠️ 小级别结构回调预警 (多头边际买盘降温，局部出现获利回吐)"
        else:
            disp_tier = "🔘 状态安全 / 广度指标未见异常分裂隐患"

        corr_bottom_active = corr_large_bottom or corr_small_bottom
        breadth_top_active = disp_large_correction or disp_small_correction
        
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
            "corr_pos": corr_pos_str,
            "dsp_pos": dsp_pos_str,
            "corr_tier": corr_tier,
            "disp_tier": disp_tier,
            "spy_rsp_ratio": round(spy_rsp_ratio, 4),
            "cta_bottom_active": cta_bottom_active,
            "cta_top_active": cta_top_active,
            "corr_bottom_active": corr_bottom_active,
            "breadth_top_active": breadth_top_active,
            "corr_is_broken": corr_is_broken,
            "df_hist": df_hist.tail(60)
        }
    except Exception as e:
        return {
            "error": True, "msg": str(e), 
            "cta_bottom_active": False, "cta_top_active": False,
            "corr_bottom_active": False, "breadth_top_active": False
        }

# -----------------------------------------------------------------------------
# 3. 业务决策逻辑组装
# -----------------------------------------------------------------------------
vix_data = fetch_vix_data()
crypto_data = fetch_crypto_signals()
sm_data = fetch_squeezemetrics_data()
quant_data = calculate_quant_and_breadth_signals()

switches = [
    {
        "id": 1,
        "name": "做市商 Gamma 净敞口 (核心护盘/闪崩开关)",
        "bottom_active": sm_data["gex_bottom_active"] if not sm_data["error"] else False,
        "top_active": sm_data["gex_top_active"] if not sm_data["error"] else False,
        "value": f"GEX 绝对值: {sm_data.get('gex', 0):,}" if not sm_data["error"] else "数据源异常",
        "source": "SqueezeMetrics (Proxy for SPX/NDX)",
        "desc_bottom": "Gamma由负转正。做市商从‘顺势砸盘’转为‘逆势稳定市场’，左侧流动性危机解除。",
        "desc_top": "Gamma高位转负。做市商对冲盘变砸盘放大器，大盘极易诱发高位闪崩。",
        "fetched_status": "成功 🟢" if (not sm_data["error"] and not sm_data.get("is_mock", False)) else ("使用兜底数据 🟡" if sm_data.get("is_mock", False) else "抓取失败 🔴")
    },
    {
        "id": 2,
        "name": "VIX 期限结构与情绪指标",
        "bottom_active": vix_data["bottom_active"] if not vix_data["error"] else False,
        "top_active": vix_data["top_active"] if not vix_data["error"] else False,
        "value": f"VIX3M/VIX 比率: {vix_data.get('ratio', 'N/A')} | VIX: {vix_data.get('vix', 'N/A')}",
        "source": "CBOE 波动率曲线",
        "desc_bottom": "结构回到 Contango (>1.0)，短期恐慌高潮褪去，买入对冲保护的资金撤退。",
        "desc_top": "结构过度溢价(>1.25)或VIX跌破12。全市场极度自满，往往是暴风雨前夕。",
        "fetched_status": "成功 🟢" if not vix_data["error"] else "抓取失败 🔴"
    },
    {
        "id": 3,
        "name": "加密离岸高杠杆流动性前哨",
        "bottom_active": crypto_data["bottom_active"] if not crypto_data["error"] else False,
        "top_active": crypto_data["top_active"] if not crypto_data["error"] else False,
        "value": f"资金费率: {crypto_data.get('funding_rate', 'N/A')} | OI: {crypto_data.get('oi', 'N/A')}",
        "source": "OKX 永续合约 API",
        "desc_bottom": "极端倒挂后费率转正 + 低位OI企稳。表明散户割肉盘结束，多头资金左侧重新建仓。",
        "desc_top": "费率突破轻微过热线(>0.01%)且OI创高位，多头杠杆出现超载隐患，易触发洗盘。",
        "fetched_status": "成功 🟢" if not crypto_data["error"] else "抓取失败 🔴"
    },
    {
        "id": 4,
        "name": "CTA 动量矩阵 (系统性抛压/买盘极值监测)",
        "bottom_active": quant_data["cta_bottom_active"] if not quant_data["error"] else False,
        "top_active": quant_data["cta_top_active"] if not quant_data["error"] else False,
        "value": f"当前状态: {quant_data.get('cta_status', 'N/A')}",
        "source": "基于 1M/3M/6M 动量偏离度演算",
        "desc_bottom": "主跌浪贯穿多周期均线且负乖离达极限。量化 CTA 的约800亿无脑跟空抛压面临彻底耗尽。",
        "desc_top": "趋势基金无脑买入的边际力量全面满仓，正乖离达极限，市场缺乏后续增量买家。",
        "fetched_status": "成功 🟢" if not quant_data["error"] else "抓取失败 🔴"
    },
    {
        "id": 5,
        "name": "暗池 DIX 机构资金出没标签",
        "bottom_active": sm_data["dix_bottom_active"] if not sm_data["error"] else False,
        "top_active": sm_data["dix_top_active"] if not sm_data["error"] else False,
        "value": f"DIX 比例: {sm_data.get('dix', 'N/A')}%",
        "source": "SqueezeMetrics 暗池吸筹/派发指数",
        "desc_bottom": "DIX 强力站上 45% 以上。明牌大跌时华尔街主力通过暗池疯狂吃单承接，强力左侧底信号。",
        "desc_top": "DIX 跌破 40% 水平。明牌高位拉升时，主力资金在暗池悄悄分批派发利润，散户接盘。",
        "fetched_status": "成功 🟢" if (not sm_data["error"] and not sm_data.get("is_mock", False)) else ("使用兜底数据 🟡" if sm_data.get("is_mock", False) else "抓取失败 🔴")
    },
    {
        "id": 6,
        "name": "全局隐含相关性拐点与离散度爆发",
        "bottom_active": quant_data["corr_bottom_active"] if not quant_data["error"] else False,
        "top_active": quant_data["breadth_top_active"] if not quant_data["error"] else False,
        # 💡 页面彻底隐藏开关6的原始数值，直接展现计算好的线序阵形
        "value": f"相关性线序: {quant_data.get('corr_pos', 'N/A')} | 离散度线序: {quant_data.get('dsp_pos', 'N/A')}",
        "source": "CBOE COR1M / DSPX 指数矩阵定位",
        "desc_bottom": f"当前状态: {quant_data.get('corr_tier', 'N/A')}",
        "desc_top": f"当前状态: {quant_data.get('disp_tier', 'N/A')}",
        "fetched_status": "成功 🟢" if (not quant_data["error"] and not quant_data.get("corr_is_broken", False)) else ("历史断流 ⚠️" if quant_data.get("corr_is_broken", False) else "抓取失败 🔴")
    }
]

bottom_score = sum([1 for s in switches if s["bottom_active"]])
top_score = sum([1 for s in switches if s["top_active"]])

# -----------------------------------------------------------------------------
# 4. Streamlit UI 界面绘制
# -----------------------------------------------------------------------------
st.title("🛡️ Sentinel 2.0 核心决策系统：大盘底层资金双向雷达")
st.subheader(f"数据实时快照: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("### 📊 综合大盘多空状态与量化应对措施")
c1, c2 = st.columns(2)

with c1:
    st.metric(label="🚀 底部共振激活数 (精准抄底信号)", value=f"{bottom_score} / 6", delta="满足分批左侧建仓" if bottom_score >= 4 else "底部尚未成型")
with c2:
    st.metric(label="🚨 见顶风控激活数 (提前逃顶信号)", value=f"{top_score} / 6", delta="-触发高位逃顶线" if top_score >= 4 else "处于安全健康牛市", delta_color="inverse")

if top_score >= 4:
    status_color = "red"
    action_title = "🚨 【红色暴风雨：触发全面防守逃顶线】"
    action_text = (
        "**底层逻辑演算**：市场表象繁荣，但 CBOE 离散度已形成大级别或小级别的高位撕裂。CTA 多头追随动能面临衰减。"
        "必须**全面收紧个股诊断模型的止盈线**，多头战略转为全面防守、保住利润阶段。"
    )
elif bottom_score >= 4:
    status_color = "green"
    action_title = "🚀 【绿色共振：泥沙俱下恐慌耗尽，触发拐点重仓抄底】"
    action_text = (
        "**底层逻辑演算**：黄金左侧大底或波段反弹底确立！CBOE 隐含相关性确立快线死叉慢线回落，宣告流动性践踏结束。<br><br>"
        "**应对措施**：多头精准抄底模式全面启动。可开启个股诊断模型输入最多 5 只目标 Tickers，利用 Random Forest 和 EV 模型精准卡位。"
    )
else:
    status_color = "orange"
    action_title = "⏳ 【黄色震荡：多空交织，结构分化】"
    action_text = "**应对措施**：无极端抄底或逃顶信号。维持中性仓位，继续针对个股进行常规诊断，捕捉中性环境下的个股错杀红利。"

st.markdown(f"""
<div style="padding:15px; border-radius:8px; border-left: 6px solid {status_color}; background-color:#fafafa; margin-bottom:20px;">
    <h4 style="color:{status_color}; margin:0 0 10px 0;">{action_title}</h4>
    <p style="font-size:11pt; line-height:1.6; color:#333;">{action_text}</p>
</div>
""", unsafe_allow_html=True)


st.markdown("### 🔌 双向资金逻辑开关实时追踪")
cols = st.columns(3)

for i, s in enumerate(switches):
    with cols[i % 3]:
        if s["top_active"]:
            box_class = "status-top-active"
            badge_html = f"<span class='badge-top'>🚨 风险预警</span>"
        elif s["bottom_active"]:
            box_class = "status-bottom-active"
            badge_html = f"<span class='badge-bottom'>🟢 信号激活</span>"
        else:
            box_class = "status-neutral"
            badge_html = f"<span class='badge-info'>⚪ 状态中性</span>"
            
        st.markdown(f"""
        <div class="metric-box {box_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 11pt; font-weight: bold; color: #2c3e50;">开关 {s['id']}: {s['name']}</span>
                {badge_html}
            </div>
            <hr style="margin: 8px 0; border: 0; border-top: 1px solid #eee;">
            <p style="margin: 2px 0; font-size:10pt;"><b>核心底层定位:</b> <span style="font-family: monospace; color:#2980b9; font-weight:bold;">{s['value']}</span></p>
            <p style="margin: 2px 0; font-size:9.5pt;"><b>📡 数据状态:</b> <span>{s['fetched_status']}</span></p>
            <p style="margin: 4px 0; font-size:9.5pt; line-height:1.4;"><b>📈 多头见底边界:</b> <span style="color:#27ae60;">{s['desc_bottom']}</span></p>
            <p style="margin: 4px 0; font-size:9.5pt; line-height:1.4;"><b>📉 空头防守边界:</b> <span style="color:#c0392b;">{s['desc_top']}</span></p>
            <p style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 8.5pt;">🧭 数据来源: {s['source']}</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. 纳指走势雷达引擎
# -----------------------------------------------------------------------------
st.markdown("### 🗺️ 纳指 100 (NDX) 承接区间与走势雷达引擎")

@st.cache_data(ttl=1800)
def fetch_ndx_chart_data():
    df = yf.Ticker('^NDX').history(period='3mo')
    return df

ndx_data = fetch_ndx_chart_data()
if not ndx_data.empty:
    fig_ndx = go.Figure()
    latest_ndx_close = float(ndx_data['Close'].iloc[-1])
    
    fig_ndx.add_trace(go.Scatter(
        x=ndx_data.index, y=ndx_data['Close'], mode='lines', name='NDX 实际走势曲线', line=dict(color='#2980b9', width=2.5)
    ))
    
    fig_ndx.add_hline(
        y=latest_ndx_close, line_dash="solid", line_color="#2c3e50", 
        annotation_text=f"动态实时收盘位 ({latest_ndx_close:,.2f})", annotation_position="top right"
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

tab1_5, tab2, tab3, tab4, tab6 = st.tabs([
    "🐳 开关1 & 5: 机构暗池 DIX & GEX 敞口",
    "📊 开关2: VIX 期限结构趋势", 
    "₿ 开关3: 加密离岸高杠杆费率", 
    "🤖 开关4: CTA 动量极值矩阵", 
    "🔄 开关6: 隐含相关性与离散度快慢线"
])

# --- TAB 1 & 5 ---
with tab1_5:
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
            fig_vix_ratio.add_trace(go.Scatter(x=v_df.index, y=v_df['Ratio'], name="VIX3M / VIX 比率", line=dict(color="#9b59b6", width=2)))
            fig_vix_ratio.add_hline(y=1.0, line_dash="dash", line_color="#2ecc71", annotation_text="Contango 线 (1.0)")
            fig_vix_ratio.add_hline(y=1.25, line_dash="dash", line_color="#e74c3c", annotation_text="极端自满 (1.25)")
            fig_vix_ratio.update_layout(title_text="图表 B: VIX3M / VIX 期限结构比率", template="plotly_white", height=400)
            st.plotly_chart(fig_vix_ratio, use_container_width=True)

# --- TAB 3 ---
with tab3:
    if not crypto_data["error"] and crypto_data.get("hist_df") is not None:
        c_df = crypto_data["hist_df"]
        fig_crypto = go.Figure()
        fig_crypto.add_trace(go.Scatter(x=c_df['fundingTime'], y=c_df['fundingRate'], name="BTC 资金费率 (%)", line=dict(color="#f1c40f", width=2)))
        fig_crypto.add_hline(y=0.0, line_dash="solid", line_color="#7f8c8d")
        fig_crypto.add_hline(y=0.01, line_dash="dash", line_color="#e74c3c", annotation_text="多头超载边界 (>0.01%)")
        fig_crypto.update_layout(title_text="OKX BTC-USDT-SWAP 历史资金费率", template="plotly_white", height=400)
        st.plotly_chart(fig_crypto, use_container_width=True)

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

# --- TAB 6 (快慢线可视化) ---
with tab6:
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

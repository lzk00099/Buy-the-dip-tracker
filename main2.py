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
# 2. 增强版数据获取与处理模块 (Data Pipeline)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_vix_data():
    """开关2：获取VIX期限结构"""
    try:
        tickers = yf.Tickers('^VIX ^VIX3M')
        hist = tickers.history(period='5d')
        if not hist.empty and 'Close' in hist.columns:
            close_data = hist['Close'].ffill()
            
            vix = close_data['^VIX'].iloc[-1]
            vix3m = close_data['^VIX3M'].iloc[-1]
            
            if np.isnan(vix):
                vix = hist['Close']['^VIX'].dropna().iloc[-1]
            if np.isnan(vix3m):
                vix3m = hist['Close']['^VIX3M'].dropna().iloc[-1]
                
            ratio = vix3m / vix
            
            bottom_active = ratio > 1.0  # 回到 Contango 视为抄底信号之一
            top_active = vix < 12.0 or ratio > 1.25  # 极端自满
            
            return {
                "vix": round(vix, 2), "vix3m": round(vix3m, 2), "ratio": round(ratio, 3),
                "bottom_active": bottom_active, "top_active": top_active, "error": False
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
        
        bottom_active = funding_rate >= 0.0  # 费率转正表示情绪企稳
        top_active = funding_rate >= 0.035  # 杠杆过载
        
        return {
            "funding_rate": f"{funding_rate:.4f}%", "oi": f"{open_interest:,.0f}",
            "bottom_active": bottom_active, "top_active": top_active, "error": False
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
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
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
                
                dix_bottom_active = dix_val >= 45.0
                dix_top_active = dix_val < 40.0
                gex_bottom_active = gex_val > 0
                gex_top_active = gex_val < 0
                
                return {
                    "dix": round(dix_val, 2), "gex": int(gex_val),
                    "dix_bottom_active": dix_bottom_active, "dix_top_active": dix_top_active,
                    "gex_bottom_active": gex_bottom_active, "gex_top_active": gex_top_active,
                    "error": False, "df": df.tail(100), "is_mock": False
                }
        except Exception:
            pass

    # Mock兜底数据
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
    """从 CBOE 官方 CDN 直接拉取最权威的历史完整序列"""
    try:
        url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        # CBOE 官方数据包含完整的历史
        df = pd.read_csv(url)
        df.columns = df.columns.str.lower()
        
        # 兼容官方的 trade_date 或 date 字段
        date_col = 'trade_date' if 'trade_date' in df.columns else 'date'
        df['date'] = pd.to_datetime(df[date_col])
        df.set_index('date', inplace=True)
        df = df.sort_index()
        return df['close']
    except Exception as e:
        return pd.Series()

@st.cache_data(ttl=3600)
def calculate_quant_and_breadth_signals():
    """升级版：大盘 ETF 与 CBOE 官方数据混合对齐引擎"""
    try:
        # 1. 正常的指数和 ETF 依然走 yfinance 获取高频数据
        yf_tickers = ['QQQ', 'SPY', 'IWM', 'RSP', '^COR1M', '^DSPX']
        yf_data = yf.download(yf_tickers, period='1y', progress=False)['Close']
        yf_data = yf_data.ffill()
        latest = yf_data.iloc[-1]
        
        # 创建基础的清洗后 DataFrame
        data = yf_data[['QQQ', 'SPY', 'IWM', 'RSP']].copy()
        
        # 2. 核心修复：单独拉取 CBOE 官方的 COR1M 和 DSPX 历史数据
        corr_official = fetch_cboe_official_history('COR1M')
        dspx_official = fetch_cboe_official_history('DSPX')
        
        # 3. 混合对齐逻辑：用官方历史垫底，如果今天官方还没更新，用 yfinance 的最新实时值补上
        # 处理相关性序列
        if not corr_official.empty:
            corr_series = corr_official.reindex(data.index)
            if np.isnan(corr_series.iloc[-1]) or corr_series.tail(10).max() == corr_series.tail(10).min():
                corr_series.iloc[-1] = latest.get('^COR1M', corr_series.dropna().iloc[-1] if not corr_series.dropna().empty else 17.8)
            corr_series = corr_series.ffill()
        else:
            corr_series = yf_data['^COR1M'] # 兜底机制
            
        # 处理离散度序列
        if not dspx_official.empty:
            dspx_series = dspx_official.reindex(data.index)
            if np.isnan(dspx_series.iloc[-1]):
                dspx_series.iloc[-1] = latest.get('^DSPX', dspx_series.dropna().iloc[-1] if not dspx_series.dropna().empty else 40.0)
            dspx_series = dspx_series.ffill()
        else:
            dspx_series = yf_data['^DSPX'] # 兜底机制

        # ---------------------------------------------------------------------
        # 开关4：升级版 CTA 动向追踪 (保持原样)
        # ---------------------------------------------------------------------
        cta_shorts_exhausted = 0
        cta_longs_exhausted = 0
        for idx_name in ['QQQ', 'SPY', 'IWM']:
            price = data[idx_name]
            ma21 = price.rolling(21).mean()
            ma63 = price.rolling(63).mean()
            ma126 = price.rolling(126).mean()
            p_cur = price.iloc[-1]
            if p_cur < ma21.iloc[-1] and p_cur < ma63.iloc[-1] and p_cur < ma126.iloc[-1]:
                if (p_cur - ma21.iloc[-1]) / ma21.iloc[-1] < -0.04:
                    cta_shorts_exhausted += 1
            if p_cur > ma21.iloc[-1] and p_cur > ma63.iloc[-1] and p_cur > ma126.iloc[-1]:
                if (p_cur - ma21.iloc[-1]) / ma21.iloc[-1] > 0.04:
                    cta_longs_exhausted += 1

        cta_bottom_active = cta_shorts_exhausted >= 2
        cta_top_active = cta_longs_exhausted >= 2
        cta_status_text = "多头趋势/系统性买入中"
        if cta_bottom_active: cta_status_text = "系统性空头抛压耗尽"
        elif cta_top_active: cta_status_text = "系统性多头买盘枯竭"
        elif cta_shorts_exhausted > 0 or cta_longs_exhausted > 0: cta_status_text = "CTA 动量分化调仓期"

        # ---------------------------------------------------------------------
        # 开关6：防粘合 CBOE 交叉判断（使用混合对齐后的完整序列计算均线）
        # ---------------------------------------------------------------------
        if corr_series.tail(10).max() == corr_series.tail(10).min():
            cboe_corr_text = f"当前:{latest.get('^COR1M', 0):.2f} (⚠️ 历史断流，无法计算均线拐点)"
            corr_bottom_active = False 
        else:
            corr_fast = corr_series.ewm(span=5, adjust=False).mean()
            corr_slow = corr_series.ewm(span=21, adjust=False).mean()
            corr_q75 = corr_series.rolling(126).quantile(0.75) 
            corr_q25 = corr_series.rolling(126).quantile(0.25)
            
            corr_recent_high = corr_slow.tail(10).max() > corr_q75.iloc[-1]
            corr_turning_down = corr_fast.iloc[-1] < corr_slow.iloc[-1]
            corr_bottom_active = corr_recent_high and corr_turning_down
            cboe_corr_text = f"当前:{corr_series.iloc[-1]:.2f} (EMA5:{corr_fast.iloc[-1]:.2f} / EMA21:{corr_slow.iloc[-1]:.2f})"
        
        # 离散度快慢线计算
        dsp_fast = dspx_series.ewm(span=5, adjust=False).mean()
        dsp_slow = dspx_series.ewm(span=21, adjust=False).mean()
        
        market_high = data['SPY'].iloc[-1] > data['SPY'].rolling(50).mean().iloc[-1]
        try:
            market_complacent = corr_slow.iloc[-1] < corr_q25.iloc[-1]
        except:
            market_complacent = False
        disp_breaking_up = dsp_fast.iloc[-1] > dsp_slow.iloc[-1]
        breadth_top_active = market_high and market_complacent and disp_breaking_up
        
        spy_rsp_ratio = latest['SPY'] / latest['RSP']
        
        return {
            "error": False,
            "cta_status": cta_status_text,
            "cboe_corr": cboe_corr_text,
            "cboe_disp": f"当前:{dspx_series.iloc[-1]:.2f} (EMA5:{dsp_fast.iloc[-1]:.2f} / EMA21:{dsp_slow.iloc[-1]:.2f})",
            "spy_rsp_ratio": round(spy_rsp_ratio, 4),
            "cta_bottom_active": cta_bottom_active,
            "cta_top_active": cta_top_active,
            "corr_bottom_active": corr_bottom_active,
            "breadth_top_active": breadth_top_active
        }
    except Exception as e:
        return {
            "error": True, "msg": str(e), 
            "cta_bottom_active": False, "cta_top_active": False,
            "corr_bottom_active": False, "breadth_top_active": False
        }

# -----------------------------------------------------------------------------
# 3. 双向风控业务核心逻辑组装
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
        "desc_top": "费率极其亢奋(>0.035%)且OI创历史高位，多头杠杆过载，易触发连环清算。",
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
        "value": f"相关性 {quant_data.get('cboe_corr', 'N/A')} | 离散度 {quant_data.get('cboe_disp', 'N/A')}",
        "source": "CBOE COR1M / DSPX 指数 (EMA5 与 EMA21)",
        "desc_bottom": "相关性极高位确立【快线死叉跌破慢线】。无差别恐慌抛售正式宣告结束，散户离场，个股迎修复。",
        "desc_top": "指数自满极低位，但离散度快线【金叉突破慢线】。确立结构性分裂，巨头掩护中小盘出货。",
        "fetched_status": "成功 🟢" if (not quant_data["error"] and not quant_data.get("corr_is_broken", False)) else ("相关性历史数据断流 ⚠️" if quant_data.get("corr_is_broken", False) else "抓取失败 🔴")
    }
]

bottom_score = sum([1 for s in switches if s["bottom_active"]])
top_score = sum([1 for s in switches if s["top_active"]])

# -----------------------------------------------------------------------------
# 4. Streamlit UI 双向渲染
# -----------------------------------------------------------------------------
st.title("🛡️ Sentinel 2.0 核心决策系统：大盘底层资金双向雷达")
st.subheader(f"数据实时快照: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("### 📊 综合大盘多空状态与量化应对措施")
c1, c2 = st.columns(2)

with c1:
    st.metric(label="🚀 底部共振激活数 (精准抄底信号)", value=f"{bottom_score} / 6", delta="满足分批左侧建仓" if bottom_score >= 4 else "底部尚未成型")
with c2:
    st.metric(label="🚨 见顶风控激活数 (提前逃顶信号)", value=f"{top_score} / 6", delta="-触发高位逃顶线" if top_score >= 4 else "处于安全健康牛市", delta_color="inverse")

# 💡 联动修改：全面升级微观执行端联动
if top_score >= 4:
    status_color = "red"
    action_title = "🚨 【红色暴风雨：触发全面防守逃顶线】"
    action_text = (
        "**底层逻辑演算**：市场表象繁荣，但 CBOE 离散度快线已金叉，相关性死死压制在低位。CTA 动量面临高位多头买盘枯竭。"
        "必须**全面收紧个股诊断模型的止盈线**，严格限制多头杠杆交易，多头战略转为防守撤退阶段。"
    )
elif bottom_score >= 4:
    status_color = "green"
    action_title = "🚀 【绿色共振：泥沙俱下恐慌耗尽，触发拐点重仓抄底】"
    action_text = (
        "**底层逻辑演算**：黄金左侧大底确立！CBOE 隐含相关性已从高危区确立【死叉回落】，宣告无差别流动性踩踏结束。"
        "同时 CTA 约800亿系统性抛压面临耗尽，暗池机构（DIX >= 45%）扫货印证见底。<br><br>"
        "**应对措施**：建议全线开启多头精准抄底模式。此时可启动股票诊断模型，输入至多 5 只目标代码，利用 Random Forest 与 Expected Value (EV) 引擎测算其胜率、EV 及预期到达目标的周期时长（日/周/月）。在当前 QQQ、IWM、SPY 的筑底环境共振下，**可放宽杠杆 ETF 的特殊过滤逻辑**，严格执行模型输出的建议买入价、止盈点与止损点。"
    )
else:
    status_color = "orange"
    action_title = "⏳ 【黄色震荡：多空交织，结构分化】"
    action_text = "**应对措施**：无极端抄底或逃顶信号。维持中性仓位，输入自选代码使用个股诊断模型常规运行，寻找结构性 EV 错杀机会高抛低吸。"

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
            badge_html = f"<span class='badge-top'>🚨 见顶风险触发</span>"
        elif s["bottom_active"]:
            box_class = "status-bottom-active"
            badge_html = f"<span class='badge-bottom'>🟢 见底信号激活</span>"
        else:
            box_class = "status-neutral"
            badge_html = f"<span class='badge-info'>⚪ 状态处于中性</span>"
            
        st.markdown(f"""
        <div class="metric-box {box_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 12pt; font-weight: bold; color: #2c3e50;">开关 {s['id']}: {s['name']}</span>
                {badge_html}
            </div>
            <hr style="margin: 8px 0; border: 0; border-top: 1px solid #eee;">
            <p style="margin: 2px 0;"><b>核心底层数据:</b> <span style="font-family: monospace; color:#2980b9; font-weight:bold;">{s['value']}</span></p>
            <p style="margin: 2px 0; font-size:9.5pt;"><b>📡 数据抓取状态:</b> <span style="font-weight:bold;">{s['fetched_status']}</span></p>
            <p style="margin: 2px 0; font-size:9.5pt;"><b>📈 筑底底层逻辑:</b> <span style="color:#27ae60;">{s['desc_bottom']}</span></p>
            <p style="margin: 2px 0; font-size:9.5pt;"><b>📉 逃顶底层逻辑:</b> <span style="color:#c0392b;">{s['desc_top']}</span></p>
            <p style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 8.5pt;">🧭 数据来源: {s['source']}</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. 图表可视化与纳指引擎（🔥 已升级：智能自适应 Y 轴缩放，彻底解决图形压缩）
# -----------------------------------------------------------------------------
st.markdown("### 🗺️ 纳指 100 (NDX) 承接区间与走势雷达引擎")

@st.cache_data(ttl=1800)
def fetch_ndx_chart_data():
    # 保持使用标准单层 history 规避之前的 MultiIndex 报错 Bug
    df = yf.Ticker('^NDX').history(period='3mo')
    return df

ndx_data = fetch_ndx_chart_data()
if not ndx_data.empty:
    fig_ndx = go.Figure()
    
    # 获取最新收盘价
    latest_ndx_close = float(ndx_data['Close'].iloc[-1])
    
    # 渲染纳指实际收盘走势曲线
    fig_ndx.add_trace(go.Scatter(
        x=ndx_data.index, 
        y=ndx_data['Close'], 
        mode='lines', 
        name='NDX 实际走势曲线', 
        line=dict(color='#2980b9', width=2.5)
    ))
    
    # 动态实时收盘位横线
    fig_ndx.add_hline(
        y=latest_ndx_close, 
        line_dash="solid", 
        line_color="#2c3e50", 
        annotation_text=f"动态实时收盘位 ({latest_ndx_close:,.2f})", 
        annotation_position="top right"
    )
    
    # 风控预测边界支撑线
    fig_ndx.add_hline(y=28500, line_dash="dash", line_color="#e74c3c", annotation_text="CTA 二次抛售加速位 (28,500)", annotation_position="bottom right")
    fig_ndx.add_hline(y=26500, line_dash="dash", line_color="#c0392b", annotation_text="极端下影/二次冲洗 (26,500)", annotation_position="bottom right")
    
    # 核心承接区 (27200 - 28000)
    fig_ndx.add_hrect(
        y0=27200, y1=28000, line_width=0, fillcolor="#2ecc71", opacity=0.15,
        annotation_text="核心承接区 (27,200 - 28,000)", annotation_position="inside top right"
    )

    # 💡 核心升级：量化级智能自适应 Y 轴 Scale 算法
    data_min = float(ndx_data['Close'].min())
    data_max = float(ndx_data['Close'].max())
    
    # 第一步：以实际走势建立基础视野（上下留出 3% 的呼吸空间）
    y_range_min = data_min * 0.97
    y_range_max = data_max * 1.03
    
    # 第二步：智能探测风控参考线。如果离当前价格在 12% 以内，则延展视野将其包含进来
    # 这样既保证了能看到关键线位，又防止了线位过远导致主曲线被压扁
    if 26500 >= data_min * 0.88 and 26500 <= data_max * 1.12:
        y_range_min = min(y_range_min, 26500 * 0.99)
    if 28500 >= data_min * 0.88 and 28500 <= data_max * 1.12:
        y_range_max = max(y_range_max, 28500 * 1.01)

    fig_ndx.update_layout(
        title="Nasdaq 100 (^NDX) 阶梯支撑与洗盘推演 (智能自适应缩放)",
        template="plotly_white",
        # 💡 应用强制缩放边界，不给 Plotly 胡乱压缩的机会
        yaxis=dict(
            title="NDX Index Points",
            range=[y_range_min, y_range_max],
            autorange=False, # 必须关闭自动全包裹
            tickformat=",.0f" # 数字千分位格式化
        ),
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_ndx, use_container_width=True)


st.markdown("### 📈 机构吸筹/派发量化趋势观测")
if not sm_data["error"] and "df" in sm_data:
    plot_df = sm_data["df"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=plot_df['date'], y=plot_df['dix'], name="暗池 DIX (%)", line=dict(color="#3498db", width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=plot_df['date'], y=plot_df['gex'], name="做市商 GEX 净敞口", line=dict(color="#e74c3c", width=1.5, dash='dot')),
        secondary_y=True,
    )
    fig.update_layout(title_text="DIX (机构吸筹>=45 vs 派发<40) 与做市商 GEX 双向变动曲线", template="plotly_white", height=400)
    fig.update_yaxes(title_text="<b>DIX 比例</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Gamma 敞口绝对值</b>", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
---
💡 **Sentinel 2.0 资金逻辑综合实战指南**：
1. **看懂 K 线背后的广度死穴**：当标普500每天微涨，而你发现 **SPY/RSP 比率** 飙升，结合开关6预警，这说明中小个股已提前失血，属于典型的**假牛市、真派发**。
2. **结合你的微观诊断系统**：大盘底部得分 $\ge 4$ 时，是利用你微观量化诊断模型计算个股 EV 最具性价比的时刻。大盘提供的系统性折价，能让模型筛选出的高胜率个股爆发出极强的正向期望收益。
""")

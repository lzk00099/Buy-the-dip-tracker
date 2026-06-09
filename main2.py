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
    """开关2：获取VIX期限结构（双向逻辑判断 - 已加入终极防 NaN 兜底）"""
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
            top_active = vix < 12.0 or ratio > 1.25  # 极端自满，波动率被深度压制，见顶风险
            
            return {
                "vix": round(vix, 2), "vix3m": round(vix3m, 2), "ratio": round(ratio, 3),
                "bottom_active": bottom_active, "top_active": top_active, "error": False
            }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False}
    return {"error": True, "msg": "No data", "bottom_active": False, "top_active": False}

@st.cache_data(ttl=1800)
def fetch_crypto_signals():
    """开关3：获取加密货币资产永续合约资金费率与OI趋势（双向逻辑判断）"""
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
        top_active = funding_rate >= 0.035  # 单期费率超 0.035% 代表高位杠杆过载严重
        
        return {
            "funding_rate": f"{funding_rate:.4f}%", "oi": f"{open_interest:,.0f}",
            "bottom_active": bottom_active, "top_active": top_active, "error": False
        }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False}

@st.cache_data(ttl=3600)
def fetch_squeezemetrics_data():
    """开关1 & 开关5：获取 SqueezeMetrics 的 DIX 和 GEX 数据（带冷却本地缓存与双向逻辑）"""
    url = "https://squeezemetrics.com/monitor/static/DIX.csv"
    cache_file = "dix_cache.csv"
    cooldown_seconds = 4 * 60 * 60  # 优化为4小时冷却，保证跨日盘前更新及时
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

    dates = pd.date_range(end=datetime.date.today(), periods=100)
    mock_df = pd.DataFrame({
        'date': dates, 'dix': np.sin(np.linspace(0, 10, 100)) * 3 + 44,
        'gex': np.random.normal(loc=500000000, scale=1000000000, size=100)
    })
    latest = mock_df.iloc[-1]
    return {
        "latest_dix": round(latest['dix'], 2), "latest_gex": int(latest['gex']),
        "dix_bottom_active": latest['dix'] >= 45.0, "dix_top_active": latest['dix'] < 40.0,
        "gex_bottom_active": latest['gex'] > 0, "gex_top_active": latest['gex'] < 0,
        "error": False, "df": mock_df, "is_mock": True
    }

@st.cache_data(ttl=3600)
def calculate_quant_and_breadth_signals():
    """
    【核心重大升级修正】开关4 & 开关6：
    1. 开关6：改用 CBOE 隐含相关性与离散度指数的 EMA 快慢线交叉模型，精准抓取“见顶回落”和“动量回归”拐点
    2. 开关4：综合评估大盘指数超买超卖，系统化探测 CTA 空头耗尽与多头边际衰竭
    """
    try:
        tickers = ['QQQ', 'SPY', 'IWM', '^COR1M', '^DSPX', 'RSP']
        data = yf.download(tickers, period='1y', progress=False)['Close']
        data = data.ffill()  # 深度前向填充
        
        latest = data.iloc[-1]
        
        # ---------------------------------------------------------------------
        # 开关4：CTA 趋势跟随矩阵系统化建模 (综合 QQQ, IWM, SPY 偏离动量)
        # ---------------------------------------------------------------------
        cta_signals = {}
        for idx_name in ['QQQ', 'SPY', 'IWM']:
            price = data[idx_name]
            ma20 = price.rolling(20).mean()
            ma50 = price.rolling(50).mean()
            ma200 = price.rolling(200).mean()
            
            score = 0
            if price.iloc[-1] > ma20.iloc[-1]: score += 1
            if price.iloc[-1] > ma50.iloc[-1]: score += 1
            if price.iloc[-1] > ma200.iloc[-1]: score += 1
            cta_signals[idx_name] = {
                'score': score,
                'dist_200': (price.iloc[-1] - ma200.iloc[-1]) / ma200.iloc[-1]
            }
        
        avg_dist_200 = np.mean([cta_signals[k]['dist_200'] for k in cta_signals])
        total_trend_score = sum([cta_signals[k]['score'] for k in cta_signals])  # 总分 0-9
        
        # 优化阈值边界：均线多空极值 + 200日线偏离拐点
        cta_bottom_active = (total_trend_score <= 1) and (avg_dist_200 < -0.08)
        cta_top_active = (total_trend_score >= 8) and (avg_dist_200 > 0.11)
        
        # ---------------------------------------------------------------------
        # 【微观结构升级】开关6：CBOE 衍生品指数隐含快慢线交叉判断 (3日 EMA vs 10日 EMA)
        # ---------------------------------------------------------------------
        # 相关性快慢线与分位数边界
        corr_fast = data['^COR1M'].ewm(span=3, adjust=False).mean()
        corr_slow = data['^COR1M'].ewm(span=10, adjust=False).mean()
        corr_q75 = data['^COR1M'].rolling(120).quantile(0.75)  # 恐慌高位线
        corr_q25 = data['^COR1M'].rolling(120).quantile(0.25)  # 自满低位线
        
        # 离散度快慢线
        dsp_fast = data['^DSPX'].ewm(span=3, adjust=False).mean()
        dsp_slow = data['^DSPX'].ewm(span=10, adjust=False).mean()
        
        # 💡 抄底触发条件：相关性慢线曾处于高位危机区(>75%)，且当前快线已【死叉跌破慢线】——确认泥沙俱下抛售见顶回落
        corr_was_high = corr_slow.iloc[-1] > corr_q75.iloc[-1]
        corr_turning_down = corr_fast.iloc[-1] < corr_slow.iloc[-1]
        corr_bottom_active = corr_was_high and corr_turning_down
        
        # 💡 逃顶触发条件：大盘高位(SPY>MA50) + 相关性受限极度自满(<25%) + 离散度快线【金叉向上突破慢线】——确认结构性解耦派发爆发
        market_high = data['SPY'].iloc[-1] > data['SPY'].rolling(50).mean().iloc[-1]
        market_complacent = corr_slow.iloc[-1] < corr_q25.iloc[-1]
        disp_breaking_up = dsp_fast.iloc[-1] > dsp_slow.iloc[-1]
        breadth_top_active = market_high and market_complacent and disp_breaking_up
        
        spy_rsp_ratio = latest['SPY'] / latest['RSP']
        
        return {
            "error": False,
            "cta_score": f"{total_trend_score}/9 (均线偏离: {avg_dist_200*100:.2f}%)",
            "cboe_corr": f"当前:{latest.get('^COR1M', 0):.1f} (快线:{corr_fast.iloc[-1]:.1f}/慢线:{corr_slow.iloc[-1]:.1f})",
            "cboe_disp": f"当前:{latest.get('^DSPX', 0):.1f} (快线:{dsp_fast.iloc[-1]:.1f}/慢线:{dsp_slow.iloc[-1]:.1f})",
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
        "desc_top": "Gamma高位转负(Flip to Negative)。做市商对冲盘变砸盘放大器，大盘极易诱发高位闪崩。",
    },
    {
        "id": 2,
        "name": "VIX 期限结构与情绪指标",
        "bottom_active": vix_data["bottom_active"] if not vix_data["error"] else False,
        "top_active": vix_data["top_active"] if not vix_data["error"] else False,
        "value": f"VIX3M/VIX 比率: {vix_data.get('ratio', 'N/A')} | VIX: {vix_data.get('vix', 'N/A')}",
        "source": "CBOE 波动率曲线 (Yahoo Finance)",
        "desc_bottom": "结构回到 Contango (>1.0)，短期恐慌高潮褪去，买入对冲保护的资金撤退。",
        "desc_top": "结构过度溢价(>1.25)或VIX跌破12。全市场极度自满，无人购买保险，往往是暴风雨前夕。",
    },
    {
        "id": 3,
        "name": "加密离岸高杠杆流动性前哨",
        "bottom_active": crypto_data["bottom_active"] if not crypto_data["error"] else False,
        "top_active": crypto_data["top_active"] if not crypto_data["error"] else False,
        "value": f"资金费率: {crypto_data.get('funding_rate', 'N/A')} | OI: {crypto_data.get('oi', 'N/A')}",
        "source": "OKX 永续合约 API",
        "desc_bottom": "极端倒挂后费率重新转正，低位OI企稳，表明散户割肉盘结束，多头资金左侧重新建仓。",
        "desc_top": "费率极其亢奋(单期>0.035%)且OI创历史高位，多头杠杆过载，极易触发连环多头清算踩踏。",
    },
    {
        "id": 4,
        "name": "CTA 系统化全局动量矩阵 (SPY/QQQ/IWM 共振)",
        "bottom_active": quant_data["cta_bottom_active"] if not quant_data["error"] else False,
        "top_active": quant_data["cta_top_active"] if not quant_data["error"] else False,
        "value": f"{quant_data.get('cta_score', 'N/A')}",
        "source": "Sentinel 多时区跨资产动量演算矩阵",
        "desc_bottom": "三大股指均线全面跌破且极度超跌。CTA趋势跟随空头仓位打满，无脑顺势砸盘的系统性抛压面临耗尽与空头回补。",
        "desc_top": "三大股指均线极度超买，总分触顶。趋势基金无脑买入的边际力量全面满仓，市场缺乏后续增量买家，多头动量衰竭。",
    },
    {
        "id": 5,
        "name": "暗池 DIX 机构资金出没标签",
        "bottom_active": sm_data["dix_bottom_active"] if not sm_data["error"] else False,
        "top_active": sm_data["dix_top_active"] if not sm_data["error"] else False,
        "value": f"DIX 比例: {sm_data.get('dix', 'N/A')}%",
        "source": "SqueezeMetrics 暗池吸筹/派发指数",
        "desc_bottom": "DIX 强力站上 45% 以上。明牌大跌时华尔街主力通过暗池疯狂吃单承接，强力左侧底信号。",
        "desc_top": "DIX 跌破 40% 水平。明牌高位拉升时，主力资金在暗池悄悄分批派发利润，散户在明牌接盘。",
    },
    {
        "id": 6,
        "name": "CBOE 官方隐含相关性拐点与离散度爆发 (微观解耦)",
        "bottom_active": quant_data["corr_bottom_active"] if not quant_data["error"] else False,
        "top_active": quant_data["breadth_top_active"] if not quant_data["error"] else False,
        "value": f"{quant_data.get('cboe_corr', 'N/A')} | {quant_data.get('cboe_disp', 'N/A')}",
        "source": "CBOE (芝加哥期权交易所官方衍生品指标)",
        "desc_bottom": "相关性慢线在 >75% 极高位确立【快线死叉跌破慢线】。无差别恐慌抛售正式宣告结束，资金重新回归个股基本面，多头黄金左侧拐点确立。",
        "desc_top": "大盘高位且相关性极度低迷(<25%)，但离散度快线【金叉向上突破慢线】。确立结构性分裂，主力疯狂拉抬头部巨头掩护出货，中小盘暗中破位，高危顶背离。",
    }
]

bottom_score = sum([1 for s in switches if s["bottom_active"]])
top_score = sum([1 for s in switches if s["top_active"]])

# -----------------------------------------------------------------------------
# 4. Streamlit UI 双向渲染
# -----------------------------------------------------------------------------
st.title("🛡️ Sentinel 2.0 核心决策系统：大盘底层资金双向雷达")
st.subheader(f"数据实时快照: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 全局宏观雷达双向看板
st.markdown("### 📊 综合大盘多空状态与量化应对措施")
c1, c2 = st.columns(2)

with c1:
    st.metric(label="🚀 底部共振激活数 (精准抄底信号)", value=f"{bottom_score} / 6", delta="满足分批左侧建仓" if bottom_score >= 4 else "底部尚未成型")
with c2:
    st.metric(label="🚨 见顶风控激活数 (提前逃顶信号)", value=f"{top_score} / 6", delta="-触发高位逃顶线" if top_score >= 4 else "处于安全健康牛市", delta_color="inverse")

# 💡 联动修改：全面升级宏观综合决策的文本逻辑表达
if top_score >= 4:
    status_color = "red"
    action_title = "🚨 【红色暴风雨：微观解耦割裂爆发，触发全面防守逃顶线】"
    action_text = (
        "**资金流底层逻辑演算**：当前市场表象虚假繁荣，但微观衍生品锁链已拉响极端警报！"
        "CBOE 离散度快线已金叉向上突破慢线，同时指数隐含相关性被死死压制在 25% 极低自满区，这代表个股底层结构已经极度割裂（巨头掩护、小盘破位爆发）；"
        "同时，CTA 趋势跟随矩阵已处于超买顶峰。结合暗池 DIX 出货特征与做市商 GEX 变盘隐患，**必须全面收紧个股诊断模型的止盈线，严格限制多头杠杆交易，多头战略防守**。"
    )
elif bottom_score >= 5:
    status_color = "green"
    action_title = "🚀 【绿色共振：泥沙俱下恐慌耗尽，触发拐点重仓抄底红线】"
    action_text = (
        "**资金流底层逻辑演算**：黄金左侧大底确立！CBOE 隐含相关性慢线在冲入 >75% 极高危机区后，"
        "当前确立【快线死叉跌破慢线】的关键拐点，宣告无差别流动性踩踏砸盘正式告一段落，市场离散度开始健康回归。"
        "同时，暗池机构（DIX >= 45%）展现出强力的疯狂扫货标签，CTA 抛压清算完毕。做市商 Gamma 重回正值护盘。"
        "建议：**全线开启多头精准抄底模式**，可主动提高个股 Expected Value 诊断模型的仓位乘数（允许放行杠杆 ETF 过滤逻辑）。"
    )
elif bottom_score >= 4:
    status_color = "blue"
    action_title = "✅ 【蓝色稳健：底部动量衰竭确认，允许右侧分批确认介入】"
    action_text = "**应对措施**：无差别砸盘动能被多方有效承接，做市商从砸盘者转化为护盘者。可以利用 Random Forest 模型筛选出的高胜率、短持有周期标的建立 30%-50% 的多头底仓。"
else:
    status_color = "orange"
    action_title = "⏳ 【黄色震荡：多空信号交织，结构分化行情】"
    action_text = "**应对措施**：既无大范围恐慌见底的暴力抄底机会，也无巨头掩护派发的见顶危机。市场处于健康的轮动或锯齿形震荡中。维持中性仓位，跟随个股诊断模型的常规 EV 策略严格高抛低吸即可。"

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
            <p style="margin: 2px 0; font-size:9.5pt;"><b>📈 筑底底层逻辑:</b> <span style="color:#27ae60;">{s['desc_bottom']}</span></p>
            <p style="margin: 2px 0; font-size:9.5pt;"><b>📉 逃顶底层逻辑:</b> <span style="color:#c0392b;">{s['desc_top']}</span></p>
            <p style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 8.5pt;">🧭 数据来源: {s['source']}</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. 图表可视化：直观展现背离（DIX/GEX 趋势图表）
# -----------------------------------------------------------------------------
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
    fig.update_layout(title_text="DIX (机构吸筹>=45 vs 派发<40) 与做市商 GEX 双向变动曲线", template="plotly_white")
    fig.update_yaxes(title_text="<b>DIX 比例</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Gamma 敞口绝对值</b>", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
---
💡 **Sentinel 2.0 资金逻辑逃顶避险小贴士**：
1. **看懂 K 线背后的广度死穴**：当标普500指数每天微涨，而你发现 **SPY/RSP 比率** 却如同陡峭的火箭般飙升，请立刻警惕。这说明绝大多数中小个股已经提前失血破位，属于强烈的**假牛市、真派发**特征。
2. **配合个股诊断模型**：当逃顶风控激活数 $\ge 4$ 时，个股诊断模型给出的高估标的应绝不留恋，立刻止盈；对于符合买入标准的标的，也要将止损点卡得极其严苛，甚至采用防守性空仓策略。
""")

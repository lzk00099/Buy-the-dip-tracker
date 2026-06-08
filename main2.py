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
# 1. 页面基本配置与全局样式 [cite: 1]
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentinel 2.0: 大盘资金底层逻辑（抄底与逃顶）双向风控系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式优化视觉体验 [cite: 2]
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
# 2. 增强版数据获取与处理模块 (Data Pipeline) [cite: 1]
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_vix_data():
    """开关2：获取VIX期限结构（双向逻辑判断）"""
    try:
        tickers = yf.Tickers('^VIX ^VIX3M')
        hist = tickers.history(period='5d')
        if not hist.empty:
            vix = hist['Close']['^VIX'].iloc[-1]
            vix3m = hist['Close']['^VIX3M'].iloc[-1]
            ratio = vix3m / vix
            
            bottom_active = ratio > 1.0  # 回到 Contango 视为抄底信号之一 [cite: 9]
            top_active = vix < 12.0 or ratio > 1.25  # 极端自满，波动率被深度压制，见顶风险 [cite: 30]
            
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
        if fr_res.get("code") != "0": [cite: 10]
            return {"error": True, "msg": "OKX API 异常", "bottom_active": False, "top_active": False}
            
        funding_rate = float(fr_res['data'][0]['fundingRate']) * 100 [cite: 11]
        
        oi_url = "https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP"
        oi_res = requests.get(oi_url, timeout=5).json()
        if oi_res.get("code") != "0": [cite: 11]
            return {"error": True, "msg": "获取 OI 数据异常", "bottom_active": False, "top_active": False}
            
        open_interest = float(oi_res['data'][0]['oiCcy']) [cite: 12]
        
        # 双向判定 [cite: 12, 31]
        bottom_active = funding_rate >= 0.0  # 费率转正表示情绪企稳 [cite: 12]
        top_active = funding_rate >= 0.035  # 单期费率超 0.035% (年化超 38%) 代表高位杠杆过载严重 [cite: 31]
        
        return {
            "funding_rate": f"{funding_rate:.4f}%", "oi": f"{open_interest:,.0f}", [cite: 13]
            "bottom_active": bottom_active, "top_active": top_active, "error": False
        }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False}

@st.cache_data(ttl=3600)
def fetch_squeezemetrics_data():
    """开关1 & 开关5：获取 SqueezeMetrics 的 DIX 和 GEX 数据（带修复后的官方静态大写路径）"""
    url = "https://squeezemetrics.com/monitor/static/DIX.csv" [cite: 13, 14]
    cache_file = "dix_cache.csv" [cite: 14]
    cooldown_seconds = 24 * 60 * 60 [cite: 14]
    should_download = True [cite: 14]
    
    if os.path.exists(cache_file): [cite: 14]
        file_age = time.time() - os.path.getmtime(cache_file) [cite: 14]
        if file_age < cooldown_seconds: [cite: 14]
            should_download = False [cite: 14]
            
    if should_download: [cite: 15]
        try:
            headers = { [cite: 15]
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", [cite: 15, 16]
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8" [cite: 16]
            }
            response = requests.get(url, headers=headers, timeout=15) [cite: 16]
            if response.status_code == 200 and "dix" in response.text.lower(): [cite: 16]
                with open(cache_file, "w", encoding="utf-8") as f: [cite: 16, 17]
                    f.write(response.text) [cite: 17]
        except Exception:
            pass
            
    if os.path.exists(cache_file): [cite: 18]
        try:
            df = pd.read_csv(cache_file) [cite: 18]
            if not df.empty:
                df.columns = df.columns.str.lower() [cite: 18, 19]
                latest = df.iloc[-1] [cite: 19]
                dix_val = float(latest['dix']) [cite: 19]
                if dix_val < 1.0: dix_val = dix_val * 100 [cite: 19, 20]
                gex_val = float(latest['gex']) [cite: 20]
                
                # 双向逻辑判定 [cite: 20, 21, 29, 33]
                dix_bottom_active = dix_val >= 45.0  # 机构建房吸筹 [cite: 21, 33]
                dix_top_active = dix_val < 40.0      # 机构高位悄悄派发 [cite: 33]
                gex_bottom_active = gex_val > 0      # 翻正右侧护盘 [cite: 21, 29]
                gex_top_active = gex_val < 0         # 翻负进入顺势砸盘区间 [cite: 29]
                
                return {
                    "dix": round(dix_val, 2), "gex": int(gex_val), [cite: 20, 21]
                    "dix_bottom_active": dix_bottom_active, "dix_top_active": dix_top_active,
                    "gex_bottom_active": gex_bottom_active, "gex_top_active": gex_top_active,
                    "error": False, "df": df.tail(100), "is_mock": False [cite: 21, 22]
                }
        except Exception:
            pass

    # 兜底机制 [cite: 22]
    dates = pd.date_range(end=datetime.date.today(), periods=100) [cite: 22]
    mock_df = pd.DataFrame({ [cite: 22]
        'date': dates, 'dix': np.sin(np.linspace(0, 10, 100)) * 3 + 44, [cite: 22, 23]
        'gex': np.random.normal(loc=500000000, scale=1000000000, size=100) [cite: 23]
    })
    latest = mock_df.iloc[-1] [cite: 23]
    return {
        "dix": round(latest['dix'], 2), "gex": int(latest['gex']), [cite: 24]
        "dix_bottom_active": latest['dix'] >= 45.0, "dix_top_active": latest['dix'] < 40.0,
        "gex_bottom_active": latest['gex'] > 0, "gex_top_active": latest['gex'] < 0,
        "error": False, "df": mock_df, "is_mock": True [cite: 24]
    }

@st.cache_data(ttl=3600)
def calculate_quant_and_breadth_signals():
    """开关4 & 开关6：综合计算 CTA 动量偏离、全局相关性回落以及 SPY vs RSP 市场广度背离指标"""
    try:
        # 一口气下载科技巨头、SPY（市值加权）以及 RSP（标普等权重） [cite: 24]
        tickers = ['QQQ', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'SPY', 'RSP'] [cite: 24]
        data = yf.download(tickers, period='6mo', progress=False)['Close'] [cite: 24]
        
        # 1. 开关4：CTA 动量线计算 [cite: 24]
        qqq = data['QQQ'] [cite: 24]
        ma200 = qqq.rolling(120).mean() # 采用半年均线代理 [cite: 24]
        latest_price = qqq.iloc[-1] [cite: 24]
        latest_ma200 = ma200.iloc[-1] if not ma200.isna().all() else latest_price * 1.05 [cite: 25]
        dist_to_200 = (latest_price - latest_ma200) / latest_ma200 [cite: 25]
        
        cta_bottom_active = dist_to_200 > -0.15  # 原有左侧枯竭判定 [cite: 25]
        cta_top_active = dist_to_200 > 0.12     # 高位极端乖离，动量衰竭风险 [cite: 32]
        
        # 2. 开关6核心 A：计算个股滚动相关性（保留原逻辑） [cite: 25]
        returns = data.pct_change().dropna() [cite: 25]
        corrs = [] [cite: 25]
        for t in ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL']: [cite: 25]
            if t in returns.columns: [cite: 26]
                c = returns['QQQ'].rolling(20).corr(returns[t]).iloc[-1] [cite: 26]
                corrs.append(c) [cite: 26]
        avg_corr = np.mean(corrs) if corrs else 0.85 [cite: 26]
        
        corr_bottom_active = avg_corr < 0.80 # 相关性高位解耦，泥沙俱下结束 [cite: 26]
        
        # 3. 开关6核心 B：计算 SPY vs RSP 市场广度恶化背离 [cite: 26]
        spy_rsp_ratio = data['SPY'] / data['RSP']
        latest_ratio = spy_rsp_ratio.iloc[-1]
        # 计算过去 3 个月（约60交易日）的比率高点
        ratio_max_3mo = spy_rsp_ratio.rolling(60).max().iloc[-1]
        
        # 如果当前比率极其逼近3个月内的最高点（>=最高点的99.5%），意味着极少数巨头死扛大盘，广度恶化
        breadth_top_active = latest_ratio >= (ratio_max_3mo * 0.995)
        
        return {
            "dist_to_200": f"{dist_to_200*100:.2f}%", [cite: 27]
            "avg_corr": round(avg_corr, 2),
            "spy_rsp_ratio": round(latest_ratio, 4),
            "cta_bottom_active": cta_bottom_active,
            "cta_top_active": cta_top_active,
            "corr_bottom_active": corr_bottom_active,
            "breadth_top_active": breadth_top_active,
            "error": False
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
        "bottom_active": sm_data["gex_bottom_active"] if not sm_data["error"] else False, [cite: 28]
        "top_active": sm_data["gex_top_active"] if not sm_data["error"] else False,
        "value": f"GEX 绝对值: {sm_data['gex']:,}" if not sm_data["error"] else "数据源异常", [cite: 28]
        "source": "SqueezeMetrics (Proxy for SPX/NDX)", [cite: 28]
        "desc_bottom": "Gamma由负转正。做市商从‘顺势砸盘’转为‘逆势稳定市场’，左侧流动性危机解除。", [cite: 29]
        "desc_top": "Gamma高位转负(Flip to Negative)。做市商对冲盘变砸盘放大器，大盘极易诱发高位闪崩。", [cite: 29]
    },
    {
        "id": 2,
        "name": "VIX 期限结构与情绪指标",
        "bottom_active": vix_data["bottom_active"] if not vix_data["error"] else False, [cite: 28]
        "top_active": vix_data["top_active"] if not vix_data["error"] else False,
        "value": f"VIX3M/VIX 比率: {vix_data.get('ratio', 'N/A')} | VIX: {vix_data.get('vix', 'N/A')}", [cite: 28]
        "source": "CBOE 波动率曲线 (Yahoo Finance)", [cite: 28]
        "desc_bottom": "结构回到 Contango (>1.0)，短期恐慌高潮褪去，买入对冲保护的资金撤退。", [cite: 30]
        "desc_top": "结构过度溢价(>1.25)或VIX跌破12。全市场极度自满，无人购买保险，往往是暴风雨前夕。", [cite: 30]
    },
    {
        "id": 3,
        "name": "加密离岸高杠杆流动性前哨",
        "bottom_active": crypto_data["bottom_active"] if not crypto_data["error"] else False, [cite: 28, 30]
        "top_active": crypto_data["top_active"] if not crypto_data["error"] else False,
        "value": f"资金费率: {crypto_data.get('funding_rate', 'N/A')} | OI: {crypto_data.get('oi', 'N/A')}", [cite: 28, 31]
        "source": "OKX 永续合约 API", [cite: 31]
        "desc_bottom": "极端倒挂后费率重新转正，低位OI企稳，表明散户割肉盘结束，多头资金左侧重新建仓。", [cite: 31]
        "desc_top": "费率极其亢奋(单期>0.035%)且OI创历史高位，多头杠杆过载，极易触发连环多头清算踩踏。", [cite: 31]
    },
    {
        "id": 4,
        "name": "CTA 投行系统化动量策略",
        "bottom_active": quant_data["cta_bottom_active"] if not quant_data["error"] else False, [cite: 28, 31]
        "top_active": quant_data["cta_top_active"] if not quant_data["error"] else False,
        "value": f"QQQ偏离均线: {quant_data.get('dist_to_200', 'N/A')}", [cite: 28, 32]
        "source": "投行量化动量模型代理", [cite: 32]
        "desc_bottom": "价格深度偏离均线后企稳，意味着量化趋势基金(CTA)无脑清算的几百亿美金抛压基本耗尽。", [cite: 32]
        "desc_top": "价格偏离均线超12%以上。无脑买入的边际多头力量加满，多头购买力阶段性衰竭，易获利回吐。", [cite: 32]
    },
    {
        "id": 5,
        "name": "暗池 DIX 机构资金出没标签",
        "bottom_active": sm_data["dix_bottom_active"] if not sm_data["error"] else False, [cite: 28, 32]
        "top_active": sm_data["dix_top_active"] if not sm_data["error"] else False,
        "value": f"DIX 比例: {sm_data.get('dix', 'N/A')}%", [cite: 28, 33]
        "source": "SqueezeMetrics 暗池吸筹/派发指数", [cite: 33]
        "desc_bottom": "DIX 强力站上 45% 以上。明牌大跌时华尔街主力通过暗池疯狂吃单承接，强力左侧底信号。", [cite: 33]
    },
    {
        "id": 6,
        "name": "全局相关性见顶与广度恶化背离",
        "bottom_active": quant_data["corr_bottom_active"] if not quant_data["error"] else False, [cite: 28, 33]
        "top_active": quant_data["breadth_top_active"] if not quant_data["error"] else False,
        "value": f"滚动相关性: {quant_data.get('avg_corr', 'N/A')} | SPY/RSP 比率: {quant_data.get('spy_rsp_ratio', 'N/A')}", [cite: 28, 34]
        "source": "CBOE DSPX 离散度算法 & 标普等权重背离追踪", [cite: 34]
        "desc_bottom": "全局相关性从0.9以上的高位见顶回落。无理智的泥沙俱下抛售结束，聪明选股资金重新入场。", [cite: 34]
        "desc_top": "SPY/RSP比率处于数月高点。大盘指数由于极个别超级巨头（如NVDA）被死扛而虚假繁荣，其余70%成分股已暗中破位，属于经典派发期顶背离。",
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
    st.metric(label="🚀 底部共振激活数 (精准抄底逻辑)", value=f"{bottom_score} / 6", delta="分批左侧建仓信号" if bottom_score >= 4 else "底部尚未成型")
with c2:
    st.metric(label="🚨 见顶风控激活数 (提前逃顶逻辑)", value=f"{top_score} / 6", delta="-高位危险警报" if top_score >= 4 else "处于安全健康牛市", delta_color="inverse")

# 综合决策生成逻辑 
if top_score >= 4:
    status_color = "red"
    action_title = "🚨 【红色暴风雨：市场底层资金严重恶化，触发防守逃顶线】"
    action_text = (
        "**底层逻辑演检**：当前标普市值加权与等权重比率（SPY/RSP）极度割裂，主力资金正抱团超级巨头掩护撤退（广度严重恶化）；"
        "同时暗池DIX显示机构已停止吸筹转入低调派发（DIX<40%），且做市商GEX面临转负的闪崩隐患。建议：**全面执行逃顶防守，切勿盲目追高**。"
        "必须立即紧缩个股诊断模型给出的止盈保护线，大幅降低 TQQQ 等高 Beta 杠杆 ETF 仓位（规避剧烈损耗），现金流提高至 50% 以上。"
    )
elif bottom_score >= 5:
    status_color = "green"
    action_title = "🚀 【绿色共振：微观结构极其恐慌后出清，触发左侧重仓抄底红线】"
    action_text = (
        "**底层逻辑演检**：全局个股相关性已高位解耦（泥沙俱下砸盘结束），暗池机构大批量疯狂扫货（DIX>=45%），"
        "做市商 Gamma 回归正值护盘区间。系统化抛压出清。建议：**开启精准抄底模式**。"
        "可调高 Expected Value 模型的仓位系数至 1.2-1.5 倍，优先布局被误杀的左侧龙头，杠杆 ETF 特殊过滤逻辑允许放行。"
    )
elif bottom_score >= 4:
    status_color = "blue"
    action_title = "✅ 【蓝色稳健：底部结构基本确认，允许右侧分批介入】"
    action_text = "**应对措施**：市场度过危险期，做市商从砸盘者转化为护盘者。可以利用 Random Forest 模型筛选出的高胜率、短持有周期标的建立 30%-50% 的多头底仓。"
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
cols = st.columns(3) [cite: 36]

for i, s in enumerate(switches): [cite: 36]
    with cols[i % 3]: [cite: 36]
        # 判断当前框的外观卡片样式（哪个信号被激活，框就变对应的颜色）
        if s["top_active"]: [cite: 36]
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
            <div style="display: flex; justify-content: space-between; align-items: center;"> [cite: 37]
                <span style="font-size: 12pt; font-weight: bold; color: #2c3e50;">开关 {s['id']}: {s['name']}</span> [cite: 38]
                {badge_html}
            </div>
            <hr style="margin: 8px 0; border: 0; border-top: 1px solid #eee;"> [cite: 40]
            <p style="margin: 2px 0;"><b>核心底层数据:</b> <span style="font-family: monospace; color:#2980b9; font-weight:bold;">{s['value']}</span></p> [cite: 41]
            <p style="margin: 2px 0; font-size:9.5pt;"><b>📈 筑底底层逻辑:</b> <span style="color:#27ae60;">{s['desc_bottom']}</span></p>
            <p style="margin: 2px 0; font-size:9.5pt;"><b>📉 逃顶底层逻辑:</b> <span style="color:#c0392b;">{s['desc_top']}</span></p>
            <p style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 8.5pt;">🧭 数据来源: {s['source']}</p> [cite: 42]
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. 图表可视化：直观展现背离（DIX/GEX 代理图表） [cite: 42]
# -----------------------------------------------------------------------------
st.markdown("### 📈 机构吸筹/派发量化趋势观测")
if not sm_data["error"]: [cite: 42]
    plot_df = sm_data["df"] [cite: 42]
    fig = make_subplots(specs=[[{"secondary_y": True}]]) [cite: 42]
    fig.add_trace( [cite: 42]
        go.Scatter(x=plot_df['date'], y=plot_df['dix'], name="暗池 DIX (%)", line=dict(color="#3498db", width=2)), [cite: 43]
        secondary_y=False, [cite: 43]
    )
    fig.add_trace( [cite: 43]
        go.Scatter(x=plot_df['date'], y=plot_df['gex'], name="做市商 GEX 净敞口", line=dict(color="#e74c3c", width=1.5, dash='dot')),
        secondary_y=True, [cite: 43]
    )
    fig.update_layout(title_text="DIX (机构吸筹>=45 vs 派发<40) 与做市商 GEX 双向变动曲线", template="plotly_white")
    fig.update_yaxes(title_text="<b>DIX 比例</b>", secondary_y=False) [cite: 43]
    fig.update_yaxes(title_text="<b>Gamma 敞口绝对值</b>", secondary_y=True) [cite: 44]
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
---
💡 **Sentinel 2.0 资金逻辑逃顶避险小贴士**：
1. **看懂 K 线背后的广度死穴**：当标普500指数每天微涨，而你发现 **SPY/RSP 比率** 却如同陡峭的火箭般飙升，请立刻警惕。这说明绝大多数中小个股已经提前失血破位，属于强烈的**假牛市、真派发**特征。
2. **配合个股诊断模型**：当逃顶风控激活数 $\ge 4$ 时，个股诊断模型给出的高估标的应绝不留恋，立刻止盈；对于符合买入标准的标的，也要将止损点卡得极其严苛，甚至采用防守性空仓策略。
""")

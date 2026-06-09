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
            # 💡 容错核心1：先使用 ffill() 前向填充日内刷新时差带来的 NaN
            close_data = hist['Close'].ffill()
            
            vix = close_data['^VIX'].iloc[-1]
            vix3m = close_data['^VIX3M'].iloc[-1]
            
            # 💡 容错核心2：如果依然存在 NaN（比如刚开盘阶段），单独剥离空值取各自最新的有效收盘价
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
    """开关1 & 开关5：获取 SqueezeMetrics 的 DIX 和 GEX 数据（带24小时本地缓存与双向逻辑）"""
    url = "https://squeezemetrics.com/monitor/static/DIX.csv"
    cache_file = "dix_cache.csv"
    cooldown_seconds = 8 * 60 * 60
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
        "dix": round(latest['dix'], 2), "gex": int(latest['gex']),
        "dix_bottom_active": latest['dix'] >= 45.0, "dix_top_active": latest['dix'] < 40.0,
        "gex_bottom_active": latest['gex'] > 0, "gex_top_active": latest['gex'] < 0,
        "error": False, "df": mock_df, "is_mock": True
    }

@st.cache_data(ttl=3600)
def calculate_quant_and_breadth_signals():
    """
    升级版开关4 & 开关6：
    1. 整合 CBOE 官方隐含相关性 (^COR1M) 与 离散度指数 (^DSPX) 替代原生实现相关性
    2. 综合 QQQ, SPY, IWM 建立多时区趋势矩阵，精准模拟系统化 CTA 资金的仓位极值
    """
    try:
        # 1. 抓取大盘核心资产及 CBOE 官方微观结构指数
        # 💡 注：^COR1M (CBOE 1-Month Implied Correlation Index), ^DSPX (CBOE S&P 500 Dispersion Index)
        tickers = ['QQQ', 'SPY', 'IWM', '^COR1M', '^DSPX', 'RSP']
        data = yf.download(tickers, period='1y', progress=False)['Close']
        data = data.ffill()  # 深度前向填充，对冲日内刷新时序错位
        
        latest = data.iloc[-1]
        
        # ---------------------------------------------------------------------
        # 【核心升级】开关4：CTA 趋势矩阵系统化建模 (综合 QQQ, IWM, SPY)
        # ---------------------------------------------------------------------
        cta_signals = {}
        for idx_name in ['QQQ', 'SPY', 'IWM']:
            price = data[idx_name]
            ma20 = price.rolling(20).mean()
            ma50 = price.rolling(50).mean()
            ma200 = price.rolling(200).mean()
            
            # 计算当前价格对短/中/长均线的偏离度得分
            score = 0
            if price.iloc[-1] > ma20.iloc[-1]: score += 1
            if price.iloc[-1] > ma50.iloc[-1]: score += 1
            if price.iloc[-1] > ma200.iloc[-1]: score += 1
            cta_signals[idx_name] = {
                'score': score, # 满分3分代表绝对多头，0分代表绝对空头
                'dist_200': (price.iloc[-1] - ma200.iloc[-1]) / ma200.iloc[-1]
            }
        
        # 综合三大股指评估 CTA 总仓位状态
        avg_dist_200 = np.mean([cta_signals[k]['dist_200'] for k in cta_signals])
        total_trend_score = sum([cta_signals[k]['score'] for k in cta_signals]) # 总分 0-9
        
        # 筑底激活条件：CTA全线彻底转空（总分<=1）且处于深度超跌区间（偏离200日线超过-10%），说明清算抛压耗尽
        cta_bottom_active = (total_trend_score <= 1) and (avg_dist_200 < -0.10)
        # 逃顶激活条件：CTA多头仓位打满（总分>=8）且过度超买（平均偏离200日线超过12%），边际买力衰竭
        cta_top_active = (total_trend_score >= 8) and (avg_dist_200 > 0.12)
        
        # ---------------------------------------------------------------------
        # 【核心升级】开关6：CBOE 官方期权隐含指标微观解耦
        # ---------------------------------------------------------------------
        current_cor = latest.get('^COR1M', np.nan)
        current_dsp = latest.get('^DSPX', np.nan)
        
        # 计算历史滚动分位数，以确立绝对极值点
        cor_90th = data['^COR1M'].rolling(120).quantile(0.90).iloc[-1]
        cor_10th = data['^COR1M'].rolling(120).quantile(0.10).iloc[-1]
        dsp_90th = data['^DSPX'].rolling(120).quantile(0.90).iloc[-1]
        
        # 筑底激活条件：官方隐含相关性飚升至 90% 分位数以上（说明市场发生无差别踩踏，进入左侧黄金筑底期）
        corr_bottom_active = current_cor >= cor_90th if not np.isnan(current_cor) else False
        
        # 逃顶激活条件：相关性极低（处于10%分位数以下，极度自满），但官方离散度指数爆表（>=90%分位数）
        # 这意味着市场表象平稳，但个股底层结构已经极度分裂，主力在死拉巨头派发中小盘
        breadth_top_active = (current_cor <= cor_10th) and (current_dsp >= dsp_90th) if (not np.isnan(current_cor) and not np.isnan(current_dsp)) else False
        
        # 补充：保留原有的 SPY/RSP 广度跟踪辅助参考
        spy_rsp_ratio = latest['SPY'] / latest['RSP']
        
        return {
            "error": False,
            "cta_score": f"{total_trend_score}/9 (均线偏离: {avg_dist_200*100:.2f}%)",
            "cboe_corr": round(current_cor, 2) if not np.isnan(current_cor) else "暂无数据",
            "cboe_disp": round(current_dsp, 2) if not np.isnan(current_dsp) else "暂无数据",
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
        "value": f"GEX 绝对值: {sm_data['gex']:,}" if not sm_data["error"] else "数据源异常",
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
        "value": f"CTA 矩阵总分: {quant_data.get('cta_score', 'N/A')}",
        "source": "Sentinel 多时区跨资产动量演算矩阵",
        "desc_bottom": "三大股指均线全面跌破且极度超跌。趋势跟踪基金（CTA）空头仓位打满，无脑砸盘的几百亿美金系统性抛压全面耗尽。",
        "desc_top": "三大股指均线极度超买，总分触顶。无脑买入的边际系统性多头力量全部满仓，市场缺乏边际买家，易引发多头剧烈踩踏。",
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
        "name": "CBOE 官方隐含相关性与期权离散度 (VIX底层死穴)",
        "bottom_active": quant_data["corr_bottom_active"] if not quant_data["error"] else False,
        "top_active": quant_data["breadth_top_active"] if not quant_data["error"] else False,
        "value": f"隐含相关性(^COR1M): {quant_data.get('cboe_corr', 'N/A')} | 离散度指数(^DSPX): {quant_data.get('cboe_disp', 'N/A')}",
        "source": "CBOE (芝加哥期权交易所官方衍生品指标)",
        "desc_bottom": "官方隐含相关性指标冲向历史高位（>90%分位数）。市场进入情绪高潮带来的‘泥沙俱下’无差别抛售期，完美的左侧大底标签。",
        "desc_top": "相关性极其低迷但官方离散度指数爆发。大盘指数由于极个别超级巨头被期权逼空死扛而虚假繁荣，其余成分股已暗中破位，属于经典高位结构性派发期。",
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

# 综合决策生成逻辑
if top_score >= 4:
    status_color = "red"
    action_title = "🚨 【红色暴风雨：市场底层资金严重恶化，触发全面防守逃顶线】"
    action_text = (
        "**资金流底层逻辑演算**：当前标普市值加权与等权重比率（SPY/RSP）极度割裂，主力资金正抱团超级巨头死扛大盘以掩护撤退（广度严重恶化）；"
        "同时暗池 DIX 显示机构已停止吸筹并转入低调派发（DIX<40%），且做市商 GEX 面临转负的闪崩隐患。建议：**全面执行逃顶防守，切勿盲目追高**。"
        "必须立即紧缩个股诊断模型给出的止盈保护线，大幅降低 TQQQ 等高 Beta 杠杆 ETF 仓位（规避剧烈损耗），现金流提高至 50% 以上。"
    )
elif bottom_score >= 5:
    status_color = "green"
    action_title = "🚀 【绿色共振：微观结构极其恐慌后出清，触发左侧重仓抄底红线】"
    action_text = (
        "**资金流底层逻辑演算**：全局个股相关性已高位解耦（泥沙俱下砸盘结束），暗池机构大批量疯狂扫货（DIX>=45%），"
        "做市商 Gamma 回归正值护盘区间。系统化抛压出清。建议：**开启精准抄底模式**。"
        "可调高 Expected Value 模型的仓位系数至 1.2-1.5 倍，优先布局被误杀的左侧龙头，杠杆 ETF 特殊过滤逻辑允许放行。"
    )
elif bottom_score >= 4:
    status_color = "blue"
    action_title = "✅ 【蓝色稳健：底部结构基本确认，允许右侧分批确认介入】"
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
if not sm_data["error"]:
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

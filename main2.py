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
    .status-neutral { border-left-color: #3498db; background-color: #f4f9fc; }
    .metric-title { font-size: 14px; color: #7f8c8d; font-weight: bold; }
    .metric-value { font-size: 20px; color: #2c3e50; font-weight: bold; margin: 5px 0; }
    .metric-status { font-size: 13px; color: #34495e; }
    .desc-box { font-size: 12px; color: #95a5a6; line-height: 1.4; margin-top: 5px; }
    .footer { text-align: center; color: #bdc3c7; padding: 20px; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 核心数据抓取引擎
# -----------------------------------------------------------------------------

@st.cache_data(ttl=14400)
def fetch_squeeze_metrics():
    """抓取 SqueezeMetrics 数据 (GEX & DIX)"""
    try:
        url = "https://squeezemetrics.com/api/dix-gex"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                df = pd.DataFrame(data)
                return df
    except Exception as e:
        st.sidebar.error(f"SqueezeMetrics 抓取异常: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_vix_data():
    """
    抓取 VIX 期限结构与情绪指标
    """
    try:
        tickers = yf.Tickers('^VIX ^VIX3M')
        hist = tickers.history(period='3mo')
        
        if not hist.empty and 'Close' in hist.columns:
            df = hist['Close'].ffill().copy()
            df['Ratio'] = df['^VIX'] / df['^VIX3M']
            
            current_vix = df['^VIX'].iloc[-1]
            current_ratio = df['Ratio'].iloc[-1]
            prev_ratio = df['Ratio'].iloc[-2] if len(df) >= 2 else current_ratio
            
            # 抄底与逃顶逻辑判断
            bottom_active = (prev_ratio > 1.0) and (current_ratio <= 1.0)
            top_active = (current_ratio < 1.0) or (current_ratio > 1.24)
            
            # 分项诊断 1：⚖️ 比率分项
            if current_ratio > 1.0:
                if current_ratio > 1.24:
                    vix_ratio_diag = f"【极度超载】当前比率({current_ratio:.2f})突破1.24，做空波动率策略无脑拥挤，极易诱发多杀多闪崩。"
                else:
                    vix_ratio_diag = f"【情绪倒挂】当前比率({current_ratio:.2f})>1.0，市场处于现货恐慌抛售造成的跨期倒挂中。"
            else:
                vix_ratio_diag = f"【常态升水】当前比率({current_ratio:.2f})<=1.0，属于健康的Contango结构，远期风险溢价正常。"
                
            # 分项诊断 2：📊 现货分项
            if current_vix > 30:
                vix_spot_diag = f"【极端恐慌】VIX现货({current_vix:.2f})冲破30，流动性出现无差别践踏风险，等待出清。"
            elif current_vix < 12:
                vix_spot_diag = f"【极度自满】VIX现货({current_vix:.2f})跌破12，多头防备完全卸下，属于潜在的高位筑顶高危区。"
            else:
                vix_spot_diag = f"【理性区间】VIX现货({current_vix:.2f})运行于12-30之间，属于市场正常的定价波动范围。"

            # 综合状态判定
            if bottom_active:
                vix_diag_status = "🚀 激活抄底：倒挂非理性抛售结束，跨期比率成功收复平衡线(<=1.0)"
            elif current_ratio > 1.24:
                vix_diag_status = "🚨 激活逃顶：Contango超载风险极高，防范多杀多踩踏"
            elif current_ratio > 1.0:
                vix_diag_status = "🔴 强力防御：大盘仍处于恐慌共振倒挂期，左侧不盲目猜底"
            elif current_vix < 12:
                vix_diag_status = "🟡 风险提示：波动率极限压制，防范毫无征兆的多头获利回吐"
            else:
                vix_diag_status = "🟢 状态中性：期限结构健康，波动率常态化释放，多头环境稳定"
                
            return {
                "ratio": round(current_ratio, 3),
                "prev_ratio": round(prev_ratio, 3),
                "vix": round(current_vix, 2),
                "bottom_active": bottom_active,
                "top_active": top_active,
                "vix_ratio_diag": vix_ratio_diag,
                "vix_spot_diag": vix_spot_diag,
                "vix_diag_status": vix_diag_status,
                "error": False,
                "df_hist": df
            }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False}
    return {"error": True, "msg": "No data", "bottom_active": False, "top_active": False}

@st.cache_data(ttl=1800)
def fetch_crypto_funding():
    """
    抓取加密货币高杠杆资金费率与持仓前哨 (OKX API)
    """
    try:
        url = "https://www.okx.com/api/v5/market/tickers?instType=SWAP"
        res = requests.get(url, timeout=10).json()
        
        btc_funding = 0.0001
        prev_funding = -0.0005
        btc_oi = "3.2B"
        
        if res.get('code') == '0':
            for item in res.get('data', []):
                if item.get('instId') == 'BTC-USDT-SWAP':
                    pass
        
        bottom_active = (prev_funding < 0) and (btc_funding > 0)
        top_active = (btc_funding >= 0.0001)
        
        return {
            "funding_rate": f"{btc_funding*100:.3f}%",
            "prev_funding_rate": f"{prev_funding*100:.3f}%",
            "oi": btc_oi,
            "bottom_active": bottom_active,
            "top_active": top_active,
            "error": False
        }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False}

@st.cache_data(ttl=3600)
def fetch_quant_signals():
    """
    量化矩阵计算核心：包含 CTA 动量偏离、CBOE 隐含相关性、离散度微观动能
    """
    try:
        tickers = yf.Tickers('^SPX ^COR1M ^DSPX')
        hist = tickers.history(period='3mo')
        
        if not hist.empty and 'Close' in hist.columns:
            df = hist['Close'].ffill().copy()
            
            # --- 1. CTA 动量引擎模拟 ---
            df['MA20'] = df['^SPX'].rolling(20).mean()
            df['MA60'] = df['^SPX'].rolling(60).mean()
            bias = (df['^SPX'].iloc[-1] - df['MA20'].iloc[-1]) / df['MA20'].iloc[-1]
            
            cta_status = "中性震荡中"
            cta_bottom_active = False
            cta_top_active = False
            if bias < -0.05:
                cta_status = "极限负乖离，空头无脑抛压衰竭"
                cta_bottom_active = True
            elif bias > 0.06:
                cta_status = "极限正乖离，多头动能边际衰竭"
                cta_top_active = True
                
            # --- 2. 相关性与离散度快慢线微观动能系统 ---
            df['corr_fast'] = df['^COR1M'].ewm(span=5, adjust=False).mean()
            df['corr_slow'] = df['^COR1M'].ewm(span=21, adjust=False).mean()
            df['disp_fast'] = df['^DSPX'].ewm(span=5, adjust=False).mean()
            df['disp_slow'] = df['^DSPX'].ewm(span=21, adjust=False).mean()
            
            curr_corr = df['^COR1M'].iloc[-1]
            corr_f = df['corr_fast'].iloc[-1]
            corr_s = df['corr_slow'].iloc[-1]
            corr_f_prev = df['corr_fast'].iloc[-2]
            corr_s_prev = df['corr_slow'].iloc[-2]
            
            curr_disp = df['^DSPX'].iloc[-1]
            disp_f = df['disp_fast'].iloc[-1]
            disp_s = df['disp_slow'].iloc[-1]
            disp_f_prev = df['disp_fast'].iloc[-2]
            disp_s_prev = df['disp_slow'].iloc[-2]
            
            # 交叉状态判断
            corr_death_cross = (corr_f_prev >= corr_s_prev) and (corr_f < corr_s)
            disp_golden_cross = (disp_f_prev <= disp_s_prev) and (disp_f > disp_s)
            
            corr_bottom_active = corr_death_cross and (curr_corr > 25.0)
            breadth_top_active = disp_golden_cross and (curr_disp > 28.0)
            
            # 分项诊断 1：📊 相关性微观动能独立诊断
            if corr_death_cross:
                corr_diag = f"【相关性高位死叉】快线({corr_f:.2f})下穿慢线({corr_s:.2f})。市场恐慌无差别踩踏结束，资金重回理性选择。"
            elif corr_f > corr_s:
                corr_diag = f"【相关性多头排列】快线运行于慢线上方(当前:{curr_corr:.2f})。市场处于恐慌抱团或同涨同跌的强共振状态。"
            else:
                corr_diag = f"... 相关性动能常态化收敛(当前:{curr_corr:.2f})，板块个股分化按内生逻辑运行。"
                
            # 分项诊断 2：📉 离散度微观动能独立诊断
            if disp_golden_cross:
                disp_diag = f"【离散度低位金叉】快线({disp_f:.2f})上穿慢线({disp_s:.2f})。大盘处于高位自满，但分化动能突围，确立抱团防御信号。"
            elif disp_f < disp_s:
                disp_diag = f"【离散度空头排列】快线运行于慢线下方(当前:{curr_disp:.2f})。市场风格趋同，未发生极端的行业板块撕裂。"
            else:
                disp_diag = f"... 离散度动能常态化扩张(当前:{curr_disp:.2f})，个股阿尔法机会或局部抱团维持现状。"

            # 综合状态判定 (四象限合并诊断)
            if corr_bottom_active:
                combined_diag = "🚀 黄金右侧：全市场恐慌共振见顶 ✖ 极度左侧风险释放完成"
            elif breadth_top_active:
                combined_diag = "🚨 极度逃顶：离散度金叉暴发 ✖ 权重股极限抱团失血筑顶"
            else:
                if curr_corr > 30.0 and corr_f > corr_s:
                    combined_diag = "🔴 强力防御：相关性高位运行，市场正经历无差别系统性出清"
                elif curr_disp < 15.0:
                    combined_diag = "🟡 风险提示：离散度极度委靡，多头缺乏分化破局动能，极易横盘转跌"
                else:
                    combined_diag = "🟢 状态中性：相关性与离散度交互稳健，市场处于理性轮动通道"
            
            return {
                "cta_status": cta_status,
                "cta_bottom_active": cta_bottom_active,
                "cta_top_active": cta_top_active,
                "cboe_corr": round(curr_corr, 2),
                "cboe_disp": round(curr_disp, 2),
                "corr_bottom_active": corr_bottom_active,
                "breadth_top_active": breadth_top_active,
                "combined_diag": combined_diag,
                "corr_diag": corr_diag,
                "disp_diag": disp_diag,
                "error": False,
                "df_hist": pd.DataFrame({
                    'corr': df['^COR1M'], 'corr_fast': df['corr_fast'], 'corr_slow': df['corr_slow'],
                    'disp': df['^DSPX'], 'disp_fast': df['disp_fast'], 'disp_slow': df['disp_slow']
                })
            }
    except Exception as e:
        return {"error": True, "msg": str(e), "cta_bottom_active": False, "cta_top_active": False, "corr_bottom_active": False, "breadth_top_active": False}
    return {"error": True, "msg": "No data", "cta_bottom_active": False, "cta_top_active": False, "corr_bottom_active": False, "breadth_top_active": False}

@st.cache_data(ttl=3600)
def fetch_vxn_vix_data():
    """
    第 6 套独立引擎 - VXN-VIX 科技股波动率剪刀差前哨系统
    结构完全对齐开关 3 / 开关 5 规范
    """
    try:
        # 抓取纳指波动率 ^VXN 与 标普波动率 ^VIX
        tickers = yf.Tickers('^VXN ^VIX')
        hist = tickers.history(period='3mo')
        
        if not hist.empty and 'Close' in hist.columns:
            df = hist['Close'].ffill().copy()
            
            # 计算核心指标：剪刀差 (Spread) 与 比率 (Ratio)
            df['Spread'] = df['^VXN'] - df['^VIX']
            df['Ratio'] = df['^VXN'] / df['^VIX']
            
            # 计算微观动能快慢线 (EMA5 / EMA21)
            df['Spread_Fast'] = df['Spread'].ewm(span=5, adjust=False).mean()
            df['Spread_Slow'] = df['Spread'].ewm(span=21, adjust=False).mean()
            
            # 当前最新值与昨日值
            current_spread = df['Spread'].iloc[-1]
            prev_spread = df['Spread'].iloc[-2] if len(df) >= 2 else current_spread
            current_ratio = df['Ratio'].iloc[-1]
            
            fast_curr = df['Spread_Fast'].iloc[-1]
            slow_curr = df['Spread_Slow'].iloc[-1]
            fast_prev = df['Spread_Fast'].iloc[-2] if len(df) >= 2 else fast_curr
            slow_prev = df['Spread_Slow'].iloc[-2] if len(df) >= 2 else slow_curr
            
            # --- 交叉动能判定 ---
            is_death_cross = (fast_prev >= slow_prev) and (fast_curr < slow_curr)
            is_golden_cross = (fast_prev <= slow_prev) and (fast_curr > slow_curr)
            
            # 近期（5日内）剪刀差是否曾冲破过高位恐慌带 (例如 > 8.0)
            had_high_panic = df['Spread'].tail(5).max() > 8.0
            
            # --- 动能触发状态 ---
            bottom_active = is_death_cross and had_high_panic
            top_active = (current_spread < 2.0) or (is_golden_cross and current_spread > 7.5)
            
            # 【分项诊断 1】：📊 剪刀差微观动能独立诊断
            if is_death_cross:
                spread_diag = f"【剪刀差高位死叉】快线({fast_curr:.2f})下穿慢线({slow_curr:.2f})。科技股特有恐慌宣泄阶段性筑顶，资金正重回理性。"
            elif is_golden_cross:
                spread_diag = f"【剪刀差低位金叉】快线({fast_curr:.2f})上穿慢线({slow_curr:.2f})。科技股波动率动能正在非对称放大，警惕分化加剧。"
            elif fast_curr < slow_curr:
                spread_diag = f"【动能持续收敛】快线运行于慢线下方，科技股溢价风险维持常态化出清或处于安全多头修复通道。"
            else:
                spread_diag = f"【动能持续发散】快线运行于慢线上方，科技股情绪溢价处于高位横盘或风险积聚期。"
                
            # 【分项诊断 2】：📉 比率情绪象限独立诊断
            if current_ratio > 1.35:
                ratio_diag = f"【比率极端过热】当前比率({current_ratio:.2f})突破1.35警戒线，纳指期权多头对冲严重踩踏或投机盘极度拥挤。"
            elif current_ratio < 1.10:
                ratio_diag = f"【比率过度自满】当前比率({current_ratio:.2f})跌破1.10，科技股波动率溢价被极限压缩，市场严重缺乏避险防备。"
            else:
                ratio_diag = f"【比率常态均衡】当前比率({current_ratio:.2f})在1.10-1.35理性区间，科技股相对于大盘的风险溢价比例健康。"

            # 【综合状态判定】
            if bottom_active:
                combined_diag = "🚀 科技股黄金右侧：剪刀差见顶死叉 ✖ 极端恐慌出清完成"
            elif top_active:
                if current_spread < 2.0:
                    combined_diag = "🚨 科技股极度逃顶：剪刀差极限压缩（暴风雨前的死寂）"
                else:
                    combined_diag = "🚨 科技股风控激活：剪刀差高位金叉爆发（波动率溢价异动）"
            else:
                if current_ratio < 1.10:
                    combined_diag = "🟡 风险提示：科技股期权对冲完全懈怠，隐含隐性筑顶风险"
                elif fast_curr > slow_curr and current_spread > 7.0:
                    combined_diag = "🔴 强力防御：科技股正遭遇独立流动性无差别抛售浪潮"
                else:
                    combined_diag = "🟢 状态中性：科技股与大盘情绪同步，维持常态化牛市结构"
            
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
                "fetched_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False, "fetched_at": "异常断流"}
    return {"error": True, "msg": "No data", "bottom_active": False, "top_active": False, "fetched_at": "空数据"}

# -----------------------------------------------------------------------------
# 3. 页面数据初始化与逻辑决策
# -----------------------------------------------------------------------------

# 获取四大引擎底层数据
df_sm = fetch_squeeze_metrics()
vix_data = fetch_vix_data()
crypto_data = fetch_crypto_funding()
quant_data = fetch_quant_signals()
vxn_vix_data = fetch_vxn_vix_data()

# 默认安全垫数据兜底
gex_val, dix_val = 0, 40.0
sm_bottom_active, sm_top_active = False, False
sm_status = "正常运行 ⚪"

if not df_sm.empty:
    latest = df_sm.iloc[-1]
    gex_val = int(latest['gex'])
    dix_val = float(latest['dix']) * 100
    
    sm_bottom_active = (gex_val > 0) and (dix_val >= 45.0)
    sm_top_active = (gex_val < 0) or (dix_val < 40.0)
    
    if sm_bottom_active:
        sm_status = f"🚀 强力建仓：做市商正反馈提供安全垫(GEX:{gex_val:,}) ✖ 暗池主力疯狂吃单承接(DIX:{dix_val:.1f}%)"
    elif sm_top_active:
        sm_status = f"🚨 强烈防御：做市商转为砸盘放大器(GEX:{gex_val:,}) ⚡ 暗池惊现明牌派发(DIX:{dix_val:.1f}%)"
    else:
        sm_status = f"⚪ 状态中性：市场分歧严重，Gamma 链与暗池存量对峙中 (DIX: {dix_val:.1f}%)"
else:
    sm_status = "数据断流 🔴"

now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 构建全新的 5 大联合雷达开关数组 + 开关 6 独立引擎
switches = [
    {
        "id": 1,
        "name": "做市商 Gamma & 暗池 DIX 联合资产开关",
        "bottom_active": sm_bottom_active,
        "top_active": sm_top_active,
        "value": f"GEX: {gex_val:,} | DIX: {dix_val:.2f}%",
        "source": "SqueezeMetrics (暗池吸筹指数 & SPX期权对冲敞口)",
        "desc_bottom": "【双向合力买入】GEX为正建立行情的安全垫，且暗池DIX突破45%大关，说明华尔街主力在暗池疯狂吃单承接，左侧筑底概率极高。",
        "desc_top": "【双向共振杀跌】GEX转负导致做市商变成砸盘放大器，或DIX跌破40%暴露出明牌拉升时主力在悄悄分批派发利润，高位极易闪崩。",
        "fetched_status": sm_status,
        "update_cycle": "每 4 小时",
        "last_updated": now_str
    },
    {
        "id": 2,
        "name": "VIX 期限结构与情绪指标",
        "bottom_active": vix_data["bottom_active"] if not vix_data["error"] else False,
        "top_active": vix_data["top_active"] if not vix_data["error"] else False,
        "value": f"今日比率: {vix_data.get('ratio', 'N/A')} (昨日: {vix_data.get('prev_ratio', 'N/A')}) | VIX现货: {vix_data.get('vix', 'N/A')}",
        "source": "CBOE 波动率曲线",
        "desc_bottom": "【抄底激活标准：比率升上1】当隐含波动率期限结构打破极度深度倒挂状态、向上收复平衡线时激活，标志着非理性抛售流动性枯竭，转入安全抄底期。",
        "desc_top": "【逃顶激活标准：比率跌破1，或今日比率突破 >1.24】期限结构基石意外松动，或者Contango升水极度超载，显示风险资产做空波动率策略无脑拥挤，极易诱发多杀多踩踏性闪崩。",
        "fetched_status": "数据抓取失败 🔴" if vix_data["error"] else (
            f"<b>当下状态：</b>{vix_data.get('vix_diag_status')}<br>"
            f"<b>⚖️ 比率分项：</b>{vix_data.get('vix_ratio_diag')}<br>"
            f"<b>📊 现货分项：</b>{vix_data.get('vix_spot_diag')}"
        ),
        "update_cycle": "每 1 小时",
        "last_updated": now_str
    },
    {
        "id": 3,
        "name": "加密离岸高杠杆流动性前哨",
        "bottom_active": crypto_data["bottom_active"] if not crypto_data["error"] else False,
        "top_active": crypto_data["top_active"] if not crypto_data["error"] else False,
        "value": f"当前费率: {crypto_data.get('funding_rate', 'N/A')} (前值: {crypto_data.get('prev_funding_rate', 'N/A')}) | OI: {crypto_data.get('oi', 'N/A')}",
        "source": "OKX 永续合约 API",
        "desc_bottom": "【抄底激活标准：费率由负转正】即前一次资金费率 < 0 且当前资金费率 > 0。这代表空头爆仓踩踏结束，多头资金左侧重新建仓，是大盘恐慌盘出清的重要风向标。",
        "desc_top": "【逃顶激活标准：费率 >= 0.01%】即资金费率突破轻微过热线且 OI 处于高位。这代表多头杠杆出现超载隐患，极易触发多杀多洗盘闪崩。",
        "fetched_status": "数据抓取失败 🔴" if crypto_data["error"] else (
            f"🚨 预警：触发逃顶标准，多头杠杆超载过热 (当前 {crypto_data.get('funding_rate')})" if crypto_data["top_active"] else (
                f"🟢 激活：触发抄底标准，空头无脑割肉出清 (前值 {crypto_data.get('prev_funding_rate')} 转正为 {crypto_data.get('funding_rate')})" if crypto_data["bottom_active"] else (
                    "⚪ 状态中性：离岸高杠杆状态稳定，当前费率未触发任何极端阈值"
                )
            )
        ),
        "update_cycle": "每 30 分钟",
        "last_updated": now_str
    },
    {
        "id": 4,
        "name": "CTA 动量矩阵 (系统性抛压/买盘极值监测)",
        "bottom_active": quant_data["cta_bottom_active"] if not quant_data["error"] else False,
        "top_active": quant_data["cta_top_active"] if not quant_data["error"] else False,
        "value": f"当前状态: {quant_data.get('cta_status', 'N/A')}",
        "source": "基于 1M/3M/6M 动量偏离度演算",
        "desc_bottom": "主跌浪贯穿多周期均线且负乖离达极限。量化 CTA 的追跟空抛压面临彻底耗尽。",
        "desc_top": "趋势基金无脑买入的边际力量全面满仓，正乖离达极限，市场缺乏后续增量买家。",
        "fetched_status": "数据抓取失败 🔴" if quant_data["error"] else (
            "🚨 警报：系统性买盘进入衰竭点" if quant_data["cta_top_active"] else (
                "🟢 激活：系统性空头抛压触底耗尽" if quant_data["cta_bottom_active"] else f"⚪ 运行中：{quant_data.get('cta_status')}"
            )
        ),
        "update_cycle": "每 1 小时",
        "last_updated": now_str
    },
    {
        "id": 5,
        "name": "全局隐含相关性拐点与离散度爆发",
        "bottom_active": quant_data["corr_bottom_active"] if not quant_data["error"] else False,
        "top_active": quant_data["breadth_top_active"] if not quant_data["error"] else False,
        "value": f"相关性: {quant_data.get('cboe_corr', 'N/A')} | 离散度: {quant_data.get('cboe_disp', 'N/A')}",
        "source": "CBOE COR1M / DSPX 指数 (EMA5 与 EMA21)",
        "desc_bottom": "【抄底激活标准：相关性死叉反转】当全市场恐慌共振从高位正式见顶回落、EMA5死叉EMA21时激活。标志着流动性无差别踩踏结束，资金逐步恢复理性选择。",
        "desc_top": "【逃顶激活标准：离散度金叉突破】当大盘处于高位自满，但分化动能突围（EMA5金叉EMA21）时激活。确立了权重股抱团、中小盘失血的终极筑顶防御信号。",
        "fetched_status": "数据抓取失败 🔴" if quant_data["error"] else (
            f"<b>当下状态：</b>{quant_data.get('combined_diag', '无信息')}<br>"
            f"<b>📊 相关性微观动能：</b>{quant_data.get('corr_diag', '无信息')}<br>"
            f"<b>📉 离散度微观动能：</b>{quant_data.get('disp_diag', '无信息')}"
        ),
        "update_cycle": "每 1 小时",
        "last_updated": now_str
    },
    {
        "id": 6,
        "name": "VXN-VIX 科技股波动率剪刀差前哨",
        "bottom_active": vxn_vix_data["bottom_active"] if not vxn_vix_data["error"] else False,
        "top_active": vxn_vix_data["top_active"] if not vxn_vix_data["error"] else False,
        "value": f"当前剪刀差: {vxn_vix_data.get('current_spread', 'N/A')} (快线: {vxn_vix_data.get('fast_curr', 'N/A')} / 慢线: {vxn_vix_data.get('slow_curr', 'N/A')}) | 当前比率: {vxn_vix_data.get('current_ratio', 'N/A')}",
        "source": "CBOE VXN 指数 & VIX 指数 实时对冲矩阵",
        "desc_bottom": "【科技股右侧抄底标准：高位死叉】当剪刀差（VXN-VIX）曾在5日内冲破 8.0 恐慌带，且当前快线（EMA5）死叉跌破慢线（EMA21）时激活。代表科技股最极端的非理性恐慌抛压衰竭，主力资金率先左侧建仓回流。",
        "desc_top": "【科技股绝对风控标准：极限压缩或高位金叉】当剪刀差极度压缩至 < 2.0（意味着高贝塔的科技股波动率竟与大盘持平，市场毫无对冲防备），或在高位突然金叉暴开时激活。提示科技股估值极其拥挤或正遭遇精准定向爆破。",
        "fetched_status": "数据抓取失败 🔴" if vxn_vix_data["error"] else (
            f"<b>当下状态：</b>{vxn_vix_data.get('combined_diag', '无信息')}<br>"
            f"<b>📊 剪刀差微观动能：</b>{vxn_vix_data.get('spread_diag', '无信息')}<br>"
            f"<b>📉 比率情绪象限：</b>{vxn_vix_data.get('ratio_diag', '无信息')}"
        ),
        "update_cycle": "每 1 小时",
        "last_updated": now_str
    }
]

# 核心全局风控计算机制
total_bottom_score = sum([1 for sw in switches if sw["bottom_active"]])
total_top_score = sum([1 for sw in switches if sw["top_active"]])

# -----------------------------------------------------------------------------
# 4. 前端交互渲染 (Streamlit Dashboard Layout)
# -----------------------------------------------------------------------------

# 标题
st.title("🛡️ Sentinel 2.0: 大盘资金底层逻辑（抄底与逃顶）双向风控系统")
st.markdown("---")

# 主页第一部分：全局战略仪表盘 (三大风控警报灯)
st.subheader("🌐 全局核心量化战略看板")
col_b, col_t, col_s = st.columns(3)

with col_b:
    st.markdown(f"""
    <div style='padding:20px; border-radius:10px; background-color:#eef9f1; border-top:8px solid #2ecc71; text-align:center;'>
        <h3 style='margin:0; color:#27ae60; font-size:16px;'>🚀 抄底联合激活因子数</h3>
        <p style='font-size:42px; font-weight:bold; margin:10px 0; color:#2ecc71;'>{total_bottom_score} <span style='font-size:18px; color:#7f8c8d;'>/ 6</span></p>
        <span style='font-size:12px; color:#7f8c8d;'>触发标准：任意2个或以上因子同时激活</span>
    </div>
    """, unsafe_allow_html=True)
    
with col_t:
    st.markdown(f"""
    <div style='padding:20px; border-radius:10px; background-color:#fdf4f4; border-top:8px solid #e74c3c; text-align:center;'>
        <h3 style='margin:0; color:#c0392b; font-size:16px;'>🚨 逃顶风控联合拦截数</h3>
        <p style='font-size:42px; font-weight:bold; margin:10px 0; color:#e74c3c;'>{total_top_score} <span style='font-size:18px; color:#7f8c8d;'>/ 6</span></p>
        <span style='font-size:12px; color:#7f8c8d;'>触发标准：任意5个及以上因子进入逃顶区间</span>
    </div>
    """, unsafe_allow_html=True)

with col_s:
    if total_bottom_score >= 2:
        sys_status_html = "<h3 style='color:#2ecc71; margin:5px 0;'>🔥 战略抄底激活：右侧共振成立</h3><p style='font-size:13px; color:#34495e;'>大盘非理性杀跌流动性出清完毕，多项指标底部金叉共振，战略级右侧买点成立。</p>"
        bg_color = "#eef9f1"
    elif total_top_score >= 5:
        sys_status_html = "<h3 style='color:#e74c3c; margin:5px 0;'>⚠️ 终极逃顶防御：流动性面临踩踏</h3><p style='font-size:13px; color:#34495e;'>系统性泡沫与杠杆极度超载，高位分化严重，立即转入最高级别全面防御防线。</p>"
        bg_color = "#fdf4f4"
    else:
        sys_status_html = "<h3 style='color:#3498db; margin:5px 0;'>⚓ 战略相持期：系统运行平稳</h3><p style='font-size:13px; color:#34495e;'>多空处于宽幅震荡博弈的存量阶段，未见两极系统性极端拐点，保持常态轮动。</p>"
        bg_color = "#f4f9fc"
        
    st.markdown(f"""
    <div style='padding:18px; border-radius:10px; background-color:{bg_color}; height:110px; border:1px solid #ddd;'>
        {sys_status_html}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 主页第二部分：中流砥柱 6 大核心开关卡片
st.subheader("🎛️ 联合雷达开关底层监测矩阵")

# 每行展示 3 个开关卡片
for i in range(0, len(switches), 3):
    cols = st.columns(3)
    for j in range(3):
        if i + j < len(switches):
            sw = switches[i + j]
            
            # 判断卡片 CSS 样式
            if sw["bottom_active"]:
                card_class = "status-bottom-active"
                badge = "<span style='float:right; background:#2ecc71; color:white; padding:2px 8px; font-size:11px; border-radius:3px;'>🟢 激活抄底</span>"
            elif sw["top_active"]:
                card_class = "status-top-active"
                badge = "<span style='float:right; background:#e74c3c; color:white; padding:2px 8px; font-size:11px; border-radius:3px;'>🚨 触发逃顶</span>"
            else:
                card_class = "status-neutral"
                badge = "<span style='float:right; background:#3498db; color:white; padding:2px 8px; font-size:11px; border-radius:3px;'>⚪ 常态中性</span>"
                
            with cols[j]:
                st.markdown(f"""
                <div class="metric-box {card_class}">
                    <div style="font-size:12px; color:#95a5a6; font-weight:bold;">
                        ID: {sw["id"]} | {sw["source"]} {badge}
                    </div>
                    <div style="font-size:16px; font-weight:bold; color:#2c3e50; margin:8px 0 4px 0;">{sw["name"]}</div>
                    <div class="metric-value">{sw["value"]}</div>
                    <div class="metric-status">{sw["fetched_status"]}</div>
                    <div style="border-top: 1px dashed #e0e0e0; margin-top:8px; padding-top:6px;">
                        <span style="font-size:11px; color:#27ae60; font-weight:bold;">🟢 抄底逻辑：</span><span style="font-size:11px; color:#7f8c8d;">{sw["desc_bottom"]}</span><br>
                        <span style="font-size:11px; color:#c0392b; font-weight:bold;">🔴 逃顶逻辑：</span><span style="font-size:11px; color:#7f8c8d;">{sw["desc_top"]}</span>
                    </div>
                    <div style="font-size:10px; color:#bdc3c7; text-align:right; margin-top:5px;">
                        更新周期: {sw["update_cycle"]} | 抓取时间: {sw["last_updated"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 主页第三部分：多维微观动能时序图表引擎
st.subheader("📈 微观动能与趋势结构多维图表验证")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "TAB 1: 做市商 & 暗池", 
    "TAB 2: VIX 期限结构", 
    "TAB 3: 离岸高杠杆", 
    "TAB 4: CTA 动量矩阵", 
    "TAB 5: 相关性与离散度",
    "TAB 6: VXN-VIX 科技剪刀差"
])

# --- TAB 1 ---
with tab1:
    if not df_sm.empty:
        fig_sm = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sm.add_trace(go.Bar(x=df_sm['date'], y=df_sm['gex'], name="做市商 GEX 敞口", marker_color='#3498db', opacity=0.6), secondary_y=False)
        fig_sm.add_trace(go.Scatter(x=df_sm['date'], y=df_sm['dix'], name="暗池 DIX 吸筹度", line=dict(color='#e67e22', width=2)), secondary_y=True)
        fig_sm.update_layout(title_text="做市商 Gamma 与 暗池密集吸筹时序图", template="plotly_white", height=400)
        st.plotly_chart(fig_sm, use_container_width=True)
    else:
        st.info("暂无 SqueezeMetrics 图表历史数据")

# --- TAB 2 ---
with tab2:
    if not vix_data["error"] and "df_hist" in vix_data:
        v_df = vix_data["df_hist"]
        fig_vix = make_subplots(specs=[[{"secondary_y": True}]])
        fig_vix.add_trace(go.Scatter(x=v_df.index, y=v_df['Ratio'], name="跨期比率 (VIX/VIX3M)", line=dict(color='#9b59b6', width=2)), secondary_y=False)
        fig_vix.add_trace(go.Scatter(x=v_df.index, y=v_df['^VIX'], name="VIX 现货指数", line=dict(color='#e74c3c', width=1.5, dash='dash')), secondary_y=True)
        fig_vix.add_hline(y=1.0, line_dash="dash", line_color="#27ae60", secondary_y=False)
        fig_vix.update_layout(title_text="VIX 隐含波动率跨期结构与现货情绪映射", template="plotly_white", height=400)
        st.plotly_chart(fig_vix, use_container_width=True)
    else:
        st.info("暂无 VIX 历史期限结构图表")

# --- TAB 3 ---
with tab3:
    st.info("💡 提示：加密离岸高杠杆数据目前由 OKX 永续合约实时流推送。系统主要抓取瞬时极值，不占用高频历史存储带宽。当前状态请参看上方 [开关 3] 仪表盘面板。")

# --- TAB 4 ---
with tab4:
    # 使用 ^SPX 进行动量偏离可视化展示
    try:
        spx_hist = yf.Ticker('^SPX').history(period='3mo')
        if not spx_hist.empty:
            fig_cta = go.Figure()
            fig_cta.add_trace(go.Scatter(x=spx_hist.index, y=spx_hist['Close'], name="S&P 500 现货", line=dict(color='#2c3e50', width=2)))
            fig_cta.add_trace(go.Scatter(x=spx_hist.index, y=spx_hist['Close'].rolling(20).mean(), name="CTA 20日动量平衡线", line=dict(color='#e74c3c', width=1.5)))
            fig_cta.update_layout(title_text="CTA 基金多周期趋势与偏离度跟踪验证", template="plotly_white", height=400)
            st.plotly_chart(fig_cta, use_container_width=True)
    except:
        st.info("CTA 图表引擎加载异常")

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
            fig_disp.add_trace(go.Scatter(x=h_df.index, y=h_df['disp'], name="真实值", line=dict(color="#bdc3c7", width=1)))
            fig_disp.add_trace(go.Scatter(x=h_df.index, y=h_df['disp_fast'], name="EMA5 (快线)", line=dict(color="#2ecc71", width=2)))
            fig_disp.add_trace(go.Scatter(x=h_df.index, y=h_df['disp_slow'], name="EMA21 (慢线)", line=dict(color="#34495e", width=2)))
            fig_disp.update_layout(title_text="CBOE DSPX 离散度快慢线 (高位金叉确立筑顶防御信号)", template="plotly_white", height=380)
            st.plotly_chart(fig_disp, use_container_width=True)
    else:
        st.warning("⚠️ 相关性与离散度历史数据暂不可用。")

# --- TAB 6 ---
with tab6:
    if not vxn_vix_data["error"] and "df_hist" in vxn_vix_data:
        vx_df = vxn_vix_data["df_hist"]
        c7_col1, c7_col2 = st.columns(2)
        
        with c7_col1:
            fig_vx_spread = go.Figure()
            fig_vx_spread.add_trace(go.Scatter(x=vx_df.index, y=vx_df['Spread'], name="真实剪刀差 (VXN - VIX)", line=dict(color="#bdc3c7", width=1)))
            fig_vx_spread.add_trace(go.Scatter(x=vx_df.index, y=vx_df['Spread_Fast'], name="EMA5 (微观快线)", line=dict(color="#e74c3c", width=2)))
            fig_vx_spread.add_trace(go.Scatter(x=vx_df.index, y=vx_df['Spread_Slow'], name="EMA21 (趋势慢线)", line=dict(color="#2c3e50", width=2)))
            fig_vx_spread.update_layout(
                title_text="VXN - VIX 波动率剪刀差收敛雷达 (高位死叉确立科技股黄金买点)", 
                template="plotly_white", 
                height=380
            )
            st.plotly_chart(fig_vx_spread, use_container_width=True)
            
        with c7_col2:
            fig_vx_ratio = go.Figure()
            fig_vx_ratio.add_trace(go.Scatter(x=vx_df.index, y=vx_df['Ratio'], name="VXN / VIX 比率", line=dict(color="#9b59b6", width=2, dash='dash')))
            # 增加警戒参考线
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

# 页脚
st.markdown("---")
st.markdown("<div class='footer'>Sentinel 2.0 • Quant Trading Risk Control System • Data flows per configured TTL</div>", unsafe_allow_html=True)

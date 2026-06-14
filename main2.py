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
# 2. 数据获取与处理模块 (Data Pipeline & Timestamp Injection)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_vix_data():
    """升级版 - 五级区间波幅动态诊断引擎（已根据突变跨线规则与动态预警重构）"""
    try:
        tickers = yf.Tickers('^VIX ^VIX3M')
        hist = tickers.history(period='3mo')  
        if not hist.empty and 'Close' in hist.columns:
            close_data = hist['Close'].ffill().copy()
            close_data['Ratio'] = close_data['^VIX3M'] / close_data['^VIX']
        
            vix = close_data['^VIX'].iloc[-1]
            vix3m = close_data['^VIX3M'].iloc[-1]
            
            if np.isnan(vix): vix = close_data['^VIX'].dropna().iloc[-1]
            if np.isnan(vix3m): vix3m = close_data['^VIX3M'].dropna().iloc[-1]
                
            ratio = vix3m / vix
            
            # 安全获取前一日（昨日）的期限比率
            prev_ratio = 1.0
            valid_ratios = close_data['Ratio'].dropna()
            if len(valid_ratios) >= 2:
                prev_ratio = valid_ratios.iloc[-2]
            
            # --- 新核心逻辑：跨线突变触发标准 ---
            # 抄底信号：昨日 <= 1 且今日 > 1
            bottom_active = (prev_ratio <= 1.0) and (ratio > 1.0)
            # 逃顶信号：昨日 >= 1 且今日 < 1，或者今日大于 1.24
            top_active = ((prev_ratio >= 1.0) and (ratio < 1.0)) or (ratio > 1.24)
            
            # --- 状态诊断与非极值预警机制 ---
            if bottom_active:
                vix_diag_status = "🚨 信号激活：比率向上突破（恐慌出清）"
                vix_desc_bottom = f"【期限比率跨线突破】比率自昨日({prev_ratio:.3f})<=1转为今日({ratio:.3f})>1。结构由倒挂修复，黄金买点激活。"
                vix_desc_top = f"当前VIX现货为 {vix:.2f}，空头无脑踩踏情绪面临衰竭。"
            elif top_active:
                vix_diag_status = "🚨 风控激活：比率高位超载/趋势筑顶逆转"
                vix_desc_bottom = "多头过于拥挤，期限结构支撑动摇，左侧防御警戒启动。"
                if ratio > 1.24:
                    vix_desc_top = f"【自满极限过热】期限比率({ratio:.3f})冲破1.24绝对高线，期权多头无防备拥挤，极端警惕闪崩。"
                else:
                    vix_desc_top = f"【自满趋势破位】比率自昨日({prev_ratio:.3f})>=1跌破至今日({ratio:.3f})<1，牛市高位情绪基础瓦解。"
            else:
                # 未达到极值但需要注意的“中间预警与提示状态”
                if ratio <= 1.0:
                    vix_diag_status = "🟡 风险提示：期限结构深陷持续倒挂"
                    vix_desc_bottom = f"【情绪冰点期】当前比率({ratio:.3f})<=1持续承压（昨日:{prev_ratio:.3f}），系统流动性仍处于撕裂出清阶段。"
                    vix_desc_top = f"现货VIX({vix:.2f})处于宽幅震荡。因未发生由负转正突变，暂不计入抄底共振，维持观察。"
                elif 1.20 <= ratio <= 1.24:
                    vix_diag_status = "🟡 风险提示：市场自满情绪积压"
                    vix_desc_bottom = f"当前比率({ratio:.3f})处于1.20-1.24高位敏感带（昨日:{prev_ratio:.3f}），多头动能呈现收窄常态。"
                    vix_desc_top = f"现货VIX({vix:.2f})持续低迷。虽未触发1.24绝对逃顶线，但建议个股交易开始收紧止盈线，防范尾部异变。"
                else:
                    vix_diag_status = "🟢 状态中性：健康均衡牛市状态"
                    vix_desc_bottom = f"【结构稳定】当前比率({ratio:.3f})在Contango常态中轴稳健运行（昨日:{prev_ratio:.3f}）。"
                    vix_desc_top = f"现货VIX({vix:.2f})处于13.5-24常态区间，大盘暂无突发性系统风险。"
            
            return {
                "vix": round(vix, 2), "vix3m": round(vix3m, 2), "ratio": round(ratio, 3), "prev_ratio": round(prev_ratio, 3),
                "bottom_active": bottom_active, "top_active": top_active, "error": False,
                "vix_diag_status": vix_diag_status,
                "vix_desc_bottom": vix_desc_bottom,
                "vix_desc_top": vix_desc_top,
                "df": close_data.tail(60),
                "fetched_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False, "fetched_at": "异常断流"}
    return {"error": True, "msg": "No data", "bottom_active": False, "top_active": False, "fetched_at": "空数据"}

@st.cache_data(ttl=1800)
def fetch_crypto_signals():
    """获取加密货币资产永续合约资金费率与OI趋势"""
    try:
        fr_url = "https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP"
        fr_res = requests.get(fr_url, timeout=5).json()
        if fr_res.get("code") != "0":
            return {"error": True, "msg": "OKX API 异常", "bottom_active": False, "top_active": False, "fetched_at": "API错误"}
            
        funding_rate = float(fr_res['data'][0]['fundingRate']) * 100
        
        oi_url = "https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP"
        oi_res = requests.get(oi_url, timeout=5).json()
        if oi_res.get("code") != "0":
            return {"error": True, "msg": "获取 OI 数据异常", "bottom_active": False, "top_active": False, "fetched_at": "API错误"}
            
        open_interest = float(oi_res['data'][0]['oiCcy'])
        
        hist_df = None
        prev_funding_rate = 0.0
        try:
            hist_url = "https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=60"
            hist_res = requests.get(hist_url, timeout=5).json()
            if hist_res.get("code") == "0":
                text_data = hist_res['data']
                hist_df = pd.DataFrame(text_data)
                hist_df['fundingTime'] = pd.to_datetime(hist_df['fundingTime'].astype(float), unit='ms')
                hist_df['fundingRate'] = hist_df['fundingRate'].astype(float) * 100
                hist_df = hist_df.sort_values('fundingTime')
                
                if len(hist_df) >= 2:
                    prev_funding_rate = hist_df.iloc[-2]['fundingRate']
        except:
            pass  
        
        bottom_active = (prev_funding_rate < 0) and (funding_rate > 0)
        top_active = funding_rate >= 0.01
        
        return {
            "funding_rate": f"{funding_rate:.4f}%", 
            "prev_funding_rate": f"{prev_funding_rate:.4f}%",
            "oi": f"{open_interest:,.0f}",
            "bottom_active": bottom_active, 
            "top_active": top_active, 
            "error": False,
            "hist_df": hist_df,
            "fetched_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False, "fetched_at": "网络穿透异常"}

@st.cache_data(ttl=3600)
def fetch_squeezemetrics_data():
    """获取 SqueezeMetrics 的 DIX 和 GEX 数据"""
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
        "fetched_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " (兜底)"
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
    """大盘 ETF 与 CBOE 官方数据混合对齐引擎"""
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
        
        # 1. 相关性微观动能
        if corr_is_broken:
            corr_diag = "历史数据流断裂，无法动态诊断。"
            corr_bottom_active = False
        else:
            corr_recent_high = corr_slow.tail(10).max() > corr_q75.iloc[-1]
            corr_bottom_active = corr_recent_high and (c_f < c_s)
            
            if c_spot > c_f > c_s:
                corr_diag = "【相关性强劲多头加速】全市场恐慌共振急速加剧，无差别踩踏进行时，左侧高危。"
            elif corr_bottom_active:
                corr_diag = "【相关性确立死叉反转】高位恐慌共振正式见顶回落！无差别抛售结束，抄底黄金信号激活。"
            elif c_f < c_s:
                corr_diag = "【相关性处于空头退潮】市场同频恐慌持续消退，资金逐步恢复理性。"
            else:
                corr_diag = "【相关性低位蓄势震荡】全市场处于非共振状态，个股走势呈现常态化独立性。"

        # 2. 离散度微观动能
        market_high = data['SPY'].iloc[-1] > data['SPY'].rolling(50).mean().iloc[-1]
        try: market_complacent = c_s < corr_q25.iloc[-1]
        except: market_complacent = False
        
        disp_breaking_up = d_f > d_s
        breadth_top_active = market_high and market_complacent and disp_breaking_up
        
        if d_spot > d_f > d_s:
            disp_diag = "【离散度强劲多头加速】市场撕裂极度恶化！巨头掩护出货迹象显著，大盘广度死穴正在拉响警报。"
        elif breadth_top_active:
            disp_diag = "【离散度确定金叉突破】大盘高位自满，但分化动能突破！确立权重抱团/个股失血的终极见顶风控信号。"
        elif d_f < d_s:
            disp_diag = "【离散度处于分化收敛】两极分化动能减弱，个股收益率分布趋同，结构性撕裂暂时缓解。"
        else:
            disp_diag = "【离散度常态化震荡】无极端抱团或分裂，板块轮动相对均衡。"

        # 3. 联合象限矩阵
        if (c_f > c_s) and (d_f <= d_s):
            combined_diag = "📊 【资金象限：泥沙俱下】流动性宣泄"
        elif (c_f <= c_s) and (d_f > d_s):
            combined_diag = "🚨 【资金象限：结构撕裂】权重抱团失血"
        elif (c_f <= c_s) and (d_f <= d_s):
            combined_diag = "⏳ 【资金象限：低波自满】常态健康运行"
        else:
            combined_diag = "⚡ 【资金象限：宏观剧震】风格暴烈洗牌"

        cboe_corr_text = f"当前:{c_spot:.2f} (EMA5:{c_f:.2f} / EMA21:{c_s:.2f})"
        cboe_disp_text = f"当前:{d_spot:.2f} (EMA5:{d_f:.2f} / EMA21:{d_s:.2f})"
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
            "df_hist": df_hist.tail(60)
        }
    except Exception as e:
        return {
            "error": True, "msg": str(e), 
            "cta_bottom_active": False, "cta_top_active": False,
            "corr_bottom_active": False, "breadth_top_active": False,
            "corr_diag": "诊断异常", "disp_diag": "诊断异常", "combined_diag": "诊断异常"
        }

# -----------------------------------------------------------------------------
# 3. 业务决策逻辑组装与元数据解析
# -----------------------------------------------------------------------------
vix_data = fetch_vix_data()
crypto_data = fetch_crypto_signals()
sm_data = fetch_squeezemetrics_data()
quant_data = calculate_quant_and_breadth_signals()

now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

# 开关 1 与 开关 5 合并逻辑解析
if not sm_data["error"]:
    gex_val = sm_data.get('gex', 0)
    dix_val = sm_data.get('dix', 44.0)
    
    sm_bottom_active = (gex_val > 0) and (dix_val >= 45.0)
    sm_top_active = (gex_val < 0) or (dix_val < 40.0)
    
    if sm_data.get("is_mock", False):
        sm_status = "使用兜底数据 🟡"
    elif sm_top_active:
        sm_status = "🚨 风险重压：空头敞口/机构派发"
    elif sm_bottom_active:
        sm_status = "🟢 信号激活：Gamma护盘/暗池吸筹"
    else:
        sm_status = "⚪ 状态中性：主力资金中性均衡"
else:
    sm_bottom_active = False
    sm_top_active = False
    sm_status = "数据抓取失败 🔴"
    gex_val, dix_val = 0, 0.0

# 构建全新的 5 大联合雷达开关数组
switches = [
    {
        "id": 1,
        "name": "做市商 Gamma & 暗池 DIX 联合资产开关",
        "bottom_active": sm_bottom_active,
        "top_active": sm_top_active,
        "value": f"GEX: {gex_val:,} | DIX: {dix_val}%",
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
        "value": f"今日比率: {vix_data.get('ratio', 'N/A')} (昨日: {vix_data.get('prev_ratio', 'N/A')}) | VIX: {vix_data.get('vix', 'N/A')}",
        "source": "CBOE 波动率曲线",
        "desc_bottom": "【抄底激活标准：比率跨线由昨日 <=1 突变为今日 >1】当隐含波动率期限结构打破极度深度倒挂状态、向上收复平衡线时激活，标志着非理性抛售流动性枯竭，转入安全抄底期。",
        "desc_top": "【逃顶激活标准：比率从昨日 >=1 转为今日 <1，或今日比率突破 >1.24】期限结构基石意外松动，或者Contango升水极度超载，显示风险资产做空波动率策略无脑拥挤，极易诱发多杀多踩踏性闪崩。",
        "fetched_status": "数据抓取失败 🔴" if vix_data["error"] else (
            f"{vix_data.get('vix_diag_status')} | 诊断详情：{vix_data.get('vix_desc_bottom')} {vix_data.get('vix_desc_top')}"
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
        "desc_bottom": "主跌浪贯穿多周期均线且负乖离达极限。量化 CTA 的约跟空抛压面临彻底耗尽。",
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
        # 新修改：不显示数值，直接在定位栏（value）显示相关性微观动能和离散度微观动能
        "value": f"相关性微观动能: {quant_data.get('corr_diag', '无信息')} | 离散度微观动能: {quant_data.get('disp_diag', '无信息')}",
        "source": "CBOE COR1M / DSPX 指数 (EMA5 与 EMA21)",
        "desc_bottom": quant_data.get("corr_diag", "无诊断信息"),
        "desc_top": quant_data.get("disp_diag", "无诊断信息"),
        "fetched_status": "数据抓取失败 🔴" if quant_data["error"] else quant_data.get("combined_diag", "无诊断信息"),
        "update_cycle": "每 1 小时",
        "last_updated": now_str
    }
]

# 综合判断当前系统状态
bottom_score = sum([1 for s in switches if s["bottom_active"]])
top_score = sum([1 for s in switches if s["top_active"]])
neutral_score = 5 - bottom_score - top_score

if top_score >= 3 and top_score >= bottom_score:
    status_color = "red"
    action_title = "🚨 【红色防御：触发系统性风控见顶逃顶线】"
    action_text = (
        f"<b>大盘底层资金面诊断</b>：当前系统 5 大核心开关中有 <b>{top_score}</b> 项联合拉响见顶警报，市场呈现极度贪婪或高位严重分裂！<br>"
        "做市商Gamma敞口恶化引发追跌放大效应，配合暗池主力高位派发及多头边际买盘枯竭。此时必须<b>全面收紧个股诊断模型的止盈线</b>，"
        "转入全面战略防御，严控多头杠杆。"
    )
elif bottom_score >= 3 and bottom_score >= top_score:
    status_color = "green"
    action_title = "🚀 【绿色共振：底部恐慌宣泄枯竭，触发拐点重仓抄底】"
    action_text = (
        f"<b>大盘底层资金面诊断</b>：当前系统 5 大核心开关中有 <b>{bottom_score}</b> 项指标达成共振，黄金左侧买点确立！<br>"
        "做市商Gamma转正提供防护，暗池机构大单托底，且离岸高杠杆和系统性CTA抛压均已砸至历史冰点。多头精准抄底模式全面启动，"
        "建议结合个股的 Expected Value 与 Random Forest 模型全力捕捉大盘错杀的EV红利股。"
    )
else:
    status_color = "orange"
    action_title = "⏳ 【黄色震荡：多空状态均衡交织，常规结构分化轮动】"
    action_text = (
        f"<b>大盘底层资金面诊断</b>：当前系统无极端单边共振信号（抄底激活:<b>{bottom_score}</b> | "
        f"见顶风控:<b>{top_score}</b> | 状态中性:<b>{neutral_score}</b>）。"
        "全市场流动性在巨头与权重股之间常规轮动，未形成系统性突变尾部风险。建议维持常态化均衡仓位，实施中性对冲策略，"
        "持续对目标个股执行日线级别常规诊断。"
    )

# -----------------------------------------------------------------------------
# 4. Streamlit UI 界面绘制
# -----------------------------------------------------------------------------
st.title("🛡️ Sentinel 2.0 核心决策系统：大盘底层资金双向雷达")
st.subheader(f"看板渲染时钟: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
        
        metadata_line = f'<div style="margin-top: 10px; padding-top: 6px; border-top: 1px dashed #e0e0e0; font-size: 8.5pt; color: #7f8c8d;"><span style="float: left;">⏱️ {s.get("update_cycle", "未知")}</span><span style="float: right; font-family: monospace;">📅 {s.get("last_updated", "实时")}</span><div style="clear: both;"></div></div>'
            
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
            {metadata_line}
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🐳 开关1: 机构暗池 DIX & 做市商 GEX 联合敞口",
    "📊 开关2: VIX 期限结构趋势", 
    "₿ 开关3: 加密离岸高杠杆费率", 
    "🤖 开关4: CTA 动量极值矩阵", 
    "🔄 开关5: 隐含相关性与离散度快慢线"
])

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
            fig_vix_ratio.add_trace(go.Scatter(x=v_df.index, y=v_df['Ratio'], name="VIX3M / VIX 比率", line=dict(color="#9b59b6", width=2)))
            fig_vix_ratio.add_hline(y=1.0, line_dash="dash", line_color="#2ecc71", annotation_text="Contango 线 (1.0)")
            fig_vix_ratio.add_hline(y=1.24, line_dash="dash", line_color="#e74c3c", annotation_text="极端自满 (1.24)")
            fig_vix_ratio.update_layout(title_text="图表 B: VIX3M / VIX 期限结构比率", template="plotly_white", height=400)
            st.plotly_chart(fig_vix_ratio, use_container_width=True)

# --- TAB 3 ---
with tab3:
    if not crypto_data["error"] and crypto_data.get("hist_df") is not None:
        c_df = crypto_data["hist_df"]
        fig_crypto = go.Figure()
        fig_crypto.add_trace(go.Scatter(x=c_df['fundingTime'], y=c_df['fundingRate'], name="BTC 资金费率 (%)", line=dict(color="#f1c40f", width=2)))
        fig_crypto.add_hline(y=0.0, line_dash="solid", line_color="#7f8c8d")
        fig_crypto.add_hline(y=0.01, line_dash="dash", line_color="#e74c3c", annotation_text="多头超载边界 (>=0.01%)")
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

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

@st.cache_data(ttl=3600)
def fetch_vix_data():
    try:
        tickers = yf.Tickers('^VIX ^VIX3M')
        hist = tickers.history(period='3mo')  
        if not hist.empty and 'Close' in hist.columns:
            close_data = hist['Close'].ffill().copy()
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
                "fetched_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False, "fetched_at": "异常断流"}
    return {"error": True, "msg": "No data", "bottom_active": False, "top_active": False, "fetched_at": "空数据"}

@st.cache_data(ttl=1800)
def fetch_crypto_signals():
    try:
        # 1. 获取 BTC 日级别 K 线数据 (判断价格趋势)
        # 使用 yfinance 获取最近5天的日线数据
        btc_df = yf.download('BTC-USD', period='5d', interval='1d', progress=False)['Close']
        btc_df = btc_df.ffill()
        current_price = btc_df.iloc[-1].item() if isinstance(btc_df.iloc[-1], pd.Series) else btc_df.iloc[-1]
        prev_price = btc_df.iloc[-2].item() if isinstance(btc_df.iloc[-2], pd.Series) else btc_df.iloc[-2]
        
        price_up = current_price > prev_price
        price_trend_str = f"上涨 (较昨日)" if price_up else f"下跌 (较昨日)"

        # 2. 获取 OKX 实时资金费率
        fr_url = "https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP"
        fr_res = requests.get(fr_url, timeout=5).json()
        if fr_res.get("code") != "0":
            raise Exception("OKX 费率 API 异常")
        funding_rate = float(fr_res['data'][0]['fundingRate']) * 100

        # 3. 获取 OKX 日级别历史持仓量 (OI) 趋势
        # 使用 OKX Rubik 统计接口获取日级别的 OI 变化
        rubik_url = "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume?ccy=BTC&period=1D"
        rubik_res = requests.get(rubik_url, timeout=5).json()
        
        oi_up = False
        oi_trend_str = "未知"
        current_oi = 0
        
        if rubik_res.get("code") == "0" and len(rubik_res.get("data", [])) >= 2:
            # 数据格式: [时间戳, OI(币本位), 交易量...]
            current_oi = float(rubik_res['data'][0][1])
            prev_oi = float(rubik_res['data'][1][1])
            oi_up = current_oi > prev_oi
            oi_trend_str = "大幅上升" if current_oi > prev_oi * 1.05 else ("上升" if oi_up else "下降")
        else:
            # 如果历史接口失败，降级使用实时快照接口
            oi_url = "https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP"
            oi_res = requests.get(oi_url, timeout=5).json()
            current_oi = float(oi_res['data'][0]['oiCcy'])
            oi_trend_str = "无历史对比"

        # 4. 核心逻辑矩阵：组合诊断 (价格 + OI + 费率)
        bottom_active = False
        top_active = False
        
        if price_up and oi_up and funding_rate >= 0.01:
            diag_status = "🚨 【极度危险/逃顶】价格上涨 ✖ OI飙升 ✖ 费率极高(>=0.01%)。杠杆多头极度拥挤，随时可能引发“插针”连环爆仓。"
            top_active = True
        elif not price_up and oi_up and funding_rate < 0:
            diag_status = "🟡 【轧空预警/博弈】价格下跌 ✖ OI飙升 ✖ 费率转负。空头大举建仓，需严防主力突然拉升触发“逼空(Short Squeeze)”暴力反弹。"
        elif not price_up and not oi_up and funding_rate <= 0:
            diag_status = "🟢 【黄金右侧/抄底】价格下跌 ✖ OI下降 ✖ 费率触底/转负。恐慌盘宣泄完毕，多头爆仓出清，杠杆泡沫刺破，极佳的左侧/右侧买点。"
            bottom_active = True
        elif price_up and not oi_up:
            diag_status = "⚠️ 【假突破预警/缩量】价格上涨 ✖ OI下降。缺乏新现货资金追高，纯靠空头平仓（踏空回补）推动，上涨大概率不可持续。"
            top_active = True
        elif price_up and oi_up and 0 <= funding_rate < 0.01:
            diag_status = "📈 【健康延续/持仓】价格上涨 ✖ OI上升 ✖ 费率正常。真金白银健康流入市场，多头趋势良性延续。"
        else:
            diag_status = f"⚪ 【震荡博弈/观望】价格{price_trend_str}，OI{oi_trend_str}，费率({funding_rate:.4f}%) 未达极端阈值，方向不明确。"

        return {
            "btc_price": f"${current_price:,.2f}",
            "price_trend": price_trend_str,
            "oi": f"{current_oi:,.0f} BTC",
            "oi_trend": oi_trend_str,
            "funding_rate": f"{funding_rate:.4f}%", 
            "diag_status": diag_status,
            "bottom_active": bottom_active, 
            "top_active": top_active, 
            "error": False,
            "fetched_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False, "fetched_at": "网络穿透异常"}

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
        "fetched_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " (兜底)"
    }

@st.cache_data(ttl=14400)
def fetch_cboe_official_history(symbol):
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
    try:
        raw_tickers = ['QQQ', 'SPY', 'IWM', 'RSP', '^COR1M', '^DSPX']
        yf_tickers = filter_leveraged_etfs(raw_tickers)
        
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
            elif c_golden_cross and not d_is_high:
                combined_diag = "🔥 【象限 II: 泥沙俱下】纯粹的同频无差别恐慌抛售，相关性极高（等待 CBOE 快慢线死叉即可抄底）"
            elif c_dead_cross and d_is_high:
                combined_diag = "🚨 【象限 III: 极致撕裂】大盘失真，资金极致抱团超级权重，掩护中小盘出货（触发终极广度逃顶线）"
            else:
                combined_diag = "⏳ 【象限 IV: 均衡收敛】常态低波运行，系统性风险真空期，个股特异性健康回归"

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
            "df_hist": df_hist.tail(60)
        }
    except Exception as e:
        return {
            "error": True, "msg": str(e), 
            "cta_bottom_active": False, "cta_top_active": False,
            "corr_bottom_active": False, "breadth_top_active": False,
            "corr_diag": "诊断异常", "disp_diag": "诊断异常", "combined_diag": "诊断异常"
        }
        
@st.cache_data(ttl=3600)
def fetch_vxn_vix_data():
    try:
        tickers = yf.Tickers('^VXN ^VIX')
        hist = tickers.history(period='3mo')
        
        if not hist.empty and 'Close' in hist.columns:
            df = hist['Close'].ffill().copy()
            
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
                "fetched_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    except Exception as e:
        return {"error": True, "msg": str(e), "bottom_active": False, "top_active": False, "fetched_at": "异常断流"}
    return {"error": True, "msg": "No data", "bottom_active": False, "top_active": False, "fetched_at": "空数据"}
    
# -----------------------------------------------------------------------------
# 3. 业务决策逻辑组装与元数据解析
# -----------------------------------------------------------------------------
vix_data = fetch_vix_data()
crypto_data = fetch_crypto_signals()
sm_data = fetch_squeezemetrics_data()
quant_data = calculate_quant_and_breadth_signals()

now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
vxn_vix_data = fetch_vxn_vix_data()

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
        "update_cycle": "每日更新 (美东盘后)",
        "last_updated": now_str
    },
    {
            "id": 2,
            "name": "VIX 期限结构与趋势动能雷达",
            "bottom_active": vix_data["bottom_active"] if not vix_data["error"] else False,
            "top_active": vix_data["top_active"] if not vix_data["error"] else False,
            "value": f"今日比率: {vix_data.get('ratio', 'N/A')} | EMA5/21状态: {'快线上穿/多头' if vix_data['bottom_active'] else '死叉/发散'} | VIX现货: {vix_data.get('vix', 'N/A')}",
            "source": "CBOE 波动率期限结构交叉矩阵",
            "desc_bottom": "【双向修复抄底标准】当隐含波动率比率向上收复突破 1.0 平衡线（摆脱远期深度倒挂状态），或者在低位倒挂修复带(<=1.05)确立微观动能均线金叉（EMA5 > EMA21）时激活。这标志着全市场非理性非对称抛售流动性枯竭，买盘筹码右侧转折确立，转入高胜率抄底期。",
            "desc_top": "【三维立体逃顶标准】满足以下任一核心条件立即拉响风控防御：①比率冲破 1.25 绝对贪婪上限，期权空头无防备极度拥挤；②比率跌破 1.0 平衡线，长短期期限结构倒挂、牛市基石全面动摇；③比率在高位自满警戒带(>=1.15)发生了 EMA5 下穿 EMA21 死叉，显示做多边际买盘已经枯竭见顶。",
            "fetched_status": "数据抓取失败 🔴" if vix_data["error"] else (
                f"<b>当下状态：</b>{vix_data.get('vix_diag_status')}<br>"
                f"<b>⚖️ 比率动能分项：</b>{vix_data.get('vix_ratio_diag')}<br>"
                f"<b>📊 现货波动分项：</b>{vix_data.get('vix_spot_diag')}"
            ),
            "update_cycle": "盘中实时波动 (YF延迟)",
            "last_updated": now_str
        },
    {
        "id": 3,
        "name": "加密离岸高杠杆流动性前哨 (Price+OI+FR 矩阵)",
        "bottom_active": crypto_data["bottom_active"] if not crypto_data["error"] else False,
        "top_active": crypto_data["top_active"] if not crypto_data["error"] else False,
        "value": f"BTC现货: {crypto_data.get('btc_price', 'N/A')} ({crypto_data.get('price_trend', '')}) | OI: {crypto_data.get('oi', 'N/A')} ({crypto_data.get('oi_trend', '')}) | 费率: {crypto_data.get('funding_rate', 'N/A')}",
        "source": "Yahoo Finance (K线) ✖ OKX 永续合约实时 API (OI与资金费率)",
        "desc_bottom": "【缩量爆仓抄底】当 **价格下跌 + OI显著下降 + 费率转负**。代表做多杠杆被彻底清算，市场流动性恐慌见底，是高胜率左侧或右侧建仓点。",
        "desc_top": "【拥挤过载逃顶】触发两种情况立即防御：① **价格上涨 + OI上升 + 费率极高** (多头拥挤，极易被爆)；② **价格上涨 + OI下降** (缺乏新资金的假突破)。",
        "fetched_status": "数据抓取失败 🔴" if crypto_data["error"] else (
            f"<div style='background-color:#f4f6f7; padding:8px; border-radius:5px; margin-bottom:5px; font-weight:bold; color:#2c3e50;'>{crypto_data.get('diag_status')}</div>"
        ),
        "update_cycle": "日线级别清洗 ✖ 盘中实时快照",
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
        "update_cycle": "盘中动态计算 (基于最新价)",
        "last_updated": now_str
    },
    {
        "id": 5,
        "name": "全局隐含相关性拐点与离散度爆发矩阵 (全象限版)",
        "bottom_active": quant_data["corr_bottom_active"] if not quant_data["error"] else False,
        "top_active": quant_data["breadth_top_active"] if not quant_data["error"] else False,
        "value": f"{quant_data.get('cboe_corr', 'N/A')} | {quant_data.get('cboe_disp', 'N/A')}",
        "source": "CBOE COR1M / DSPX 联动矩阵 (微观导数交叉 ✖ 滚动Z-Score状态机)",
        "desc_bottom": "【抄底激活：恐慌死叉✖撕裂收敛】当相关性极值冲顶后向下死叉确立（恐慌抛售衰退），且离散度未出现背离爆发时激活。此时大盘无差别抛压清空，回归估值红利期。",
        "desc_top": "【逃顶激活：隐蔽自满✖抱团发散】当指数高位且相关性极弱（掩饰流动性干涸），但离散度暴拉金叉（极少数巨头拉盘掩护出货）时激活。此时大盘广度深度崩塌，触发终极防御。",
        "fetched_status": "数据抓取失败 🔴" if quant_data["error"] else (
            f"<div style='background-color:#f4f6f7; padding:8px; border-radius:5px; margin-bottom:5px; font-weight:bold; color:#d35400;'>{quant_data.get('combined_diag', '无信息')}</div>"
            f"<b>📊 相关性微观动能：</b>{quant_data.get('corr_diag', '无信息')}<br>"
            f"<b>📉 离散度微观动能：</b>{quant_data.get('disp_diag', '无信息')}"
        ),
        "update_cycle": "每日更新 (盘终结算)",
        "last_updated": now_str
    },
    {
        "id": 6,
        "name": "VXN-VIX 科技股雷达",
        "bottom_active": vxn_vix_data["bottom_active"] if not vxn_vix_data["error"] else False,
        "top_active": vxn_vix_data["top_active"] if not vxn_vix_data["error"] else False,
        "value": f"Spread: {vxn_vix_data.get('current_spread', 'N/A')} | Ratio: {vxn_vix_data.get('current_ratio', 'N/A')} | 熔断风控实时检测",
        "source": "CBOE 波动率剪刀差 & EMA 一阶导数交叉",
        "desc_bottom": "【右侧出击】当剪刀差自高位（>8.0）回落，且微观动能死叉（EMA5 < EMA21）时激活。此时非对称踩踏结束，IV Crush 来临，是高弹性科技股胜率极高的反转买点。",
        "desc_top": "【双重风控防御】① 火山口（单边踩踏）：极高位金叉发散，无条件熔断科技股多头；② 暴风雨前夜（隐性筑顶）：低位自满区间突发金叉，主力悄然买入 Put，需立刻收紧止盈或做空保护。",
        "fetched_status": "数据抓取失败 🔴" if vxn_vix_data["error"] else (
            f"<div style='background-color:#f4f6f7; padding:8px; border-radius:5px; margin-bottom:5px; font-weight:bold; color:#d35400;'>{vxn_vix_data.get('combined_diag', '无信息')}</div>"
            f"<b>📊 微观动能：</b>{vxn_vix_data.get('spread_diag', '无信息')}<br>"
            f"<b>📉 情绪象限：</b>{vxn_vix_data.get('ratio_diag', '无信息')}"
        ),
        "update_cycle": "盘中实时波动 (YF延迟)",
        "last_updated": now_str
    }
]

bottom_score = sum([1 for s in switches if s["bottom_active"]])
top_score = sum([1 for s in switches if s["top_active"]])
neutral_score = len(switches) - bottom_score - top_score 

if top_score >= 3 and top_score >= bottom_score:
    status_color = "red"
    action_title = "🚨 【红色防御：触发系统性风控见顶逃顶线】"
    action_text = (
        f"<b>大盘底层资金面诊断</b>：当前系统 {len(switches)} 大核心开关中有 <b>{top_score}</b> 项联合拉响见顶警报，市场呈现极度贪婪或高位严重分裂！<br>"
        "做市商Gamma敞口恶化引发追跌放大效应，配合暗池主力高位派发及多头边际买盘枯竭。此时必须<b>全面收紧个股诊断模型的止盈线</b>，"
        "转入全面战略防御，严控多头杠杆。"
    )
elif bottom_score >= 3 and bottom_score >= top_score:
    status_color = "green"
    action_title = "🚀 【绿色共振：底部恐慌宣泄枯竭，触发拐点重仓抄底】"
    action_text = (
        f"<b>大盘底层资金面诊断</b>：当前系统 {len(switches)} 大核心开关中有 <b>{bottom_score}</b> 项指标达成共振，黄金左侧买点确立！<br>"
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
            <p class="switch-value"><b>核心底层定位:</b> <span style="font-family: monospace; color:#2980b9; font-weight:bold;">{s['value']}</span></p>
            <div class="switch-status"><b>📡 数据状态:</b> <span>{s['fetched_status']}</span></div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔘 点击展开：多空防守边界逻辑", expanded=False):
            st.markdown(f"""
            <div class="switch-boundary-panel">
                <p><b>📈 多头见底边界:</b> <span style="color:#27ae60;">{safe_desc_bottom}</span></p>
                <p><b>📉 空头防守边界:</b> <span style="color:#c0392b;">{safe_desc_top}</span></p>
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
# 假设这是你的 UI 渲染部分
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

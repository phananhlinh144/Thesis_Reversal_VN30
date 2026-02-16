import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from vnstock import Vnstock

# ==============================================================================
# 1. CẤU HÌNH & CACHE MODEL (Fix Warning Streamlit)
# ==============================================================================
st.set_page_config(page_title="VN30 AI Forecast 2026", layout="wide")

MODEL_WIN50_PATH = 'models_scaling/Full_K10_Win50_Hybrid.keras'
MODEL_WIN10_PATH = 'models_scaling/Baseline_K10_Win10_Hybrid.keras'
SCALER_PATH      = 'models_scaling/smart_scaler_system.pkl'
CSV_PATH         = 'vn30_data_raw.csv'

FINAL_FEATURES = ['RC_1', 'RC_2', 'RC_3', 'RC_5', 'RC_8', 'RC_13', 'RC_21', 'RC_34', 'RC_55',
                  'Grad_5', 'Grad_10', 'Grad_20', 'RSI', 'BB_PctB', 'MACD_Hist', 'Vol_Ratio', 'ATR_Rel']
FEATS_FULL = FINAL_FEATURES + ['Dist_Prev_K10']

@st.cache_resource
def load_ai_system():
    try:
        m50 = tf.keras.models.load_model(MODEL_WIN50_PATH)
        m10 = tf.keras.models.load_model(MODEL_WIN10_PATH)
        scaler_data = joblib.load(SCALER_PATH)
        return m50, m10, scaler_data
    except Exception as e:
        st.error(f"Lỗi tải Model: {e}")
        return None, None, None

model_win50, model_win10, scaler_bundle = load_ai_system()

# ==============================================================================
# 2. HÀM XỬ LÝ (Fix FutureWarnings)
# ==============================================================================

def compute_features(df):
    g = df.copy()
    if len(g) < 60: return pd.DataFrame()
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
    # Fix: dùng .bfill() thay vì fillna(method='bfill')
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean()
        g[f'Grad_{n}'] = np.gradient(ma.bfill())
        
    g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
    g['RSI'] = ta.rsi(g['Close'], length=14)
    bb = ta.bbands(g['Close'], length=20, std=2)
    g['BB_PctB'], g['BB_Upper'], g['BB_Lower'] = bb.iloc[:, 4], bb.iloc[:, 2], bb.iloc[:, 0]
    g['MACD_Hist'] = ta.macd(g['Close']).iloc[:, 1]
    g['ATR_Rel'] = ta.atr(g['High'], g['Low'], g['Close'], length=14) / g['Close']
    
    rmin, rmax = g['Close'].rolling(20).min(), g['Close'].rolling(20).max()
    ma20 = g['Close'].rolling(20).mean()
    g['Dist_Prev_K10'] = 0.0
    g.loc[g['Close'] >= ma20, 'Dist_Prev_K10'] = (g['Close'] - rmin) / rmin
    g.loc[g['Close'] < ma20, 'Dist_Prev_K10'] = (g['Close'] - rmax) / rmax
    return g.dropna().reset_index(drop=True)

def get_data_for_symbol(symbol, fetch_live=True):
    try:
        df_hist = pd.read_csv(CSV_PATH)
        df_hist = df_hist[df_hist['Symbol'] == symbol].copy()
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        df_hist = df_hist.sort_values('Date')
        if fetch_live:
            try:
                live = Vnstock().stock(symbol=symbol, source='VCI').quote.now()
                if not live.empty:
                    p, v, h, l = float(live['close'].iloc[0]), float(live['volume'].iloc[0]), float(live['high'].iloc[0]), float(live['low'].iloc[0])
                    today = pd.Timestamp(datetime.now().date())
                    if df_hist.empty or df_hist.iloc[-1]['Date'].date() < today.date():
                        df_hist = pd.concat([df_hist, pd.DataFrame([{'Date':today,'Open':p,'High':h,'Low':l,'Close':p,'Volume':v,'Symbol':symbol}])], ignore_index=True)
                    else:
                        idx = df_hist.index[-1]
                        df_hist.at[idx,'Close'], df_hist.at[idx,'High'], df_hist.at[idx,'Low'], df_hist.at[idx,'Volume'] = p, max(df_hist.at[idx,'High'],h), min(df_hist.at[idx,'Low'],l), v
            except: pass
        return df_hist
    except: return pd.DataFrame()

def predict_single_row(df_calc, idx_target=-1, symbol=''):
    if len(df_calc) < 55: return None
    end = idx_target + 1 if idx_target != -1 else len(df_calc)
    d50, d10 = df_calc.iloc[end-50:end], df_calc.iloc[end-10:end]
    scaler = scaler_bundle['local_scalers_dict'].get(symbol, scaler_bundle['global_scaler'])
    s50, s10 = scaler.transform(d50[FEATS_FULL]), scaler.transform(d10[FEATS_FULL])
    p50 = model_win50.predict(np.expand_dims(s50, axis=0), verbose=0)[0]
    p10 = model_win10.predict(np.expand_dims(s10[:, :17], axis=0), verbose=0)[0]
    c50, c10 = np.argmax(p50), np.argmax(p10)
    sig = "NGANG"
    if c50 == 0 and c10 == 0: sig = "MUA"
    elif c50 == 2 and c10 == 2: sig = "BÁN"
    return {'Date': df_calc.iloc[end-1]['Date'], 'Close': df_calc.iloc[end-1]['Close'], 'Ensemble': sig,
            'Model_50': f"{['mua','ngang','bán'][c50]} ({p50[c50]:.0%})", 'Model_10': f"{['mua','ngang','bán'][c10]} ({p10[c10]:.0%})"}

# ==============================================================================
# 3. GIAO DIỆN CHÍNH
# ==============================================================================
st.title("🤖 VN30 AI TRADING SYSTEM")
vn30_list = ['ACB','BCM','BID','CTG','DGC','FPT','GAS','GVR','HDB','HPG','LPB','MSN','MBB','MWG','PLX','SAB','SHB','SSB','SSI','STB','TCB','TPB','VCB','VIC','VHM','VIB','VJC','VNM','VPB','VRE']
t1, t2, t3 = st.tabs(["📊 DỰ BÁO CHUNG", "📈 BIỂU ĐỒ CHI TIẾT", "📝 DỮ LIỆU CHI TIẾT"])

with t1:
    if st.button("🚀 QUÉT TOÀN BỘ VN30"):
        res_all = []
        bar = st.progress(0)
        for i, s in enumerate(vn30_list):
            d = compute_features(get_data_for_symbol(s))
            r = predict_single_row(d, symbol=s)
            if r: res_all.append({'Mã':s, 'Giá':f"{int(r['Close']):,}", 'AI':r['Ensemble'], 'Dài':r['Model_50'], 'Ngắn':r['Model_10']})
            bar.progress((i+1)/len(vn30_list))
        df_res = pd.DataFrame(res_all)
        c_mua, c_ban, c_ngang = st.columns(3)
        with c_mua: st.success("🟢 MUA"); st.table(df_res[df_res['AI']=='MUA'])
        with c_ban: st.error("🔴 BÁN"); st.table(df_res[df_res['AI']=='BÁN'])
        with c_ngang: st.warning("🟡 NGANG"); st.table(df_res[df_res['AI']=='NGANG'])

with t2:
    col_sym, col_type = st.columns([2, 2])
    with col_sym: s_chart = st.selectbox("Chọn mã:", vn30_list, key='s2')
    with col_type: v_type = st.radio("Kiểu giá:", ["Nến", "Đường"], horizontal=True)
    
    if s_chart:
        df_c = compute_features(get_data_for_symbol(s_chart))
        plot_df = df_c.tail(60)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # 1. Biểu đồ giá (Candle or Line)
        if v_type == "Nến":
            fig.add_trace(go.Candlestick(x=plot_df['Date'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='Nến'), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Close'], mode='lines+markers', name='Đường giá', line=dict(color='blue')), row=1, col=1)
        
        # 2. RSI với vạch 70/30 đúng hình mẫu
        fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

with t3:
    c_s, c_d = st.columns([1, 2])
    with c_s: s_tab3 = st.selectbox("Chọn mã:", vn30_list, key='s3')
    with c_d: lookback = st.slider("Xem lại bao nhiêu ngày?", 5, 30, 10)
    
    if s_tab3:
        df_c = compute_features(get_data_for_symbol(s_tab3))
        # Fix Warning applymap -> map
        re_list = [predict_single_row(df_c, idx_target=i, symbol=s_tab3) for i in range(len(df_c)-1, len(df_c)-1-lookback, -1)]
        df_t = pd.DataFrame([r for r in re_list if r])
        st.dataframe(df_t.style.map(lambda v: f"color: {'#00CC00' if v=='MUA' else '#FF0000' if v=='BÁN' else '#FFBB00'}; font-weight: bold", subset=['Ensemble']), use_container_width=True)

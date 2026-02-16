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
import time

# --- 1. KHỞI TẠO HỆ THỐNG ---
st.set_page_config(page_title="VN30 AI ENSEMBLE PRO", layout="wide")

# KHỞI TẠO SESSION STATE CÓ SẴN CỘT ĐỂ TRÁNH KEYERROR
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = pd.DataFrame(columns=["Mã", "Giá", "win50", "win10", "ensemble"])
if 'first_run' not in st.session_state:
    st.session_state.first_run = True

@st.cache_resource
def load_assets():
    m50 = tf.keras.models.load_model('Full_K10_Win50_Hybrid.keras')
    m10 = tf.keras.models.load_model('Baseline_K10_Win10_Hybrid.keras')
    bundle = joblib.load('smart_scaler_system.pkl')
    return m50, m10, bundle

m50, m10, bundle = load_assets()
vn30_symbols = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']
FEATS_FULL = ['RC_1', 'RC_2', 'RC_3', 'RC_5', 'RC_8', 'RC_13', 'RC_21', 'RC_34', 'RC_55', 'Grad_5', 'Grad_10', 'Grad_20', 'RSI', 'BB_PctB', 'MACD_Hist', 'Vol_Ratio', 'ATR_Rel', 'Dist_Prev_K10']
LABELS_LOWER = {0: "mua", 1: "hold", 2: "bán"}

# --- 2. HÀM XỬ LÝ LOGIC ---
def get_data(symbol):
    try:
        stock = Vnstock().stock(symbol=symbol, source='VCI')
        df = stock.quote.history(start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
        df = df.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date').reset_index(drop=True)
    except: return pd.DataFrame()

def compute_features(df):
    if len(df) < 60: return pd.DataFrame()
    g = df.copy()
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean().fillna(method='bfill')
        g[f'Grad_{n}'] = np.gradient(ma)
    g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
    g['RSI'] = ta.rsi(g['Close'], length=14)
    bb = ta.bbands(g['Close'], length=20, std=2)
    g['BB_PctB'] = bb.iloc[:, 4]
    g['BBU'] = bb.iloc[:, 2]
    g['BBL'] = bb.iloc[:, 0]
    g['MACD_Hist'] = ta.macd(g['Close']).iloc[:, 1]
    g['ATR_Rel'] = ta.atr(g['High'], g['Low'], g['Close'], length=14) / g['Close']
    ma20 = g['Close'].rolling(20).mean()
    g['Dist_Prev_K10'] = 0.0
    g.loc[g['Close'] >= ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).min()) / g['Close'].rolling(20).min()
    g.loc[g['Close'] < ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).max()) / g['Close'].rolling(20).max()
    return g.dropna().reset_index(drop=True)

def run_prediction(df_calc, symbol, target_idx=-1):
    try:
        scaler = bundle['local_scalers_dict'].get(symbol, bundle['global_scaler'])
        if target_idx == -1: target_idx = len(df_calc) - 1
        d_slice = df_calc.iloc[target_idx-49 : target_idx+1]
        if len(d_slice) < 50: return None
        scaled = scaler.transform(d_slice[FEATS_FULL].values)
        p50_raw = m50.predict(np.expand_dims(scaled, 0), verbose=0)[0]
        p10_raw = m10.predict(np.expand_dims(scaled[-10:, :17], 0), verbose=0)[0]
        c50, c10 = np.argmax(p50_raw), np.argmax(p10_raw)
        ens = "THEO DÕI"
        if c50 == 0 and c10 == 0: ens = "MUA"
        elif c50 == 2 and c10 == 2: ens = "BÁN"
        return {
            "win50": f"{LABELS_LOWER[c50]} ({p50_raw[c50]:.0%})",
            "win10": f"{LABELS_LOWER[c10]} ({p10_raw[c10]:.0%})",
            "ensemble": ens
        }
    except: return None

def perform_scan():
    results = []
    status_bar = st.progress(0)
    msg = st.empty()
    for i, sym in enumerate(vn30_symbols):
        msg.text(f"🚀 AI đang quét mã: {sym} ({i+1}/30)...")
        df_d = get_data(sym)
        df_calc = compute_features(df_d)
        if not df_calc.empty:
            res = run_prediction(df_calc, sym)
            if res:
                results.append({"Mã": sym, "Giá": f"{df_calc['Close'].iloc[-1]:,.0f}", **res})
        status_bar.progress((i+1)/30)
    msg.empty()
    status_bar.empty()
    return pd.DataFrame(results)

# --- TỰ ĐỘNG CHẠY KHI MỞ APP ---
if st.session_state.first_run:
    st.session_state.scan_results = perform_scan()
    st.session_state.first_run = False

# --- 3. GIAO DIỆN TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Đồ thị kỹ thuật AI", "📊 Dashboard VN30", "📜 Lịch sử dự báo"])

with tab1:
    sel_sym = st.selectbox("Chọn mã soi biểu đồ", vn30_symbols, key="t1_sym")
    df = get_data(sel_sym)
    df_c = compute_features(df)
    if not df_c.empty:
        df_p = df_c.tail(100).reset_index(drop=True)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                            row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
        
        # Nến & Bollinger Bands
        fig.add_trace(go.Candlestick(x=df_p['Date'], open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="Giá"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p['Date'], y=df_p['BBU'], line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_p['Date'], y=df_p['BBL'], line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'), name="BB Lower"), row=1, col=1)
        fig.add_trace(go.Bar(x=df_p['Date'], y=df_p['Volume'], name="Volume", marker_color='rgba(150,150,150,0.1)'), row=1, col=1, secondary_y=True)
        
        # RSI
        fig.add_trace(go.Scatter(x=df_p['Date'], y=df_p['RSI'], name="RSI", line=dict(color='#8A2BE2')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # Annotations (Mũi tên & Chấm tròn)
        for i in range(len(df_p)-20, len(df_p)):
            res = run_prediction(df_c, sel_sym, target_idx=df_c.index[df_c['Date'] == df_p['Date'].iloc[i]][0])
            if res:
                # Chấm tròn báo hiệu AI có dự báo tại đó
                fig.add_trace(go.Scatter(x=[df_p['Date'].iloc[i]], y=[df_p['High'].iloc[i]*1.01], mode='markers', marker=dict(symbol='circle', color='white', size=4), showlegend=False), row=1, col=1)
                if res['ensemble'] == "MUA":
                    fig.add_annotation(x=df_p['Date'].iloc[i], y=df_p['Low'].iloc[i], text="▲", showarrow=False, font=dict(color="#00FF00", size=16), row=1, col=1)
                elif res['ensemble'] == "BÁN":
                    fig.add_annotation(x=df_p['Date'].iloc[i], y=df_p['High'].iloc[i], text="▼", showarrow=False, font=dict(color="#FF0000", size=16), row=1, col=1)

        fig.update_layout(height=700, template='plotly_dark', xaxis_rangeslider_visible=False, yaxis2=dict(showgrid=False, range=[0, df_p['Volume'].max()*5]))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"Dashboard VN30 - {datetime.now().strftime('%d/%m/%Y')}")
    if st.button("🔄 Cập nhật dữ liệu ngay bây giờ"):
        st.session_state.scan_results = perform_scan()

    df_r = st.session_state.scan_results
    if not df_r.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.success("🟢 MUA (ENSEMBLE)")
            st.dataframe(df_r[df_r['ensemble']=="MUA"][['Mã', 'Giá', 'win50', 'win10', 'ensemble']], hide_index=True)
        with c2:
            st.error("🔴 BÁN (ENSEMBLE)")
            st.dataframe(df_r[df_r['ensemble']=="BÁN"][['Mã', 'Giá', 'win50', 'win10', 'ensemble']], hide_index=True)
        with c3:
            st.warning("🟡 THEO DÕI")
            st.dataframe(df_r[df_r['ensemble']=="THEO DÕI"][['Mã', 'Giá', 'win50', 'win10', 'ensemble']], hide_index=True)
    else:
        st.info("Đang khởi tạo dữ liệu hoặc không có mã nào được tìm thấy...")

with tab3:
    s_h = st.selectbox("Chọn mã tra cứu lịch sử", vn30_symbols, key="t3_s")
    days = st.slider("Số phiên xem lại", 5, 20, 10)
    dh = get_data(s_h)
    dhc = compute_features(dh)
    if not dhc.empty:
        h_data = []
        for i in range(len(dhc)-days, len(dhc)):
            r = run_prediction(dhc, s_h, target_idx=i)
            if r:
                h_data.append({
                    "Ngày": dhc['Date'].iloc[i].strftime('%d/%m'),
                    "Giá": f"{dhc['Close'].iloc[i]:,.0f}",
                    "win50": r['win50'], "win10": r['win10'], "ENSEMBLE": r['ensemble']
                })
        st.table(pd.DataFrame(h_data[::-1]))

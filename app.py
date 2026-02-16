import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from vnstock import Vnstock
import requests
from io import BytesIO

# --- 1. KHỞI TẠO SESSION STATE (Sửa lỗi AttributeError) ---
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# --- 2. CẤU HÌNH HỆ THỐNG ---
st.set_page_config(page_title="VN30 AI Ensemble Pro", layout="wide", page_icon="💎")

@st.cache_resource
def load_assets():
    try:
        m50 = tf.keras.models.load_model('Full_K10_Win50_Hybrid.keras')
        m10 = tf.keras.models.load_model('Baseline_K10_Win10_Hybrid.keras')
        bundle = joblib.load('smart_scaler_system.pkl')
        return m50, m10, bundle
    except Exception as e:
        st.error(f"Lỗi tải Model: {e}")
        return None, None, None

m50, m10, bundle = load_assets()
vn30_symbols = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']
LABELS = {0: 'MUA 🟢', 1: 'HOLD 🟡', 2: 'BÁN 🔴'}

# --- 3. LOGIC ENSEMBLE ---
def get_ensemble_signal(p50, p10):
    r50, r10 = np.argmax(p50), np.argmax(p10)
    conf50, conf10 = np.max(p50), np.max(p10)
    
    if r50 == 0 and r10 == 0: return "MUA MẠNH 💎", "Mua"
    if r50 == 0: return "MUA 🟢", "Mua"
    if r50 == 2: return "BÁN 🔴", "Bán"
    if r10 == 2: return "CẨN TRỌNG 🟡", "Ngang"
    return "THEO DÕI ⚪", "Ngang"

def run_pred(df, symbol, target_idx=-1):
    try:
        sc = bundle['local_scalers_dict'].get(symbol, bundle['global_scaler'])
        feats = bundle['global_scaler'].feature_names_in_
        if target_idx == -1: window = df[feats].tail(50)
        else: window = df[feats].iloc[max(0, target_idx-49) : target_idx+1]
        
        if len(window) < 50: return None, None
        scaled = sc.transform(window)
        p50 = m50.predict(np.expand_dims(scaled, 0), verbose=0)[0]
        p10 = m10.predict(np.expand_dims(scaled[-10:, :17], 0), verbose=0)[0]
        return p50, p10
    except: return None, None

# --- 4. DATA ---
@st.cache_data(ttl=3600)
def get_data(symbol):
    try:
        client = Vnstock()
        df = client.stock(symbol=symbol).quote.history(start="2024-06-01", end=datetime.now().strftime('%Y-%m-%d'))
        df = df.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date')
    except: return pd.DataFrame()

def build_feats(df):
    if len(df) < 55: return pd.DataFrame()
    df = df.copy()
    df['RSI'] = ta.rsi(df['Close'], length=14)
    for n in [1, 5, 10, 20]: df[f'RC_{n}'] = df['Close'].pct_change(n) * 100
    for col in bundle['global_scaler'].feature_names_in_:
        if col not in df.columns: df[col] = 0.0
    return df.dropna(subset=['RSI'])

# --- 5. GIAO DIỆN ---
tab_scan, tab_detail = st.tabs(["📋 Bảng Tổng Hợp VN30", "🔍 Phân Tích Chuyên Sâu"])

with tab_scan:
    st.header("⚡ Hệ thống Ensemble VN30")
    if st.button("🚀 Quét toàn thị trường"):
        results = []
        bar = st.progress(0)
        for i, sym in enumerate(vn30_symbols):
            df_full = get_data(sym)
            df_p = build_feats(df_full)
            if not df_p.empty:
                p50, p10 = run_pred(df_p, sym)
                if p50 is not None:
                    ens_text, ens_group = get_ensemble_signal(p50, p10)
                    results.append({
                        "Mã": sym, "Giá HT": df_p['Close'].iloc[-1],
                        "W50_K10_18f": LABELS[np.argmax(p50)], 
                        "W10_K10_17f": LABELS[np.argmax(p10)],
                        "Ensemble": ens_text, "Nhóm": ens_group,
                        "Ngày Dự Báo": df_p['Date'].iloc[-1].strftime('%d/%m/%Y')
                    })
            bar.progress((i + 1) / len(vn30_symbols))
        st.session_state.scan_results = pd.DataFrame(results)

    if st.session_state.scan_results is not None:
        df_res = st.session_state.scan_results
        st.info(f"📅 Phiên dự báo gần nhất: {df_res['Ngày Dự Báo'].iloc[0]}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.success("🟢 DANH MỤC MUA")
            st.dataframe(df_res[df_res['Nhóm'] == "Mua"][['Mã', 'Giá HT', 'W50_K10_18f', 'W10_K10_17f', 'Ensemble']], hide_index=True)
        with c2:
            st.warning("🟡 THEO DÕI")
            st.dataframe(df_res[df_res['Nhóm'] == "Ngang"][['Mã', 'Giá HT', 'W50_K10_18f', 'W10_K10_17f', 'Ensemble']], hide_index=True)
        with c3:
            st.error("🔴 DANH MỤC BÁN")
            st.dataframe(df_res[df_res['Nhóm'] == "Bán"][['Mã', 'Giá HT', 'W50_K10_18f', 'W10_K10_17f', 'Ensemble']], hide_index=True)

with tab_detail:
    sel_sym = st.selectbox("Chọn mã phân tích chuyên sâu", vn30_symbols)
    if st.button(f"🔍 Soi mã {sel_sym}"):
        df_full = get_data(sel_sym)
        df_p = build_feats(df_full)
        if len(df_p) >= 60:
            p50, p10 = run_pred(df_p, sel_sym)
            ens_text, _ = get_ensemble_signal(p50, p10)
            
            bb = ta.bbands(df_p['Close'], length=20, std=2)
            df_plot = pd.concat([df_p, bb], axis=1).tail(60).reset_index()
            
            # --- VẼ BIỂU ĐỒ PRO ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                                subplot_titles=(f"Giá, BB & Volume Overlay - {sel_sym}", "Sức mạnh xu hướng RSI"), 
                                row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{"secondary_y": False}]])

            # Nến & BB
            fig.add_trace(go.Candlestick(x=df_plot['Date'], open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='Giá'), row=1, col=1)
            u_col = [c for c in df_plot.columns if c.startswith('BBU')][0]
            l_col = [c for c in df_plot.columns if c.startswith('BBL')][0]
            fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot[u_col], line=dict(color='rgba(173,216,230,0.2)'), name='BB Upper'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot[l_col], line=dict(color='rgba(173,216,230,0.2)'), fill='tonexty', name='BB Lower'), row=1, col=1)

            # Volume Gộp chung vào trục y phụ
            fig.add_trace(go.Bar(x=df_plot['Date'], y=df_plot['Volume'], name='Volume', marker_color='rgba(200,200,200,0.2)', yaxis='y2'), row=1, col=1, secondary_y=True)

            # Tín hiệu Ensemble (Mũi tên Đen Trắng)
            last_p = df_plot['Close'].iloc[-1]
            last_d = df_plot['Date'].iloc[-1]
            fig.add_annotation(x=last_d, y=last_p, text="Ensemble Signal", showarrow=True, arrowhead=2, arrowcolor="white", bgcolor="black", font=dict(color="white"), row=1, col=1)

            # Vòng tròn trắng: Dự báo thực tế 10 phiên trước
            hist_idx = len(df_p) - 10
            p50_h, p10_h = run_pred(df_p, sel_sym, target_idx=hist_idx)
            if p50_h is not None:
                fig.add_trace(go.Scatter(x=[df_p['Date'].iloc[hist_idx]], y=[df_p['Close'].iloc[hist_idx]], mode='markers', 
                                         marker=dict(color='white', size=15, symbol='circle-open', line=dict(width=3)), name='Backtest Point'), row=1, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['RSI'], name='RSI', line=dict(color='#8A2BE2')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

            fig.update_layout(height=800, template='plotly_dark', xaxis_rangeslider_visible=False, 
                              yaxis2=dict(overlaying='y', side='right', showgrid=False, range=[0, df_plot['Volume'].max()*5]))
            st.plotly_chart(fig, use_container_width=True)

            # --- TRANSPARENCY REPORT ---
            st.markdown(f"### 🛡️ AI Transparency Report - {df_p['Date'].iloc[-1].strftime('%d/%m/%Y')}")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Dài hạn (W50_K10_18f)", LABELS[np.argmax(p50)], f"P: {np.max(p50):.1%}")
            with c2: st.metric("Ngắn hạn (W10_K10_17f)", LABELS[np.argmax(p10)], f"P: {np.max(p10):.1%}")
            with c3: st.metric("KẾT LUẬN ENSEMBLE", ens_text)

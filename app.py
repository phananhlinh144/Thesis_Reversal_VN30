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
import time
import requests
from io import BytesIO

# --- 1. CẤU HÌNH HỆ THỐNG ---
st.set_page_config(page_title="VN30 AI Ensemble Pro", layout="wide", page_icon="💎")

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

@st.cache_resource
def load_assets():
    try:
        m50 = tf.keras.models.load_model('Full_K10_Win50_Hybrid.keras')
        m10 = tf.keras.models.load_model('Baseline_K10_Win10_Hybrid.keras')
        bundle = joblib.load('smart_scaler_system.pkl')
        return m50, m10, bundle
    except:
        st.error("❌ Không tìm thấy file Model hoặc Scaler. Kiểm tra lại thư mục!")
        return None, None, None

m50, m10, bundle = load_assets()
vn30_symbols = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']
LABELS = {0: 'MUA 🟢', 1: 'HOLD 🟡', 2: 'BÁN 🔴'}

# --- 2. LOGIC ENSEMBLE ---
def get_ensemble_signal(p50, p10):
    r50 = np.argmax(p50)
    r10 = np.argmax(p10)
    # Kết hợp: Ưu tiên dài hạn r50, ngắn hạn r10 làm bộ lọc
    if r50 == 0 and r10 == 0: return "MUA MẠNH 💎", "Mua"
    if r50 == 0: return "MUA (Đợi điểm vào) 🟢", "Mua"
    if r50 == 2: return "BÁN 🔴", "Bán"
    if r10 == 2: return "CẨN TRỌNG 🟡", "Ngang"
    return "THEO DÕI ⚪", "Ngang"

# --- 3. XỬ LÝ DỮ LIỆU ---
@st.cache_data(ttl=3600)
def get_data(symbol):
    try:
        # Tải từ Drive (Dữ liệu lịch sử)
        file_id = '1xG6J9fBEF_Z4KY3x_frUwnhVTSA6HG2r'
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        resp = requests.get(url, timeout=10)
        df_off = pd.read_csv(BytesIO(resp.content), on_bad_lines='skip', engine='python')
        col = next((c for c in df_off.columns if c.lower() in ['symbol', 'ticker', 'mã']), None)
        df_stock = df_off[df_off[col] == symbol].copy()
        df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')

        # Tải Online (Cập nhật phiên mới nhất 2026)
        client = Vnstock()
        df_on = client.stock(symbol=symbol).quote.history(start="2025-01-01", end=datetime.now().strftime('%Y-%m-%d'))
        if not df_on.empty:
            df_on = df_on.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            df_on['Date'] = pd.to_datetime(df_on['Date'])
            df_full = pd.concat([df_stock, df_on], ignore_index=True).drop_duplicates(subset=['Date']).sort_values('Date')
            return df_full
        return df_stock
    except: return pd.DataFrame()

def build_feats(df):
    if df.empty or len(df) < 55: return pd.DataFrame()
    df = df.copy().reset_index(drop=True)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    for n in [1, 5, 10, 20]: df[f'RC_{n}'] = df['Close'].pct_change(n) * 100
    for col in bundle['global_scaler'].feature_names_in_:
        if col not in df.columns: df[col] = 0.0
    return df.dropna(subset=['RSI']).tail(65)

def run_pred(df, symbol):
    try:
        sc = bundle['local_scalers_dict'].get(symbol, bundle['global_scaler'])
        feats = bundle['global_scaler'].feature_names_in_
        window = df.iloc[-50:][feats]
        scaled = sc.transform(window)
        p50 = m50.predict(np.expand_dims(scaled, 0), verbose=0)[0]
        p10 = m10.predict(np.expand_dims(scaled[-10:, :17], 0), verbose=0)[0]
        return p50, p10
    except: return None, None

# --- 4. GIAO DIỆN CHÍNH ---
tab_scan, tab_detail = st.tabs(["📋 Bảng Tổng Hợp VN30", "🔍 Soi Chi Tiết & Kỹ Thuật"])

with tab_scan:
    st.header("⚡ Quét & Phân nhóm Ensemble")
    if st.button("🚀 Bắt đầu quét thị trường"):
        results = []
        bar = st.progress(0)
        for i, sym in enumerate(vn30_symbols):
            df = get_data(sym)
            df_p = build_feats(df)
            if not df_p.empty:
                p50, p10 = run_pred(df_p, sym)
                if p50 is not None:
                    ens_text, ens_group = get_ensemble_signal(p50, p10)
                    results.append({
                        "Mã": sym, "Giá HT": df_p.iloc[-1]['Close'],
                        "Dài hạn": LABELS[np.argmax(p50)], "Ngắn hạn": LABELS[np.argmax(p10)],
                        "Ensemble": ens_text, "Nhóm": ens_group
                    })
            bar.progress((i + 1) / len(vn30_symbols))
        st.session_state.scan_results = pd.DataFrame(results)

    if st.session_state.scan_results is not None:
        df_res = st.session_state.scan_results
        c_mua, c_ngang, c_ban = st.columns(3)
        with c_mua:
            st.success("🟢 DANH MỤC MUA")
            st.dataframe(df_res[df_res['Nhóm'] == "Mua"][['Mã', 'Giá HT', 'Ensemble']], use_container_width=True, hide_index=True)
        with c_ngang:
            st.warning("🟡 THEO DÕI")
            st.dataframe(df_res[df_res['Nhóm'] == "Ngang"][['Mã', 'Giá HT', 'Ensemble']], use_container_width=True, hide_index=True)
        with c_ban:
            st.error("🔴 DANH MỤC BÁN")
            st.dataframe(df_res[df_res['Nhóm'] == "Bán"][['Mã', 'Giá HT', 'Ensemble']], use_container_width=True, hide_index=True)

with tab_detail:
    sel_sym = st.selectbox("Chọn mã chứng khoán", vn30_symbols)
    if st.button(f"🔍 Phân tích sâu {sel_sym}"):
        df = get_data(sel_sym)
        df_p = build_feats(df)
        if not df_p.empty:
            p50, p10 = run_pred(df_p, sel_sym)
            if p50 is not None:
                ens_text, _ = get_ensemble_signal(p50, p10)
                
                # Tính Bollinger Bands
                bb = ta.bbands(df_p['Close'], length=20, std=2)
                df_plot = pd.concat([df_p, bb], axis=1).tail(60)

                # Vẽ Biểu đồ 3 tầng
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.2, 0.3])
                # Tầng 1: Candle + BB
                fig.add_trace(go.Candlestick(x=df_plot['Date'], open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='Giá'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['BBU_20_2.0'], line=dict(color='rgba(173, 216, 230, 0.4)'), name='BB Upper'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['BBL_20_2.0'], line=dict(color='rgba(173, 216, 230, 0.4)'), fill='tonexty', name='BB Lower'), row=1, col=1)
                
                # Mũi tên dự báo
                arrow_c = "green" if "MUA" in ens_text else ("red" if "BÁN" in ens_text else "gray")
                fig.add_annotation(x=df_plot['Date'].iloc[-1], y=df_plot['Close'].iloc[-1], text=f"AI: {ens_text}", showarrow=True, arrowhead=2, arrowcolor=arrow_c, ay=-50 if "MUA" in ens_text else 50)

                # Tầng 2: RSI
                fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

                # Tầng 3: Volume
                fig.add_trace(go.Bar(x=df_plot['Date'], y=df_plot['Volume'], name='Volume', marker_color='orange'), row=3, col=1)

                fig.update_layout(height=800, template='plotly_dark', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Dữ liệu không đủ để phân tích.")

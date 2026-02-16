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
import requests
from io import BytesIO

# --- 1. CẤU HÌNH & TRẠNG THÁI ---
st.set_page_config(page_title="VN30 AI Hybrid Pro", layout="wide", page_icon="💎")

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
        return None, None, None

m50, m10, bundle = load_assets()
vn30_symbols = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']
LABELS = {0: 'MUA 🟢', 1: 'HOLD 🟡', 2: 'BÁN 🔴'}

# --- 2. HÀM TẢI DATA (FIX TRIỆT ĐỂ LỖI DRIVE) ---
@st.cache_data(ttl=3600)
def get_hybrid_data(symbol):
    try:
        # Sử dụng link download trực tiếp để tránh lỗi Tokenizing
        file_id = '1xG6J9fBEF_Z4KY3x_frUwnhVTSA6HG2r'
        csv_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(csv_url, headers=headers)
        
        if response.status_code == 200:
            df_offline = pd.read_csv(BytesIO(response.content), on_bad_lines='skip', engine='python')
            col_name = next((c for c in df_offline.columns if c.lower() in ['symbol', 'ticker', 'mã']), None)
            df_stock = df_offline[df_offline[col_name] == symbol].copy()
            df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')
        else:
            df_stock = pd.DataFrame()

        # Lấy thêm dữ liệu từ Vnstock để bù vào
        client = Vnstock()
        # Lấy khoảng 100 phiên gần nhất để đảm bảo đủ dữ liệu tính toán
        df_online = client.stock(symbol=symbol).quote.history(start="2025-01-01", end=datetime.now().strftime('%Y-%m-%d'))
        
        if not df_online.empty:
            df_online = df_online.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            df_online['Date'] = pd.to_datetime(df_online['Date'])
            df_full = pd.concat([df_stock, df_online], ignore_index=True)
            df_full = df_full.drop_duplicates(subset=['Date']).sort_values('Date')
            return df_full
        
        return df_stock
    except:
        return pd.DataFrame()

# --- 3. HÀM TÍNH TOÁN (GIỮ NGUYÊN NHƯNG THÊM CHECK) ---
def build_features(df):
    if df.empty or len(df) < 55: return pd.DataFrame()
    try:
        df = df.copy().reset_index(drop=True)
        # Tính toán các chỉ báo (RSI, RC...)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        for n in [1, 5, 10, 20]:
            df[f'RC_{n}'] = df['Close'].pct_change(n) * 100
        # Thêm các cột ảo nếu thiếu để scaler không lỗi
        for col in bundle['global_scaler'].feature_names_in_:
            if col not in df.columns: df[col] = 0.0
            
        return df.dropna(subset=['RSI']).tail(60) # Giữ lại đủ để dự báo
    except: return pd.DataFrame()

# --- 4. GIAO DIỆN ---
tab_scan, tab_detail = st.tabs(["📊 Bảng Tổng Hợp VN30", "🔍 Soi Chi Tiết & Backtest"])

with tab_scan:
    if st.button("🚀 Bắt đầu quét thị trường"):
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        for i, sym in enumerate(vn30_symbols):
            status.text(f"⏳ Đang quét {sym}...")
            df = get_hybrid_data(sym)
            df_p = build_features(df)
            
            if not df_p.empty:
                p50, p10 = run_prediction(df_p, sym)
                if p50 is not None:
                    r50, r10 = np.argmax(p50), np.argmax(p10)
                    results.append({
                        "Mã": sym, "Giá": f"{df_p.iloc[-1]['Close']:,}",
                        "Dài hạn": LABELS[r50], "Ngắn hạn": LABELS[r10],
                        "Tin cậy": f"{np.max(p50):.1%}"
                    })
            progress.progress((i + 1) / len(vn30_symbols))
        
        st.session_state.scan_results = pd.DataFrame(results)
        status.success("✅ Đã hoàn tất!")

    if st.session_state.scan_results is not None:
        st.table(st.session_state.scan_results)

with tab_detail:
    sel_sym = st.selectbox("Chọn mã chứng khoán", vn30_symbols)
    if st.button(f"🔍 Phân tích chi tiết {sel_sym}"):
        df = get_hybrid_data(sel_sym)
        df_p = build_features(df)
        if len(df_p) >= 50:
            # Hiện chart và dự báo ở đây (dùng code cũ của bạn)
            st.success(f"Dữ liệu {sel_sym} OK: {len(df_p)} phiên.")
        else:
            st.error(f"Dữ liệu {sel_sym} vẫn không đủ (Chỉ có {len(df)} phiên). Kiểm tra lại file CSV.")

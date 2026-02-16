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

# --- 1. CẤU HÌNH & TRẠNG THÁI ---
st.set_page_config(page_title="VN30 AI Hybrid Pro", layout="wide", page_icon="💎")

# Khởi tạo Session State để lưu kết quả quét
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

# --- 2. HÀM DỰ BÁO (Hàm bị thiếu dẫn đến NameError) ---
def run_prediction(df, symbol, end_idx=None):
    if end_idx is None: end_idx = len(df)
    try:
        # Lấy scaler cho từng mã hoặc dùng scaler chung
        sc = bundle['local_scalers_dict'].get(symbol, bundle['global_scaler'])
        feats_18 = bundle['global_scaler'].feature_names_in_
        
        # Lấy window 50 phiên
        window = df.iloc[end_idx-50 : end_idx][feats_18]
        if len(window) < 50: return None, None
        
        scaled = sc.transform(window)
        p50 = m50.predict(np.expand_dims(scaled, 0), verbose=0)[0]
        # Model ngắn hạn dùng 17 feature đầu
        p10 = m10.predict(np.expand_dims(scaled[-10:, :17], 0), verbose=0)[0]
        return p50, p10
    except:
        return None, None

# --- 3. HÀM TẢI DATA ---
@st.cache_data(ttl=3600)
def get_hybrid_data(symbol):
    try:
        file_id = '1xG6J9fBEF_Z4KY3x_frUwnhVTSA6HG2r'
        csv_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        response = requests.get(csv_url, timeout=10)
        
        df_stock = pd.DataFrame()
        if response.status_code == 200:
            df_offline = pd.read_csv(BytesIO(response.content), on_bad_lines='skip', engine='python')
            col_name = next((c for c in df_offline.columns if c.lower() in ['symbol', 'ticker', 'mã']), None)
            df_stock = df_offline[df_offline[col_name] == symbol].copy()
            df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')

        # Lấy thêm dữ liệu mới từ Vnstock để đảm bảo đủ phiên
        client = Vnstock()
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

def build_features(df):
    if df.empty or len(df) < 55: return pd.DataFrame()
    try:
        df = df.copy().reset_index(drop=True)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        for n in [1, 5, 10, 20]:
            df[f'RC_{n}'] = df['Close'].pct_change(n) * 100
        # Đảm bảo đủ các cột cho mô hình
        for col in bundle['global_scaler'].feature_names_in_:
            if col not in df.columns: df[col] = 0.0
        return df.dropna(subset=['RSI']).tail(65)
    except:
        return pd.DataFrame()

# --- 4. GIAO DIỆN ---
tab_scan, tab_detail = st.tabs(["📊 Bảng Tổng Hợp VN30", "🔍 Soi Chi Tiết & Backtest"])

with tab_scan:
    st.header("⚡ Quét Tín Hiệu Toàn Thị Trường")
    if st.button("🚀 Bắt đầu quét thị trường"):
        summary_list = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, sym in enumerate(vn30_symbols):
            status_text.text(f"⏳ Đang xử lý: {sym} ({i+1}/30)...")
            df_full = get_hybrid_data(sym)
            df_p = build_features(df_full)
            
            if len(df_p) >= 50:
                p50, p10 = run_prediction(df_p, sym)
                if p50 is not None:
                    r50, r10 = np.argmax(p50), np.argmax(p10)
                    summary_list.append({
                        "Mã": sym,
                        "Giá HT": f"{df_p.iloc[-1]['Close']:,}",
                        "Dài hạn (50)": LABELS[r50],
                        "Ngắn hạn (10)": LABELS[r10],
                        "Độ tin cậy": f"{np.max(p50):.1%}"
                    })
            progress_bar.progress((i + 1) / len(vn30_symbols))
        
        st.session_state.scan_results = pd.DataFrame(summary_list)
        status_text.success("✅ Đã quét xong!")

    # Hiển thị kết quả từ bộ nhớ tạm (không mất khi chuyển tab)
    if st.session_state.scan_results is not None:
        st.dataframe(st.session_state.scan_results, use_container_width=True, height=500)
    else:
        st.info("Nhấn 'Bắt đầu quét' để xem tín hiệu VN30.")

with tab_detail:
    sel_sym = st.selectbox("Chọn mã", vn30_symbols)
    if st.button(f"🔍 Phân tích sâu {sel_sym}"):
        df_full = get_hybrid_data(sel_sym)
        df_p = build_features(df_full)
        if len(df_p) >= 50:
            p50, p10 = run_prediction(df_p, sel_sym)
            if p50 is not None:
                st.subheader(f"Kết quả cho {sel_sym}")
                c1, c2 = st.columns(2)
                c1.metric("Xu hướng Dài hạn", LABELS[np.argmax(p50)])
                c2.metric("Xu hướng Ngắn hạn", LABELS[np.argmax(p10)])
        else:
            st.error(f"Dữ liệu {sel_sym} không đủ (Chỉ có {len(df_full)} phiên).")

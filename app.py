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
from io import StringIO

# --- 1. CẤU HÌNH & TRẠNG THÁI ---
st.set_page_config(page_title="VN30 AI Hybrid Pro", layout="wide", page_icon="💎")

# Khởi tạo kho lưu trữ (Session State) để không bị mất dữ liệu khi chuyển Tab
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'is_scanning' not in st.session_state:
    st.session_state.is_scanning = False

@st.cache_resource
def load_assets():
    try:
        m50 = tf.keras.models.load_model('Full_K10_Win50_Hybrid.keras')
        m10 = tf.keras.models.load_model('Baseline_K10_Win10_Hybrid.keras')
        bundle = joblib.load('smart_scaler_system.pkl')
        return m50, m10, bundle
    except Exception as e:
        return None, None, None

m50, m10, bundle = load_assets()
vn30_symbols = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']
LABELS = {0: 'MUA 🟢', 1: 'HOLD 🟡', 2: 'BÁN 🔴'}

# --- 2. HÀM TẢI DATA (FIX LỖI TOKENIZING) ---
@st.cache_data(ttl=3600) # Cache lại để không tải đi tải lại
def get_hybrid_data(symbol):
    try:
        # Link tải trực tiếp
        file_id = '1xG6J9fBEF_Z4KY3x_frUwnhVTSA6HG2r'
        csv_url = f'https://drive.google.com/uc?id={file_id}'
        
        # FIX LỖI ĐỌC CSV: Dùng engine python và xử lý bad lines
        try:
            df_offline = pd.read_csv(csv_url, on_bad_lines='skip', engine='python')
        except:
            # Nếu lỗi, thử tải raw text về rồi đọc
            response = requests.get(csv_url)
            df_offline = pd.read_csv(StringIO(response.text), on_bad_lines='skip')

        # Tìm cột mã chứng khoán
        col_name = next((c for c in df_offline.columns if c.lower() in ['symbol', 'ticker', 'mã', 'ticker_name']), None)
        if not col_name: return pd.DataFrame()
            
        df_stock = df_offline[df_offline[col_name] == symbol].copy()
        
        # Convert ngày tháng chuẩn
        df_stock['Date'] = pd.to_datetime(df_stock['Date'], errors='coerce')
        df_stock = df_stock.dropna(subset=['Date'])
        
        # Lấy data online (Sleep ít hơn vì đã cache)
        # time.sleep(0.5) 
        client = Vnstock()
        df_online = client.stock(symbol=symbol).quote.history(start="2026-01-11", end=datetime.now().strftime('%Y-%m-%d'))
        
        if not df_online.empty:
            df_online = df_online.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            df_online['Date'] = pd.to_datetime(df_online['Date'])
            df_full = pd.concat([df_stock, df_online], ignore_index=True)
            df_full = df_full.drop_duplicates(subset=['Date']).sort_values('Date')
            return df_full
        
        return df_stock
    except Exception:
        return pd.DataFrame()

# --- 3. TÍNH TOÁN ---
def build_features(df):
    if df.empty or len(df) < 60: return pd.DataFrame()
    try:
        df = df.copy()
        for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: df[f'RC_{n}'] = df['Close'].pct_change(n) * 100
        for n in [5, 10, 20]:
            ma = df['Close'].rolling(n).mean()
            df[f'Grad_{n}'] = np.gradient(ma.fillna(method='bfill').fillna(method='ffill'))
        
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Peak'] = df['High'][(df['High'] == df['High'].rolling(11, center=True).max())]
        df['Trough'] = df['Low'][(df['Low'] == df['Low'].rolling(11, center=True).min())]
        
        return df.dropna().reset_index(drop=True)
    except: return pd.DataFrame()

def run_prediction(df, symbol, end_idx=None):
    if end_idx is None: end_idx = len(df)
    try:
        sc = bundle['local_scalers_dict'].get(symbol, bundle['global_scaler'])
        feats_18 = bundle['global_scaler'].feature_names_in_
        window = df.iloc[end_idx-50 : end_idx][feats_18]
        if len(window) < 50: return None, None
        
        scaled = sc.transform(window)
        p50 = m50.predict(np.expand_dims(scaled, 0), verbose=0)[0]
        p10 = m10.predict(np.expand_dims(scaled[-10:, :17], 0), verbose=0)[0]
        return p50, p10
    except: return None, None

# --- 4. GIAO DIỆN CHÍNH (ĐÃ ĐỔI THỨ TỰ TAB) ---
# Tab 1 là Tổng Hợp, Tab 2 là Chi Tiết
tab_scan, tab_detail = st.tabs(["📊 Bảng Tổng Hợp VN30", "🔍 Soi Chi Tiết & Backtest"])

# --- TAB 1: TỔNG HỢP (CÓ LƯU TRẠNG THÁI) ---
with tab_scan:
    st.header("⚡ Quét Tín Hiệu Toàn Thị Trường")
    
    col_a, col_b = st.columns([1, 4])
    start_btn = col_a.button("🚀 Bắt đầu quét", use_container_width=True)
    
    # Logic: Nếu bấm nút -> Quét lại. Nếu không bấm nhưng đã có kết quả cũ -> Hiện lại kết quả cũ.
    if start_btn:
        st.session_state.is_scanning = True
        summary_list = []
        my_bar = st.progress(0)
        status_text = st.empty()
        
        for i, sym in enumerate(vn30_symbols):
            status_text.text(f"⏳ Đang xử lý: {sym} ({i+1}/30)...")
            
            # Data được cache, nên lần đầu sẽ lâu, lần sau bấm lại sẽ rất nhanh
            df_full = get_hybrid_data(sym)
            df_p = build_features(df_full)
            
            if len(df_p) >= 50:
                p50, p10 = run_prediction(df_p, sym)
                if p50 is not None:
                    r50, r10 = np.argmax(p50), np.argmax(p10)
                    summary_list.append({
                        "Mã": sym,
                        "Giá": f"{df_p.iloc[-1]['Close']:,}",
                        "Dài hạn (50)": LABELS[r50],
                        "Ngắn hạn (10)": LABELS[r10],
                        "Độ tin cậy": f"{np.max(p50):.1%}",
                        "Tín hiệu": "💎 MUA NGAY" if (r50==0 and r10==0) else ("⚠️ BÁN" if r50==2 else "Chờ")
                    })
            
            my_bar.progress((i + 1) / len(vn30_symbols))
            time.sleep(0.1) # Sleep nhẹ để UI mượt hơn
            
        st.session_state.scan_results = pd.DataFrame(summary_list)
        st.session_state.is_scanning = False
        status_text.success("✅ Đã quét xong!")
        st.rerun() # Load lại trang để hiển thị kết quả từ session state

    # Hiển thị kết quả từ bộ nhớ (không bị mất khi đổi tab)
    if st.session_state.scan_results is not None and not st.session_state.scan_results.empty:
        st.dataframe(
            st.session_state.scan_results.style.map(lambda x: 'color: green; font-weight: bold' if x == 'MUA 🟢' else ('color: red' if x == 'BÁN 🔴' else ''), subset=['Dài hạn (50)', 'Ngắn hạn (10)']),
            use_container_width=True,
            height=600
        )
    elif st.session_state.scan_results is not None and st.session_state.scan_results.empty:
        st.warning("Không tìm thấy tín hiệu nào (hoặc lỗi dữ liệu).")
    else:
        st.info("Nhấn 'Bắt đầu quét' để phân tích.")

# --- TAB 2: CHI TIẾT (DÙNG LẠI DATA ĐÃ CACHE NÊN NHANH) ---
with tab_detail:
    c1, c2 = st.columns([1, 2])
    sel_sym = c1.selectbox("Chọn mã", vn30_symbols)
    
    # Nút này chỉ để trigger vẽ lại, không cần load lại data nặng
    if c2.button(f"🔎 Phân tích sâu {sel_sym}", use_container_width=True):
        df_full = get_hybrid_data(sel_sym) # Lấy từ cache, siêu nhanh
        df_p = build_features(df_full)
        
        if len(df_p) >= 60:
            # 1. Dự báo
            p50, p10 = run_prediction(df_p, sel_sym)
            if p50 is not None:
                r50, r10 = np.argmax(p50), np.argmax(p10)
                st.markdown(f"### 🎯 Kết quả: {sel_sym}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Xu hướng Dài", LABELS[r50], delta_color="normal" if r50!=0 else "inverse")
                m2.metric("Xu hướng Ngắn", LABELS[r10])
                m3.write(f"Độ tin cậy: **{np.max(p50):.1%}**")
                
                # 2. Lịch sử 10 phiên
                st.subheader("📜 Phong độ AI (10 phiên trước)")
                hist_rows = []
                for i in range(1, 11):
                    idx = len(df_p) - i
                    ph50, ph10 = run_prediction(df_p, sel_sym, idx)
                    if ph50 is not None:
                        rh50 = np.argmax(ph50)
                        hist_rows.append({
                            "Ngày": df_p.iloc[idx-1]['Date'].date(),
                            "Giá": f"{df_p.iloc[idx-1]['Close']:,}",
                            "AI Dự báo": LABELS[rh50],
                            "Thực tế": "Tăng" if df_p.iloc[idx]['Close'] > df_p.iloc[idx-1]['Close'] else "Giảm"
                        })
                st.dataframe(pd.DataFrame(hist_rows), use_container_width=True)

                # 3. Biểu đồ
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                df_v = df_p.tail(80)
                fig.add_trace(go.Candlestick(x=df_v['Date'], open=df_v['Open'], high=df_v['High'], low=df_v['Low'], close=df_v['Close'], name='Giá'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_v['Date'], y=df_v['Peak'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_v['Date'], y=df_v['Trough'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10)), row=1, col=1)
                fig.add_trace(go.Bar(x=df_v['Date'], y=df_v['Volume'], name='Volume'), row=2, col=1)
                fig.update_layout(height=600, template='plotly_dark', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"⚠️ Dữ liệu {sel_sym} bị lỗi hoặc không đủ 60 phiên.")

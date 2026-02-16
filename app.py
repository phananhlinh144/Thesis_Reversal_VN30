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
import re

# --- 1. CẤU HÌNH & LOAD MÔ HÌNH ---
st.set_page_config(page_title="VN30 AI Hybrid Pro", layout="wide", page_icon="📈")

@st.cache_resource
def load_assets():
    try:
        m50 = tf.keras.models.load_model('Full_K10_Win50_Hybrid.keras')
        m10 = tf.keras.models.load_model('Baseline_K10_Win10_Hybrid.keras')
        bundle = joblib.load('smart_scaler_system.pkl')
        return m50, m10, bundle
    except Exception as e:
        st.error(f"Lỗi Load Assets: {e}")
        return None, None, None

m50, m10, bundle = load_assets()
vn30_symbols = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']
LABELS = {0: 'MUA 🟢', 1: 'HOLD 🟡', 2: 'BÁN 🔴'}

# --- 2. HÀM LẤY DỮ LIỆU HYBRID (CSV + API) ---
@st.cache_data(ttl=3600)
def get_hybrid_data(symbol):
    try:
        # Link gốc bạn đưa
        share_link = 'https://drive.google.com/file/d/1xG6J9fBEF_Z4KY3x_frUwnhVTSA6HG2r/view?usp=drive_link'
        
        # Trích xuất ID từ link để tạo link tải trực tiếp
        file_id = share_link.split('/d/')[1].split('/')[0]
        csv_url = f'https://drive.google.com/uc?id={file_id}'
        
        # 1. Đọc data offline
        df_offline = pd.read_csv(csv_url)
        
        # Tự động tìm cột chứa mã chứng khoán
        col_name = next((c for c in df_offline.columns if c.lower() in ['symbol', 'ticker', 'mã', 'ticker_name']), None)
        if col_name is None:
            st.error("Không tìm thấy cột chứa mã chứng khoán trong file CSV!")
            return pd.DataFrame()
            
        df_stock_offline = df_offline[df_offline[col_name] == symbol].copy()
        df_stock_offline['Date'] = pd.to_datetime(df_stock_offline['Date'])
        
        # 2. Lấy data online bù vào (từ 11/01/2026 đến nay)
        client = Vnstock()
        time.sleep(1.7) # Sleep bảo vệ API
        
        # Lấy đến hiện tại
        df_online = client.stock(symbol=symbol).quote.history(start="2026-01-11", end=datetime.now().strftime('%Y-%m-%d'))
        
        if not df_online.empty:
            df_online = df_online.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            df_online['Date'] = pd.to_datetime(df_online['Date'])
            # Gộp lại và xóa trùng
            df_full = pd.concat([df_stock_offline, df_online], ignore_index=True)
            df_full = df_full.drop_duplicates(subset=['Date']).sort_values('Date')
            return df_full
        
        return df_stock_offline
    except Exception as e:
        st.error(f"Lỗi tải data {symbol}: {e}")
        return pd.DataFrame()

# --- 3. HÀM TÍNH TOÁN & DỰ BÁO ---
def build_features(df):
    if df.empty or len(df) < 60: return pd.DataFrame()
    try:
        df = df.copy()
        # Tính toán các chỉ báo
        for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
            df[f'RC_{n}'] = df['Close'].pct_change(n) * 100
        for n in [5, 10, 20]:
            ma = df['Close'].rolling(n).mean()
            df[f'Grad_{n}'] = np.gradient(ma.fillna(method='bfill').fillna(method='ffill'))
        
        df['RSI'] = ta.rsi(df['Close'], length=14)
        bb = ta.bbands(df['Close'], length=20, std=2)
        df['BB_PctB'] = bb.iloc[:, 4] if bb is not None else 0.5
        df['MACD_Hist'] = ta.macd(df['Close']).iloc[:, 1]
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['ATR_Rel'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) / df['Close']
        
        ma20 = df['Close'].rolling(20).mean()
        rmin, rmax = df['Close'].rolling(20).min(), df['Close'].rolling(20).max()
        df['Dist_Prev_K10'] = 0.0
        df.loc[df['Close'] >= ma20, 'Dist_Prev_K10'] = (df['Close'] - rmin) / rmin
        df.loc[df['Close'] < ma20, 'Dist_Prev_K10'] = (df['Close'] - rmax) / rmax
        
        # Điểm đảo chiều thực tế
        df['Peak'] = df['High'][(df['High'] == df['High'].rolling(11, center=True).max())]
        df['Trough'] = df['Low'][(df['Low'] == df['Low'].rolling(11, center=True).min())]
        
        return df.dropna().reset_index(drop=True)
    except: return pd.DataFrame()

def run_prediction(df, symbol, end_idx=None):
    if end_idx is None: end_idx = len(df)
    try:
        sc = bundle['local_scalers_dict'].get(symbol, bundle['global_scaler'])
        feats_18 = bundle['global_scaler'].feature_names_in_
        # Lấy window 50 phiên trước end_idx
        window = df.iloc[end_idx-50 : end_idx][feats_18]
        if len(window) < 50: return None, None # Kiểm tra đủ độ dài
        
        scaled = sc.transform(window)
        p50 = m50.predict(np.expand_dims(scaled, 0), verbose=0)[0]
        p10 = m10.predict(np.expand_dims(scaled[-10:, :17], 0), verbose=0)[0]
        return p50, p10
    except: return None, None

# --- 4. GIAO DIỆN ---
tab1, tab2 = st.tabs(["🔍 Soi Chi Tiết Mã", "📊 Tổng Hợp VN30"])

with tab1:
    cc1, cc2 = st.columns([1, 2])
    sel_stock = cc1.selectbox("Chọn mã chứng khoán", vn30_symbols)
    run_btn = cc2.button(f"🚀 Phân tích & Xem lịch sử {sel_stock}", use_container_width=True)

    if run_btn:
        with st.spinner("Đang tải dữ liệu Hybrid..."):
            df_full = get_hybrid_data(sel_stock)
            df_p = build_features(df_full)
            
            if len(df_p) >= 60: # Cần dư ra 10 phiên để backtest lịch sử
                # 1. Dự báo hiện tại (Phiên mới nhất)
                t_idx = len(df_p)
                p50, p10 = run_prediction(df_p, sel_stock, t_idx)
                
                if p50 is not None:
                    st.markdown(f"### Kết quả mới nhất: {df_p.iloc[-1]['Date'].date()}")
                    res_c1, res_c2, res_c3 = st.columns(3)
                    r50, r10 = np.argmax(p50), np.argmax(p10)
                    res_c1.metric("Model Win50 (Dài)", LABELS[r50], f"{np.max(p50):.1%}")
                    res_c2.metric("Model Win10 (Ngắn)", LABELS[r10], f"{np.max(p10):.1%}")
                    res_c3.info(f"Giá đóng cửa: {df_p.iloc[-1]['Close']:,}")

                    # 2. Biểu đồ
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.2, 0.3], vertical_spacing=0.03)
                    df_v = df_p.tail(100)
                    
                    fig.add_trace(go.Candlestick(x=df_v['Date'], open=df_v['Open'], high=df_v['High'], low=df_v['Low'], close=df_v['Close'], name='Giá'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_v['Date'], y=df_v['Peak'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='Đỉnh thực'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_v['Date'], y=df_v['Trough'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name='Đáy thực'), row=1, col=1)
                    
                    v_colors = ['red' if r['Open'] > r['Close'] else 'green' for _, r in df_v.iterrows()]
                    fig.add_trace(go.Bar(x=df_v['Date'], y=df_v['Volume'], marker_color=v_colors, name='Volume'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df_v['Date'], y=df_v['RSI'], line=dict(color='orange'), name='RSI'), row=3, col=1)
                    
                    fig.update_layout(height=600, template='plotly_dark', xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # 3. Lịch sử dự báo 10 phiên gần nhất
                    st.divider()
                    st.subheader("📜 Lịch sử dự báo (10 phiên gần nhất)")
                    
                    hist_data = []
                    # Lặp lùi 10 phiên (từ t-1 đến t-10)
                    for i in range(1, 11):
                        idx_hist = len(df_p) - i
                        if idx_hist < 50: break # Không đủ data thì dừng
                        
                        p50_h, p10_h = run_prediction(df_p, sel_stock, idx_hist)
                        if p50_h is not None:
                            r50_h, r10_h = np.argmax(p50_h), np.argmax(p10_h)
                            hist_data.append({
                                "Ngày": df_p.iloc[idx_hist-1]['Date'].date(),
                                "Giá Đóng": f"{df_p.iloc[idx_hist-1]['Close']:,}",
                                "Model Dài": LABELS[r50_h],
                                "Model Ngắn": LABELS[r10_h],
                                "Độ tin cậy": f"{np.max(p50_h):.1%}",
                                "Kết quả": "✅" if r50_h == r10_h else "❌" # Đồng thuận hay không
                            })
                    
                    if hist_data:
                        st.table(pd.DataFrame(hist_data))
                    else:
                        st.warning("Không đủ dữ liệu lịch sử để hiển thị.")

            else:
                st.error("Dữ liệu không đủ để phân tích.")

with tab2:
    st.header("📊 Quét Toàn Bộ VN30")
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    start_btn = col_ctrl1.button("▶️ Bắt đầu quét", use_container_width=True)
    
    if "stop" not in st.session_state: st.session_state.stop = False
    
    if col_ctrl2.button("⏹️ Dừng quét", use_container_width=True):
        st.session_state.stop = True
        st.rerun()

    if start_btn:
        st.session_state.stop = False
        summary_list = []
        prog = st.progress(0)
        status_info = st.empty()
        
        for i, sym in enumerate(vn30_symbols):
            if st.session_state.stop:
                st.warning("Đã dừng quét theo yêu cầu.")
                break
                
            status_info.info(f"🔍 Đang quét: **{sym}** (Chờ 1.7s để API không bị khóa...)")
            df_f = get_hybrid_data(sym)
            df_ready = build_features(df_f)
            
            if not df_ready.empty and len(df_ready) >= 50:
                p50, p10 = run_prediction(df_ready, sym)
                if p50 is not None:
                    r50, r10 = np.argmax(p50), np.argmax(p10)
                    summary_list.append({
                        "Mã": sym,
                        "Giá Hiện Tại": f"{df_ready.iloc[-1]['Close']:,}",
                        "Win50 (Dài)": LABELS[r50],
                        "Win10 (Ngắn)": LABELS[r10],
                        "Độ tin cậy": f"{np.max(p50):.1%}",
                        "Đồng thuận": "✅" if r50 == r10 else "❌"
                    })
            
            prog.progress((i + 1) / len(vn30_symbols))
        
        status_info.success("✅ Đã hoàn tất quét 30 mã VN30!")
        if summary_list:
            st.table(pd.DataFrame(summary_list))

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

# --- 1. CẤU HÌNH & LOAD ASSETS ---
st.set_page_config(page_title="VN30 AI Hybrid Pro", layout="wide", page_icon="🤖")

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
    """Kết hợp dữ liệu từ CSV Drive (tới 10/01/2026) và API (từ 11/01/2026)"""
    try:
        # Link direct download từ Drive bạn gửi
        file_id = '1xG6J9fBEF_Z4KY3x_frUwnhVTSA6HG2r'
        csv_url = f'https://drive.google.com/uc?id={file_id}'
        
        # 1. Đọc data offline
        df_offline = pd.read_csv(csv_url)
        df_offline['Date'] = pd.to_datetime(df_offline['Date'])
        df_stock_offline = df_offline[df_offline['Ticker'] == symbol].copy()
        
        # 2. Lấy data online (từ ngày 2026-01-11 đến nay)
        client = Vnstock()
        start_date = "2026-01-11"
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Chờ 1.7s để né rate limit API
        time.sleep(1.7)
        
        df_online = client.stock(symbol=symbol).quote.history(start=start_date, end=end_date)
        
        if not df_online.empty:
            df_online = df_online.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            df_online['Date'] = pd.to_datetime(df_online['Date'])
            # Gộp lại
            df_full = pd.concat([df_stock_offline, df_online], ignore_index=True)
            # Xóa trùng nếu có
            df_full = df_full.drop_duplicates(subset=['Date']).sort_values('Date')
            return df_full
        
        return df_stock_offline
    except Exception as e:
        st.error(f"Lỗi tải data {symbol}: {e}")
        return pd.DataFrame()

# --- 3. FEATURE ENGINEERING & AI LOGIC ---
def build_features(df):
    if df.empty or len(df) < 60: return pd.DataFrame() # Cần tối thiểu để tính RC_55
    try:
        df = df.copy()
        # Chỉ báo
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
        
        # Đảo chiều
        df['Peak'] = df['High'][(df['High'] == df['High'].rolling(11, center=True).max())]
        df['Trough'] = df['Low'][(df['Low'] == df['Low'].rolling(11, center=True).min())]
        
        return df.dropna().reset_index(drop=True)
    except: return pd.DataFrame()

def run_prediction(df, symbol, end_idx=None):
    if end_idx is None: end_idx = len(df)
    if end_idx < 50: return None, None
    try:
        sc = bundle['local_scalers_dict'].get(symbol, bundle['global_scaler'])
        feats_18 = bundle['global_scaler'].feature_names_in_
        window = df.iloc[end_idx-50 : end_idx][feats_18]
        scaled = sc.transform(window)
        p50 = m50.predict(np.expand_dims(scaled, 0), verbose=0)[0]
        p10 = m10.predict(np.expand_dims(scaled[-10:, :17], 0), verbose=0)[0]
        return p50, p10
    except: return None, None

# --- 4. GIAO DIỆN CHÍNH ---
tab1, tab2 = st.tabs(["🔍 Soi Chi Tiết", "📊 Tổng Hợp VN30"])

with tab1:
    st.sidebar.header("Cấu hình")
    sel_stock = st.sidebar.selectbox("Chọn mã", vn30_symbols)
    hist_step = st.sidebar.slider("Lùi phiên", 0, 50, 0)
    
    if st.button(f"Phân tích chuyên sâu {sel_stock}"):
        with st.spinner(f"Đang xử lý {sel_stock}..."):
            df_full = get_hybrid_data(sel_stock)
            df_p = build_features(df_full)
            
            if len(df_p) >= 50:
                t_idx = len(df_p) - hist_step
                p50, p10 = run_prediction(df_p, sel_stock, t_idx)
                
                # Signal
                st.write(f"### Tín hiệu ngày: {df_p.iloc[t_idx-1]['Date'].date()}")
                c1, c2, c3 = st.columns(3)
                r50, r10 = np.argmax(p50), np.argmax(p10)
                c1.metric("Model Win50", LABELS[r50], f"{np.max(p50):.1%}")
                c2.metric("Model Win10", LABELS[r10], f"{np.max(p10):.1%}")
                c3.success(f"KẾT LUẬN: {LABELS[r50] if r50==r10 else 'THEO DÕI'}")

                # Biểu đồ
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.2, 0.3], vertical_spacing=0.03)
                df_v = df_p.tail(100 + hist_step)
                fig.add_trace(go.Candlestick(x=df_v['Date'], open=df_v['Open'], high=df_v['High'], low=df_v['Low'], close=df_v['Close'], name='Giá'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_v['Date'], y=df_v['Peak'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Đỉnh'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_v['Date'], y=df_v['Trough'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='lime'), name='Đáy'), row=1, col=1)
                
                v_cols = ['red' if r['Open'] > r['Close'] else 'green' for _, r in df_v.iterrows()]
                fig.add_trace(go.Bar(x=df_v['Date'], y=df_v['Volume'], marker_color=v_cols, name='Volume'), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_v['Date'], y=df_v['RSI'], line=dict(color='orange'), name='RSI'), row=3, col=1)
                
                fig.add_vline(x=df_p.iloc[t_idx-1]['Date'], line_dash="dot", line_color="white")
                fig.update_layout(height=800, template='plotly_dark', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Dữ liệu sau khi gộp vẫn không đủ 50 phiên sạch.")

with tab2:
    st.header("📊 Quét toàn bộ VN30 (Dữ liệu Hybrid)")
    if st.button("🚀 Bắt đầu quét thị trường"):
        summary = []
        prog = st.progress(0)
        status = st.empty()
        
        for i, sym in enumerate(vn30_symbols):
            status.text(f"🔍 Đang quét: {sym} (Chờ 1.7s...)")
            df_full = get_hybrid_data(sym)
            df_p = build_features(df_full)
            
            if len(df_p) >= 50:
                p50, p10 = run_prediction(df_p, sym)
                if p50 is not None:
                    r50, r10 = np.argmax(p50), np.argmax(p10)
                    summary.append({
                        "Mã": sym,
                        "Giá": f"{df_p.iloc[-1]['Close']:,}",
                        "Win50": LABELS[r50],
                        "Win10": LABELS[r10],
                        "Độ tin cậy": f"{np.max(p50):.1%}",
                        "Đồng thuận": "✅" if r50 == r10 else "❌"
                    })
            prog.progress((i + 1) / len(vn30_symbols))
            
        status.text("✅ Hoàn tất!")
        if summary:
            st.dataframe(pd.DataFrame(summary), use_container_width=True)
        else:
            st.warning("Không quét được mã nào. Hãy kiểm tra lại file CSV trên Drive.")

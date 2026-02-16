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

# --- 1. CẤU HÌNH & LOAD MÔ HÌNH ---
st.set_page_config(page_title="VN30 AI Pro System", layout="wide", page_icon="🚀")

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

# --- 2. HÀM TÍNH TOÁN KỸ THUẬT (FEATURE ENGINEERING) ---
def build_features(df):
    try:
        df = df.copy()
        # Rate of Change
        for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
            df[f'RC_{n}'] = df['Close'].pct_change(n) * 100
        # Gradients
        for n in [5, 10, 20]:
            ma = df['Close'].rolling(n).mean()
            df[f'Grad_{n}'] = np.gradient(ma.fillna(method='bfill').fillna(method='ffill'))
        # Indicators
        df['RSI'] = ta.rsi(df['Close'], length=14)
        bb = ta.bbands(df['Close'], length=20, std=2)
        df['BB_PctB'] = bb.iloc[:, 4] if bb is not None else 0.5
        df['MACD_Hist'] = ta.macd(df['Close']).iloc[:, 1]
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['ATR_Rel'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) / df['Close']
        # Dist_Prev_K10
        ma20 = df['Close'].rolling(20).mean()
        rmin, rmax = df['Close'].rolling(20).min(), df['Close'].rolling(20).max()
        df['Dist_Prev_K10'] = 0.0
        df.loc[df['Close'] >= ma20, 'Dist_Prev_K10'] = (df['Close'] - rmin) / rmin
        df.loc[df['Close'] < ma20, 'Dist_Prev_K10'] = (df['Close'] - rmax) / rmax
        # Tìm điểm đảo chiều thực tế (Để hậu kiểm)
        df['Actual_Peak'] = df['High'][(df['High'] == df['High'].rolling(11, center=True).max())]
        df['Actual_Trough'] = df['Low'][(df['Low'] == df['Low'].rolling(11, center=True).min())]
        return df.dropna().reset_index(drop=True)
    except:
        return pd.DataFrame()

# --- 3. HÀM DỰ BÁO CHUNG ---
def get_prediction(df, symbol, end_idx=None):
    if end_idx is None: end_idx = len(df)
    if end_idx < 50: return None, None
    try:
        feats_18 = bundle['global_scaler'].feature_names_in_
        sc = bundle['local_scalers_dict'].get(symbol, bundle['global_scaler'])
        # Lấy window 50 phiên tại thời điểm end_idx
        data_window = df.iloc[end_idx-50 : end_idx][feats_18]
        scaled_data = sc.transform(data_window)
        # Inference
        p50 = m50.predict(np.expand_dims(scaled_data, axis=0), verbose=0)[0]
        p10 = m10.predict(np.expand_dims(scaled_data[-10:, :17], axis=0), verbose=0)[0]
        return p50, p10
    except: return None, None

# --- 4. GIAO DIỆN ---
tab1, tab2 = st.tabs(["🔍 Soi Chi Tiết & Backtest", "📋 Bảng Tổng Hợp VN30"])

with tab1:
    st.sidebar.title("🎮 Điều khiển")
    selected_stock = st.sidebar.selectbox("Mã chứng khoán", vn30_symbols)
    hist_step = st.sidebar.slider("Lùi về quá khứ (phiên)", 0, 50, 0, help="Dùng để xem AI dự báo đúng hay sai trong quá khứ")
    lookback_view = st.sidebar.slider("Số phiên hiển thị", 50, 250, 100)

    if st.button(f"Phân tích {selected_stock}"):
        client = Vnstock()
        df_raw = client.stock(symbol=selected_stock).quote.history(start='2024-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        df_raw = df_raw.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        df_p = build_features(df_raw)

        if len(df_p) >= 50:
            target_idx = len(df_p) - hist_step
            p50, p10 = get_prediction(df_p, selected_stock, end_idx=target_idx)
            
            # Hiển thị Signal
            st.subheader(f"Tín hiệu tại ngày: {df_p.iloc[target_idx-1]['Date']}")
            c1, c2, c3 = st.columns(3)
            r50, r10 = np.argmax(p50), np.argmax(p10)
            c1.metric("Model Win50", LABELS[r50], f"{np.max(p50):.1%}")
            c2.metric("Model Win10", LABELS[r10], f"{np.max(p10):.1%}")
            advice = LABELS[r50] if r50 == r10 else "THEO DÕI"
            c3.info(f"KẾT LUẬN: {advice}")

            # Biểu đồ 3 tầng
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.2, 0.3])
            df_v = df_p.tail(lookback_view + hist_step)
            # Tầng 1: Giá & Đảo chiều
            fig.add_trace(go.Candlestick(x=df_v['Date'], open=df_v['Open'], high=df_v['High'], low=df_v['Low'], close=df_v['Close'], name='Giá'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_v['Date'], y=df_v['Actual_Peak'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Đỉnh thực'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_v['Date'], y=df_v['Actual_Trough'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='lime'), name='Đáy thực'), row=1, col=1)
            # Tầng 2: Volume
            v_colors = ['red' if r['Open'] > r['Close'] else 'green' for _, r in df_v.iterrows()]
            fig.add_trace(go.Bar(x=df_v['Date'], y=df_v['Volume'], marker_color=v_colors, name='Volume'), row=2, col=1)
            # Tầng 3: RSI
            fig.add_trace(go.Scatter(x=df_v['Date'], y=df_v['RSI'], line=dict(color='orange'), name='RSI'), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            fig.add_vline(x=df_p.iloc[target_idx-1]['Date'], line_dash="dot", line_color="white")
            fig.update_layout(height=800, template='plotly_dark', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("📊 Quét tín hiệu toàn bộ VN30")
    if st.button("🚀 Bắt đầu quét"):
        summary_list = []
        prog = st.progress(0)
        client = Vnstock()
        for i, sym in enumerate(vn30_symbols):
            try:
                # Lấy dữ liệu nhanh 100 phiên để tính toán
                df_s = client.stock(symbol=sym).quote.history(start=(datetime.now()-timedelta(days=150)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
                df_s = df_s.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
                df_s_p = build_features(df_s)
                p50, p10 = get_prediction(df_s_p, sym)
                if p50 is not None:
                    r50, r10 = np.argmax(p50), np.argmax(p10)
                    summary_list.append({
                        "Mã": sym,
                        "Giá": f"{df_s_p.iloc[-1]['Close']:,}",
                        "Win50 (Dài)": LABELS[r50],
                        "Win10 (Ngắn)": LABELS[r10],
                        "Tin cậy": f"{np.max(p50):.1%}",
                        "Đồng thuận": "✅" if r50 == r10 else "❌"
                    })
            except: pass
            prog.progress((i + 1) / len(vn30_symbols))
        st.dataframe(pd.DataFrame(summary_list), use_container_width=True)

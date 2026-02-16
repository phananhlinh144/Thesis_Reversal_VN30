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

# --- 1. CẤU HÌNH HỆ THỐNG ---
st.set_page_config(page_title="VN30 AI Hybrid System", layout="wide", page_icon="🤖")

@st.cache_resource
def load_assets():
    """Load các file model và scaler bạn đã upload"""
    try:
        m50 = tf.keras.models.load_model('Full_K10_Win50_Hybrid.keras')
        m10 = tf.keras.models.load_model('Baseline_K10_Win10_Hybrid.keras')
        bundle = joblib.load('smart_scaler_system.pkl')
        return m50, m10, bundle
    except Exception as e:
        st.error(f"Lỗi Load Assets: {e}")
        return None, None, None

m50, m10, bundle = load_assets()
# Thứ tự feature bắt buộc phải đúng theo lúc train (từ file pkl)
FEATS_18 = bundle['global_scaler'].feature_names_in_
FEATS_17 = FEATS_18[:17]

# --- 2. HÀM TÍNH TOÁN KỸ THUẬT (FEATURE ENGINEERING) ---
def build_features(df):
    try:
        df = df.copy()
        # Tính Rate of Change (RC)
        for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
            df[f'RC_{n}'] = df['Close'].pct_change(n) * 100
        
        # Tính Gradients (Độ dốc MA)
        for n in [5, 10, 20]:
            ma = df['Close'].rolling(n).mean()
            # Xử lý bfill để không bị mất dòng ở bước np.gradient
            df[f'Grad_{n}'] = np.gradient(ma.fillna(method='bfill').fillna(method='ffill'))
            
        # Chỉ báo kỹ thuật phổ biến
        df['RSI'] = ta.rsi(df['Close'], length=14)
        bb = ta.bbands(df['Close'], length=20, std=2)
        df['BB_PctB'] = bb.iloc[:, 4] if bb is not None else 0.5
        df['MACD_Hist'] = ta.macd(df['Close']).iloc[:, 1]
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['ATR_Rel'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) / df['Close']
        
        # Chỉ báo Dist_Prev_K10 (Khoảng cách nến so với đỉnh/đáy 20 phiên)
        ma20 = df['Close'].rolling(20).mean()
        rmin = df['Close'].rolling(20).min()
        rmax = df['Close'].rolling(20).max()
        df['Dist_Prev_K10'] = 0.0
        df.loc[df['Close'] >= ma20, 'Dist_Prev_K10'] = (df['Close'] - rmin) / rmin
        df.loc[df['Close'] < ma20, 'Dist_Prev_K10'] = (df['Close'] - rmax) / rmax
        
        # XỬ LÝ NAN: Cắt bỏ các dòng trống do tính rolling (ít nhất 60 dòng đầu)
        df_clean = df.dropna().reset_index(drop=True)
        return df_clean
    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu: {e}")
        return pd.DataFrame()

# --- 3. HÀM DỰ BÁO (SCALING & INFERENCE) ---
def run_ai_logic(df, symbol):
    if len(df) < 50:
        return None
    
    try:
        # --- BƯỚC QUAN TRỌNG: LOCAL SCALING ---
        # Kiểm tra xem mã này có Scaler riêng không
        if symbol in bundle['local_scalers_dict']:
            sc = bundle['local_scalers_dict'][symbol]
            st.info(f"✅ Đang sử dụng Local Scaler riêng cho mã: {symbol}")
        else:
            sc = bundle['global_scaler']
            st.warning(f"⚠️ Không tìm thấy Local Scaler cho {symbol}, đang dùng Global Scaler.")
        
        # Lấy 50 phiên cuối (Chống Data Leaking)
        data_window = df.iloc[-50:][FEATS_18]
        
        # Thực hiện biến đổi (transform) dựa trên "thước đo" của chính mã đó
        scaled_data = sc.transform(data_window)
        
        # --- DỰ BÁO ---
        # Model 50 (Yêu cầu 18 features)
        input_50 = np.expand_dims(scaled_data, axis=0) # Shape: (1, 50, 18)
        p50 = m50.predict(input_50, verbose=0)[0]
        
        # Model 10 (Yêu cầu 17 features)
        # Cắt 10 dòng cuối và 17 cột đầu tiên (bỏ Dist_Prev_K10 nếu FEATS_18 có nó ở cuối)
        scaled_10 = scaled_data[-10:, :17]
        input_10 = np.expand_dims(scaled_10, axis=0) # Shape: (1, 10, 17)
        p10 = m10.predict(input_10, verbose=0)[0]
        
        return p50, p10
    except Exception as e:
        st.error(f"Lỗi Scaling/Inference: {e}")
        return None, None
        
# --- 4. GIAO DIỆN VÀ LUỒNG CHẠY ---
vn30_symbols = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']

st.sidebar.title("🎮 Control Panel")
selected_stock = st.sidebar.selectbox("Chọn mã VN30", vn30_symbols)
lookback_view = st.sidebar.slider("Số phiên hiển thị biểu đồ", 50, 250, 100)

if st.button(f"Phân tích chuyên sâu {selected_stock}"):
    with st.status(f"Đang phân tích {selected_stock}...") as status:
        # Bước 1: Lấy dữ liệu an toàn (kiểu Jupyter)
        st.write("📡 Đang tải dữ liệu từ VNStock...")
        client = Vnstock()
        # Lấy dôi ra 365 ngày để đảm bảo đủ window tính RC_55 và Grad_20
        df_raw = client.stock(symbol=selected_stock).quote.history(
            start=(datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d'),
            end=datetime.now().strftime('%Y-%m-%d')
        )
        
        if not df_raw.empty:
            # Chuẩn hóa tên cột
            df_raw = df_raw.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            st.write("📊 Dữ liệu thô (3 phiên gần nhất):")
            st.table(df_raw.tail(3))
            
            # Bước 2: Feature Engineering & Clean NaN
            df_p = build_features(df_raw)
            st.write(f"✅ Đã xử lý NaN. Dữ liệu sạch: {len(df_p)} phiên.")
            
            if len(df_p) >= 50:
                # Bước 3: AI Inference
                st.write("🧠 Đang chạy mô hình Hybrid...")
                p50, p10 = run_ai_logic(df_p, selected_stock)
                
                if p50 is not None:
                    # Hiển thị kết quả
                    c1, c2, c3 = st.columns(3)
                    res50, res10 = np.argmax(p50), np.argmax(p10)
                    labels = {0: 'MUA 🟢', 1: 'HOLD 🟡', 2: 'BÁN 🔴'}
                    
                    c1.metric("Model Dài (Win50)", labels[res50], f"{np.max(p50):.1%}")
                    c2.metric("Model Ngắn (Win10)", labels[res10], f"{np.max(p10):.1%}")
                    
                    # Logic đồng thuận (Hybrid)
                    final_advice = "THEO DÕI"
                    if res50 == res10: final_advice = labels[res50]
                    c3.subheader(f"Kết luận: {final_advice}")

                    # Bước 4: Vẽ biểu đồ chuyên nghiệp
                    
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                    df_v = df_p.tail(lookback_view)
                    
                    fig.add_trace(go.Candlestick(x=df_v['Date'], open=df_v['Open'], high=df_v['High'], 
                                  low=df_v['Low'], close=df_v['Close'], name='Giá'), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(x=df_v['Date'], y=df_v['RSI'], name='RSI', line=dict(color='orange')), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    fig.update_layout(height=700, template='plotly_dark', xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                status.update(label="Hoàn tất!", state="complete")
            else:
                st.error("Dữ liệu không đủ để AI dự báo (Cần 50 phiên sạch).")
        else:
            st.error("Lỗi kết nối API.")

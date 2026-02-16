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

# --- CẤU HÌNH ---
st.set_page_config(page_title="VN30 AI Pro Dashboard", layout="wide", page_icon="📈")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_models():
    try:
        # Đảm bảo đường dẫn file đúng với nơi bạn lưu
        m50 = tf.keras.models.load_model('Full_K10_Win50_Hybrid.keras')
        m10 = tf.keras.models.load_model('Baseline_K10_Win10_Hybrid.keras')
        scaler = joblib.load('smart_scaler_system.pkl')
        return m50, m10, scaler
    except Exception as e:
        return None, None, None

model_win50, model_win10, scaler_bundle = load_models()
if scaler_bundle:
    global_scaler = scaler_bundle['global_scaler']
    local_scalers = scaler_bundle['local_scalers_dict']

FINAL_FEATURES = [
    'RC_1', 'RC_2', 'RC_3', 'RC_5', 'RC_8', 'RC_13', 'RC_21', 'RC_34', 'RC_55',
    'Grad_5', 'Grad_10', 'Grad_20', 'RSI', 'BB_PctB', 'MACD_Hist', 'Vol_Ratio', 'ATR_Rel'
]
FEATS_FULL = FINAL_FEATURES + ['Dist_Prev_K10']

# --- 2. XỬ LÝ DỮ LIỆU ---
def get_data(symbol):
    try:
        stock = Vnstock().stock(symbol=symbol, source='VCI')
        df = stock.quote.history(start=(datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d'), 
                                 end=datetime.now().strftime('%Y-%m-%d'))
        if df.empty: return pd.DataFrame()
        
        df = df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 
                                'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']: df[c] = pd.to_numeric(df[c])
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date').reset_index(drop=True)
    except: return pd.DataFrame()

def add_indicators(df):
    g = df.copy()
    # Feature cho AI
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
    for n in [5, 10, 20]: 
        ma = g['Close'].rolling(n).mean()
        g[f'Grad_{n}'] = np.gradient(ma.fillna(method='bfill'))
    
    # Chỉ báo để vẽ biểu đồ
    g['SMA_20'] = ta.sma(g['Close'], length=20)
    bb = ta.bbands(g['Close'], length=20, std=2)
    g['BB_Upper'] = bb.iloc[:, 0]
    g['BB_Lower'] = bb.iloc[:, 2]
    g['BB_PctB'] = bb.iloc[:, 4]
    
    g['RSI'] = ta.rsi(g['Close'], length=14)
    g['MACD_Hist'] = ta.macd(g['Close']).iloc[:, 1]
    g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
    g['ATR_Rel'] = ta.atr(g['High'], g['Low'], g['Close'], length=14) / g['Close']
    
    # K10 Logic (Mô phỏng Dist_Prev)
    rmin = g['Close'].rolling(20).min()
    rmax = g['Close'].rolling(20).max()
    ma20 = g['Close'].rolling(20).mean()
    g['Dist_Prev_K10'] = 0.0
    mask_up = g['Close'] >= ma20
    g.loc[mask_up, 'Dist_Prev_K10'] = (g['Close'] - rmin) / rmin
    g.loc[~mask_up, 'Dist_Prev_K10'] = (g['Close'] - rmax) / rmax
    
    return g.dropna().reset_index(drop=True)

# --- 3. DỰ BÁO ---
def predict_single(df_calc, symbol, idx):
    # Cắt dữ liệu tại thời điểm idx
    end = idx + 1
    d50 = df_calc.iloc[end-50:end]
    d10 = df_calc.iloc[end-10:end]
    if len(d50) < 50: return None
    
    scaler = local_scalers.get(symbol, global_scaler)
    try:
        s50 = scaler.transform(d50[FEATS_FULL].values)
        s10 = scaler.transform(d10[FEATS_FULL].values)
    except: return None # Fallback nếu lỗi

    p50 = model_win50.predict(np.expand_dims(s50, axis=0), verbose=0)[0]
    p10 = model_win10.predict(np.expand_dims(s10[:,:17], axis=0), verbose=0)[0]
    
    cls50, cls10 = np.argmax(p50), np.argmax(p10)
    prob = (p50[cls50] + p10[cls10])/2
    
    sig = 1 # Hold
    if cls50 == 0 and cls10 == 0: sig = 0 # Buy
    elif cls50 == 2 and cls10 == 2: sig = 2 # Sell
    
    return sig, prob

# --- 4. VẼ BIỂU ĐỒ NÂNG CAO ---
def plot_advanced_chart(df, ai_signals, k10_points):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # --- PANEL 1: GIÁ + BB + MA + AI ---
    # 1. Nến
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Giá'), row=1, col=1)
    
    # 2. Bollinger Bands
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], line=dict(color='gray', width=1), 
                             name='BB Upper', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], line=dict(color='gray', width=1), 
                             fill='tonexty', fillcolor='rgba(200,200,200,0.1)', name='BB Lower'), row=1, col=1)
    
    # 3. SMA 20
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='orange', width=1.5), 
                             name='SMA 20'), row=1, col=1)

    # 4. ĐIỂM ĐẢO CHIỀU K10 (Thực tế - Vòng tròn)
    # Tìm đỉnh đáy thực tế (Local Min/Max trong window 10)
    # Đơn giản hóa: Vẽ những điểm mà Dist_Prev_K10 reset về 0 (đáy/đỉnh mới)
    # Hoặc vẽ dựa trên logic Peak/Valley
    for pt in k10_points:
        col = 'blue' if pt['Type'] == 'Bottom' else 'orange'
        fig.add_trace(go.Scatter(x=[pt['Date']], y=[pt['Price']], mode='markers',
                                 marker=dict(symbol='circle-open', size=12, color=col, line=dict(width=3)),
                                 showlegend=False), row=1, col=1)

    # 5. TÍN HIỆU AI (Mũi tên)
    buys = [s for s in ai_signals if s['Signal'] == 0]
    sells = [s for s in ai_signals if s['Signal'] == 2]
    
    if buys:
        fig.add_trace(go.Scatter(x=[x['Date'] for x in buys], y=[x['Price'] for x in buys], 
                                 mode='markers', name='AI Mua',
                                 marker=dict(symbol='triangle-up', size=12, color='#00CC96')))
    if sells:
        fig.add_trace(go.Scatter(x=[x['Date'] for x in sells], y=[x['Price'] for x in sells], 
                                 mode='markers', name='AI Bán',
                                 marker=dict(symbol='triangle-down', size=12, color='#EF553B')))

    # --- PANEL 2: RSI ---
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], line=dict(color='#AB63FA'), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="purple", opacity=0.1, line_width=0, row=2, col=1)

    fig.update_layout(height=700, xaxis_rangeslider_visible=False, 
                      title_text=f"Phân tích Kỹ thuật & AI Forecast", template='plotly_dark')
    return fig

# --- 5. MAIN APP ---
VN30 = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
        'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
        'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']

st.sidebar.title("🎛️ Control Panel")
option = st.sidebar.selectbox("Chọn chế độ", ["Dashboard Tổng hợp", "Phân tích Chuyên sâu (Chart)"])

if option == "Dashboard Tổng hợp":
    st.title("⚡ VN30 Real-time Signals")
    if st.button("Quét thị trường ngay"):
        # (Giữ code quét thị trường cũ của bạn ở đây nếu muốn)
        st.info("Chức năng quét nhanh (Copy code cũ vào đây nếu cần dùng)")

elif option == "Phân tích Chuyên sâu (Chart)":
    symbol = st.sidebar.selectbox("Chọn mã", VN30)
    days_back = st.sidebar.slider("Số phiên Backtest AI", 30, 90, 60)
    
    if st.button(f"Phân tích {symbol}"):
        with st.spinner("Đang tải dữ liệu và chạy AI..."):
            df = get_data(symbol)
            if df.empty:
                st.error("Lỗi dữ liệu")
            else:
                df_calc = add_indicators(df)
                
                # --- CHẠY AI QUÁ KHỨ ---
                # Chỉ chạy trên days_back phiên cuối cùng để vẽ chart
                ai_signals = []
                start_idx = len(df_calc) - days_back
                
                # Progress bar
                my_bar = st.progress(0)
                
                for i in range(start_idx, len(df_calc)):
                    sig, prob = predict_single(df_calc, symbol, i)
                    if sig in [0, 2]: # Chỉ lưu Mua/Bán
                        ai_signals.append({
                            'Date': df_calc.iloc[i]['Date'],
                            'Price': df_calc.iloc[i]['Close'],
                            'Signal': sig,
                            'Prob': prob
                        })
                    my_bar.progress((i - start_idx) / days_back)
                my_bar.empty()
                
                # --- TÌM ĐIỂM ĐẢO CHIỀU K10 (GROUND TRUTH) ---
                # Logic: Tìm Local Min/Max trong window 10
                k10_points = []
                # Cắt đoạn dữ liệu cần vẽ
                df_plot = df_calc.iloc[start_idx-20:].copy().reset_index(drop=True) 
                
                # Tìm đỉnh đáy thủ công (Window 10)
                for i in range(10, len(df_plot)-10):
                    curr = df_plot.iloc[i]['Close']
                    # Đáy
                    if curr == df_plot.iloc[i-10:i+11]['Close'].min():
                        k10_points.append({'Date': df_plot.iloc[i]['Date'], 'Price': curr, 'Type': 'Bottom'})
                    # Đỉnh
                    elif curr == df_plot.iloc[i-10:i+11]['Close'].max():
                        k10_points.append({'Date': df_plot.iloc[i]['Date'], 'Price': curr, 'Type': 'Top'})

                # --- VẼ CHART ---
                st.plotly_chart(plot_advanced_chart(df_plot, ai_signals, k10_points), use_container_width=True)
                
                # --- THỐNG KÊ ---
                st.subheader("📊 Thống kê Tín hiệu AI (Giai đoạn này)")
                if ai_signals:
                    df_sig = pd.DataFrame(ai_signals)
                    df_sig['Loại'] = df_sig['Signal'].map({0: 'MUA', 2: 'BÁN'})
                    st.dataframe(df_sig[['Date', 'Price', 'Loại', 'Prob']].style.format({"Price": "{:,.0f}", "Prob": "{:.1%}"}))
                else:
                    st.write("Không có tín hiệu Mua/Bán trong giai đoạn này.")

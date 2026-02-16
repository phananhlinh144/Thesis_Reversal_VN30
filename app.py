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

# --- 1. LOAD MODEL (GIỮ NGUYÊN) ---
@st.cache_resource
def load_models():
    try:
        m50 = tf.keras.models.load_model('Full_K10_Win50_Hybrid.keras')
        m10 = tf.keras.models.load_model('Baseline_K10_Win10_Hybrid.keras')
        scaler = joblib.load('smart_scaler_system.pkl')
        return m50, m10, scaler
    except Exception as e:
        st.error(f"Lỗi Load Model: {e}")
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

# --- 2. XỬ LÝ DỮ LIỆU (ĐÃ SỬA LỖI) ---
def get_data(symbol):
    try:
        # Sử dụng TCBS hoặc SSI để ổn định hơn trên server
        stock = Vnstock().stock(symbol=symbol, source='TCBS')
        
        # Lấy dữ liệu 1 năm để đảm bảo đủ window tính toán
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Gọi hàm quote.history đúng chuẩn Vnstock V2
        df = stock.quote.history(start=start_date, end=end_date)
        
        if df is None or df.empty:
            # Thử lại với nguồn SSI nếu TCBS lỗi
            stock = Vnstock().stock(symbol=symbol, source='SSI')
            df = stock.quote.history(start=start_date, end=end_date)

        if df is not None and not df.empty:
            df = df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 
                                    'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            df['Date'] = pd.to_datetime(df['Date'])
            for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            return df.sort_values('Date').reset_index(drop=True)
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Không lấy được dữ liệu cho {symbol}: {e}")
        return pd.DataFrame()

def add_indicators(df):
    if len(df) < 60: return pd.DataFrame()
    g = df.copy()
    
    # Chỉ báo cho AI
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: 
        g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
    for n in [5, 10, 20]: 
        ma = g['Close'].rolling(n).mean()
        g[f'Grad_{n}'] = np.gradient(ma.fillna(method='bfill'))
    
    # Chỉ báo Chart
    g['SMA_20'] = ta.sma(g['Close'], length=20)
    bb = ta.bbands(g['Close'], length=20, std=2)
    g['BB_Upper'] = bb.iloc[:, 0]
    g['BB_Lower'] = bb.iloc[:, 2]
    g['BB_PctB'] = bb.iloc[:, 4]
    g['RSI'] = ta.rsi(g['Close'], length=14)
    g['MACD_Hist'] = ta.macd(g['Close']).iloc[:, 1]
    g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
    g['ATR_Rel'] = ta.atr(g['High'], g['Low'], g['Close'], length=14) / g['Close']
    
    # Logic K10
    rmin = g['Close'].rolling(20).min()
    rmax = g['Close'].rolling(20).max()
    ma20 = g['Close'].rolling(20).mean()
    g['Dist_Prev_K10'] = 0.0
    g.loc[g['Close'] >= ma20, 'Dist_Prev_K10'] = (g['Close'] - rmin) / rmin
    g.loc[g['Close'] < ma20, 'Dist_Prev_K10'] = (g['Close'] - rmax) / rmax
    
    return g.dropna().reset_index(drop=True)

# --- 3. DỰ BÁO (GIỮ NGUYÊN) ---
def predict_single(df_calc, symbol, idx):
    end = idx + 1
    if end < 50: return None
    d50 = df_calc.iloc[end-50:end]
    d10 = df_calc.iloc[end-10:end]
    
    scaler = local_scalers.get(symbol, global_scaler)
    try:
        s50 = scaler.transform(d50[FEATS_FULL].values)
        s10 = scaler.transform(d10[FEATS_FULL].values)
    except:
        s50 = global_scaler.transform(d50[FEATS_FULL].values)
        s10 = global_scaler.transform(d10[FEATS_FULL].values)

    p50 = model_win50.predict(np.expand_dims(s50, axis=0), verbose=0)[0]
    p10 = model_win10.predict(np.expand_dims(s10[:,:17], axis=0), verbose=0)[0]
    
    c50, c10 = np.argmax(p50), np.argmax(p10)
    sig = 1
    if c50 == 0 and c10 == 0: sig = 0
    elif c50 == 2 and c10 == 2: sig = 2
    return sig, (p50[c50] + p10[c10])/2

# --- 4. VẼ BIỂU ĐỒ (SỬA ĐỂ HIỆN THỊ TỐT HƠN) ---

def plot_advanced_chart(df, ai_signals, k10_points):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Nến
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Giá'), row=1, col=1)
    
    # BB & MA
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], line=dict(color='rgba(173,216,230,0.4)'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], line=dict(color='rgba(173,216,230,0.4)'), fill='tonexty', name='Bollinger Bands'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='orange', width=1), name='MA20'), row=1, col=1)

    # Điểm K10 thực tế
    for pt in k10_points:
        col = 'cyan' if pt['Type'] == 'Bottom' else 'yellow'
        fig.add_trace(go.Scatter(x=[pt['Date']], y=[pt['Price']], mode='markers',
                                 marker=dict(symbol='circle-open', size=10, color=col, line=dict(width=2)),
                                 name=f"K10 {pt['Type']}"), row=1, col=1)

    # Tín hiệu AI
    for s in ai_signals:
        symbol_type = 'triangle-up' if s['Signal'] == 0 else 'triangle-down'
        color_type = '#00FF00' if s['Signal'] == 0 else '#FF0000'
        fig.add_trace(go.Scatter(x=[s['Date']], y=[s['Price']], mode='markers',
                                 marker=dict(symbol=symbol_type, size=13, color=color_type),
                                 showlegend=False), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], line=dict(color='#AB63FA'), name='RSI'), row=2, col=1)
    fig.update_layout(height=800, template='plotly_dark', xaxis_rangeslider_visible=False)
    return fig

# --- 5. MAIN APP ---
VN30 = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
        'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
        'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']

st.sidebar.title("🤖 VN30 AI PRO")
option = st.sidebar.selectbox("Chế độ", ["Tổng hợp VN30", "Chi tiết mã & Chart"])

if option == "Tổng hợp VN30":
    st.title("🚀 Tín hiệu Real-time VN30")
    if st.button("Bắt đầu quét"):
        results = []
        pbar = st.progress(0)
        for i, sym in enumerate(VN30):
            df = get_data(sym)
            if not df.empty:
                df_c = add_indicators(df)
                res = predict_single(df_c, sym, len(df_c)-1)
                if res:
                    results.append({'Mã': sym, 'Giá': df_c.iloc[-1]['Close'], 'AI': res[0], 'Prob': res[1]})
            pbar.progress((i+1)/len(VN30))
        
        # Hiển thị bảng kết quả
        res_df = pd.DataFrame(results)
        res_df['Tín hiệu'] = res_df['AI'].map({0: 'MUA 🟢', 1: 'Hold 🟡', 2: 'BÁN 🔴'})
        st.dataframe(res_df[['Mã', 'Giá', 'Tín hiệu', 'Prob']].style.format({'Giá': '{:,.0f}', 'Prob': '{:.1%}'}))

else:
    symbol = st.sidebar.selectbox("Chọn mã", VN30)
    if st.button(f"Phân tích chuyên sâu {symbol}"):
        df = get_data(symbol)
        if not df.empty:
            df_c = add_indicators(df)
            
            # Backtest AI 60 ngày
            ai_sigs = []
            for i in range(len(df_c)-60, len(df_c)):
                r = predict_single(df_c, symbol, i)
                if r and r[0] != 1:
                    ai_sigs.append({'Date': df_c.iloc[i]['Date'], 'Price': df_c.iloc[i]['Close'], 'Signal': r[0], 'Prob': r[1]})
            
            # Tìm K10 thực tế
            k10s = []
            for i in range(len(df_c)-60, len(df_c)-5):
                window = df_c.iloc[i-10:i+11]['Close']
                if df_c.iloc[i]['Close'] == window.min(): k10s.append({'Date': df_c.iloc[i]['Date'], 'Price': df_c.iloc[i]['Close'], 'Type': 'Bottom'})
                if df_c.iloc[i]['Close'] == window.max(): k10s.append({'Date': df_c.iloc[i]['Date'], 'Price': df_c.iloc[i]['Close'], 'Type': 'Top'})
            
            st.plotly_chart(plot_advanced_chart(df_c.tail(80), ai_sigs, k10s), use_container_width=True)

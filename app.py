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
import random

# --- CẤU HÌNH ---
st.set_page_config(page_title="VN30 AI Pro Dashboard", layout="wide", page_icon="📈")

# --- 1. LOAD MODEL & SCALER ---
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
global_scaler = scaler_bundle['global_scaler'] if scaler_bundle else None
local_scalers = scaler_bundle['local_scalers_dict'] if scaler_bundle else {}

FEATS_FULL = [
    'RC_1', 'RC_2', 'RC_3', 'RC_5', 'RC_8', 'RC_13', 'RC_21', 'RC_34', 'RC_55',
    'Grad_5', 'Grad_10', 'Grad_20', 'RSI', 'BB_PctB', 'MACD_Hist', 'Vol_Ratio', 'ATR_Rel', 'Dist_Prev_K10'
]

# --- 2. XỬ LÝ DỮ LIỆU ---
@st.cache_data(ttl=600) # Lưu 10 phút để tránh bị chặn IP
def fetch_stock_data(symbol):
    sources = ['VCI', 'SSI', 'DNSE']
    for src in sources:
        try:
            time.sleep(random.uniform(0.2, 0.5))
            stock = Vnstock().stock(symbol=symbol, source=src)
            df = stock.quote.history(start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), 
                                     end=datetime.now().strftime('%Y-%m-%d'))
            if df is not None and not df.empty:
                return df, src
        except:
            continue
    return pd.DataFrame(), None

def process_data(df):
    try:
        df = df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('Date').drop_duplicates().reset_index(drop=True)
        
        # Chỉ báo AI
        for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: df[f'RC_{n}'] = df['Close'].pct_change(n) * 100
        for n in [5, 10, 20]:
            ma = df['Close'].rolling(n).mean()
            df[f'Grad_{n}'] = np.gradient(ma.fillna(method='bfill').fillna(method='ffill'))
        
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        bb = ta.bbands(df['Close'], length=20, std=2)
        df['BB_Upper'], df['BB_Lower'], df['BB_PctB'] = bb.iloc[:, 0], bb.iloc[:, 2], bb.iloc[:, 4]
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MACD_Hist'] = ta.macd(df['Close']).iloc[:, 1]
        df['Vol_Ratio'] = df['Volume'] / ta.sma(df['Volume'], length=20)
        df['ATR_Rel'] = ta.atr(df['High'], df['Low'], df['Close'], length=14) / df['Close']
        
        # K10
        rmin, rmax, ma20 = df['Close'].rolling(20).min(), df['Close'].rolling(20).max(), df['Close'].rolling(20).mean()
        df['Dist_Prev_K10'] = 0.0
        df.loc[df['Close'] >= ma20, 'Dist_Prev_K10'] = (df['Close'] - rmin) / rmin
        df.loc[df['Close'] < ma20, 'Dist_Prev_K10'] = (df['Close'] - rmax) / rmax
        
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        st.error(f"Lỗi tính toán: {e}")
        return pd.DataFrame()

# --- 3. DỰ BÁO ---
def run_ai_prediction(df, symbol, idx):
    if idx < 50: return None
    try:
        d50 = df.iloc[idx-49:idx+1][FEATS_FULL].values
        d10 = df.iloc[idx-9:idx+1][FEATS_FULL[:17]].values # Model 10 ko có Dist_Prev_K10
        
        sc = local_scalers.get(symbol, global_scaler)
        s50 = sc.transform(d50)
        s10 = sc.transform(np.pad(d10, ((0,0),(0,1))))[:, :17] # Giữ đúng shape 17
        
        p50 = model_win50.predict(np.expand_dims(s50, axis=0), verbose=0)[0]
        p10 = model_win10.predict(np.expand_dims(s10, axis=0), verbose=0)[0]
        
        c50, c10 = np.argmax(p50), np.argmax(p10)
        sig = 0 if (c50 == 0 and c10 == 0) else (2 if (c50 == 2 and c10 == 2) else 1)
        return sig, (p50[c50] + p10[c10])/2
    except:
        return None

# --- 4. GIAO DIỆN ---
VN30 = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']

st.sidebar.title("🤖 VN30 AI PRO")
if st.sidebar.button("🔄 Làm mới dữ liệu (Clear Cache)"):
    st.cache_data.clear()
    st.sidebar.success("Đã xóa bộ nhớ đệm!")

mode = st.sidebar.selectbox("Chế độ", ["Quét Toàn VN30", "Chi tiết mã"])

if mode == "Quét Toàn VN30":
    st.title("🚀 Tín hiệu Real-time")
    if st.button("Bắt đầu quét thị trường"):
        results = []
        pbar = st.progress(0)
        for i, sym in enumerate(VN30):
            df_raw, _ = fetch_stock_data(sym)
            if not df_raw.empty:
                df_p = process_data(df_raw)
                if not df_p.empty:
                    res = run_ai_prediction(df_p, sym, len(df_p)-1)
                    if res:
                        results.append({'Mã': sym, 'Giá': df_p.iloc[-1]['Close'], 'AI': res[0], 'Prob': res[1]})
            pbar.progress((i+1)/len(VN30))
        
        if results:
            res_df = pd.DataFrame(results)
            res_df['Tín hiệu'] = res_df['AI'].map({0: 'MUA 🟢', 1: 'Hold 🟡', 2: 'BÁN 🔴'})
            st.dataframe(res_df[['Mã', 'Giá', 'Tín hiệu', 'Prob']].sort_values('AI'))

else:
    symbol = st.sidebar.selectbox("Chọn mã", VN30)
    lookback = st.sidebar.slider("Xem lại (phiên)", 50, 200, 100)
    
    if st.button(f"Phân tích {symbol}"):
        with st.status(f"Đang xử lý dữ liệu {symbol}...") as status:
            df_raw, source = fetch_stock_data(symbol)
            if not df_raw.empty:
                st.write(f"✅ Đã lấy dữ liệu từ nguồn: **{source}**")
                df_p = process_data(df_raw)
                
                if not df_p.empty:
                    st.write("🧠 Đang tính toán tín hiệu AI...")
                    # 1. Dự báo hiện tại
                    curr = run_ai_prediction(df_p, symbol, len(df_p)-1)
                    
                    if curr:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Giá hiện tại", f"{df_p.iloc[-1]['Close']:,.0f}")
                        c2.subheader(f"Dự báo: {['MUA 🟢', 'THEO DÕI 🟡', 'BÁN 🔴'][curr[0]]}")
                        c3.metric("Xác suất", f"{curr[1]:.1%}")

                        # 2. Backtest để vẽ lên biểu đồ
                        ai_sigs, k10s = [], []
                        for i in range(len(df_p)-lookback, len(df_p)):
                            r = run_ai_prediction(df_p, symbol, i)
                            if r and r[0] != 1:
                                ai_sigs.append({'Date': df_p.iloc[i]['Date'], 'Price': df_p.iloc[i]['Close'], 'Signal': r[0]})
                            
                            if i < len(df_p)-5:
                                win = df_p.iloc[i-10:i+11]['Close']
                                if df_p.iloc[i]['Close'] == win.min(): k10s.append({'Date': df_p.iloc[i]['Date'], 'Price': df_p.iloc[i]['Close'], 'Type': 'Bottom'})
                                if df_p.iloc[i]['Close'] == win.max(): k10s.append({'Date': df_p.iloc[i]['Date'], 'Price': df_p.iloc[i]['Close'], 'Type': 'Top'})
                        
                        # 3. Vẽ biểu đồ
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                        fig.add_trace(go.Candlestick(x=df_p.tail(lookback)['Date'], open=df_p.tail(lookback)['Open'], high=df_p.tail(lookback)['High'], low=df_p.tail(lookback)['Low'], close=df_p.tail(lookback)['Close'], name='Giá'), row=1, col=1)
                        
                        for s in ai_sigs:
                            fig.add_trace(go.Scatter(x=[s['Date']], y=[s['Price']], mode='markers', marker=dict(symbol='triangle-up' if s['Signal']==0 else 'triangle-down', size=12, color='#00FF00' if s['Signal']==0 else '#FF0000'), showlegend=False), row=1, col=1)
                        
                        fig.update_layout(height=600, template='plotly_dark', xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                        status.update(label="Hoàn tất!", state="complete")
                else:
                    st.error("Dữ liệu lỗi sau khi xử lý.")
            else:
                st.error("Không thể kết nối đến server chứng khoán. Thử nhấn 'Làm mới dữ liệu' ở cột trái.")

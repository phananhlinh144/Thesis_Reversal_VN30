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
        st.error(f"Lỗi Load Model/Scaler: {e}")
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
@st.cache_data(ttl=3600)
def get_cached_data(symbol):
    sources = ['VCI', 'SSI', 'DNSE']
    df = None
    for src in sources:
        try:
            # Giảm thời gian chờ để app mượt hơn
            time.sleep(random.uniform(0.1, 0.3)) 
            stock = Vnstock().stock(symbol=symbol, source=src)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
            df = stock.quote.history(start=start_date, end=end_date)
            if df is not None and not df.empty:
                return df, src # Trả về cả dữ liệu và nguồn để hiển thị
        except:
            continue
    return pd.DataFrame(), None

def add_indicators(df):
    if len(df) < 60: return pd.DataFrame()
    try:
        g = df.copy()
        g = g.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        g['Date'] = pd.to_datetime(g['Date'])
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            g[c] = pd.to_numeric(g[c], errors='coerce')
        g = g.sort_values('Date').reset_index(drop=True)

        # AI Features
        for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: 
            g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
        for n in [5, 10, 20]: 
            ma = g['Close'].rolling(n).mean()
            g[f'Grad_{n}'] = np.gradient(ma.fillna(method='ffill').fillna(method='bfill'))
        
        # Indicators
        g['SMA_20'] = ta.sma(g['Close'], length=20)
        bb = ta.bbands(g['Close'], length=20, std=2)
        g['BB_Upper'], g['BB_Lower'], g['BB_PctB'] = bb.iloc[:, 0], bb.iloc[:, 2], bb.iloc[:, 4]
        g['RSI'] = ta.rsi(g['Close'], length=14)
        g['MACD_Hist'] = ta.macd(g['Close']).iloc[:, 1]
        g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
        g['ATR_Rel'] = ta.atr(g['High'], g['Low'], g['Close'], length=14) / g['Close']
        
        # K10 Logic
        rmin, rmax, ma20 = g['Close'].rolling(20).min(), g['Close'].rolling(20).max(), g['Close'].rolling(20).mean()
        g['Dist_Prev_K10'] = 0.0
        g.loc[g['Close'] >= ma20, 'Dist_Prev_K10'] = (g['Close'] - rmin) / rmin
        g.loc[g['Close'] < ma20, 'Dist_Prev_K10'] = (g['Close'] - rmax) / rmax
        
        return g.dropna().reset_index(drop=True)
    except Exception as e:
        st.error(f"Lỗi tính toán chỉ báo: {e}")
        return pd.DataFrame()

# --- 3. DỰ BÁO ---
def predict_single(df_calc, symbol, idx):
    if idx < 50 or idx >= len(df_calc): return None
    try:
        d50 = df_calc.iloc[idx-49:idx+1]
        d10 = df_calc.iloc[idx-9:idx+1]
        
        scaler = local_scalers.get(symbol, global_scaler)
        s50 = scaler.transform(d50[FEATS_FULL].values)
        s10 = scaler.transform(d10[FEATS_FULL].values)

        p50 = model_win50.predict(np.expand_dims(s50, axis=0), verbose=0)[0]
        p10 = model_win10.predict(np.expand_dims(s10[:,:17], axis=0), verbose=0)[0]
        
        c50, c10 = np.argmax(p50), np.argmax(p10)
        sig = 1 
        if c50 == 0 and c10 == 0: sig = 0 
        elif c50 == 2 and c10 == 2: sig = 2 
        return sig, (p50[c50] + p10[c10])/2
    except:
        return None

# --- 4. BIỂU ĐỒ ---
def plot_advanced_chart(df, ai_signals, k10_points):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # Nến giá
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Giá'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
    
    # K10
    for pt in k10_points:
        col = 'cyan' if pt['Type'] == 'Bottom' else 'yellow'
        fig.add_trace(go.Scatter(x=[pt['Date']], y=[pt['Price']], mode='markers', marker=dict(symbol='circle-open', size=10, color=col, line=dict(width=2)), showlegend=False), row=1, col=1)

    # Tín hiệu AI
    for s in ai_signals:
        sym, col = ('triangle-up', '#00FF00') if s['Signal'] == 0 else ('triangle-down', '#FF0000')
        fig.add_trace(go.Scatter(x=[s['Date']], y=[s['Price']], mode='markers', marker=dict(symbol=sym, size=12, color=col), showlegend=False), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], line=dict(color='#AB63FA'), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(height=650, template='plotly_dark', xaxis_rangeslider_visible=False, margin=dict(t=30, b=10))
    return fig

# --- 5. MAIN ---
VN30 = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']

st.sidebar.title("🤖 VN30 AI PRO")
mode = st.sidebar.selectbox("Chế độ", ["Quét VN30", "Chi tiết mã"])

if mode == "Quét VN30":
    st.title("🚀 Tín hiệu AI Real-time")
    if st.button("Bắt đầu quét"):
        results = []
        pbar = st.progress(0)
        for i, sym in enumerate(VN30):
            df_raw, _ = get_cached_data(sym)
            if not df_raw.empty:
                df_c = add_indicators(df_raw)
                if not df_c.empty:
                    res = predict_single(df_c, sym, len(df_c)-1)
                    if res:
                        results.append({'Mã': sym, 'Giá': df_c.iloc[-1]['Close'], 'AI': res[0], 'Prob': res[1]})
            pbar.progress((i+1)/len(VN30))
        
        if results:
            res_df = pd.DataFrame(results)
            res_df['Tín hiệu'] = res_df['AI'].map({0: 'MUA 🟢', 1: 'Hold 🟡', 2: 'BÁN 🔴'})
            st.success("Đã quét xong VN30!")
            st.dataframe(res_df[['Mã', 'Giá', 'Tín hiệu', 'Prob']].sort_values('AI').style.format({'Giá': '{:,.0f}', 'Prob': '{:.1%}'}))

else:
    symbol = st.sidebar.selectbox("Chọn mã", VN30)
    lookback = st.sidebar.slider("Số phiên xem lại", 50, 200, 100)
    
    if st.button(f"Phân tích chuyên sâu {symbol}"):
        with st.status(f"Đang xử lý {symbol}...", expanded=True) as status:
            st.write("Đang tải dữ liệu từ máy chủ...")
            df_raw, source = get_cached_data(symbol)
            
            if not df_raw.empty:
                st.write(f"Đã lấy dữ liệu thành công từ nguồn {source}.")
                df_c = add_indicators(df_raw)
                
                if not df_c.empty:
                    st.write("Đang chạy mô hình AI & Backtest...")
                    # Dự báo hiện tại
                    curr_res = predict_single(df_c, symbol, len(df_c)-1)
                    
                    # Hiển thị kết quả AI
                    if curr_res:
                        status.update(label=f"Hoàn tất phân tích {symbol}!", state="complete")
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric("Giá hiện tại", f"{df_c.iloc[-1]['Close']:,.0f}")
                        with c2: 
                            txt = {0: 'MUA 🟢', 1: 'THEO DÕI 🟡', 2: 'BÁN 🔴'}[curr_res[0]]
                            st.subheader(f"Kết luận: {txt}")
                        with c3: st.metric("Xác suất AI", f"{curr_res[1]:.1%}")
                    
                    # Tính toán biểu đồ
                    ai_sigs, k10s = [], []
                    start_idx = max(0, len(df_c) - lookback)
                    for i in range(start_idx, len(df_c)):
                        r = predict_single(df_c, symbol, i)
                        if r and r[0] != 1:
                            ai_sigs.append({'Date': df_c.iloc[i]['Date'], 'Price': df_c.iloc[i]['Close'], 'Signal': r[0]})
                        
                        if i >= 10 and i < len(df_c) - 5:
                            win = df_c.iloc[i-10:i+11]['Close']
                            if df_c.iloc[i]['Close'] == win.min(): k10s.append({'Type': 'Bottom', 'Date': df_c.iloc[i]['Date'], 'Price': df_c.iloc[i]['Close']})
                            if df_c.iloc[i]['Close'] == win.max(): k10s.append({'Type': 'Top', 'Date': df_c.iloc[i]['Date'], 'Price': df_c.iloc[i]['Close']})
                    
                    st.plotly_chart(plot_advanced_chart(df_c.tail(lookback+20), ai_sigs, k10s), use_container_width=True)
                else:
                    status.update(label="Lỗi: Dữ liệu không đủ để tính toán chỉ báo.", state="error")
            else:
                status.update(label="Lỗi: Không thể tải dữ liệu từ server.", state="error")

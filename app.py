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
import gdown
import os

# --- 1. CẤU HÌNH & LOAD MODEL ---
st.set_page_config(page_title="VN30 AI PRO - Auto Scan", layout="wide")

@st.cache_resource
def load_ai_assets():
    m50 = tf.keras.models.load_model('Full_K10_Win50_Hybrid.keras')
    m10 = tf.keras.models.load_model('Baseline_K10_Win10_Hybrid.keras')
    bundle = joblib.load('smart_scaler_system.pkl')
    return m50, m10, bundle

m50, m10, bundle = load_ai_assets()
vn30_symbols = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']
FEATS_FULL = ['RC_1', 'RC_2', 'RC_3', 'RC_5', 'RC_8', 'RC_13', 'RC_21', 'RC_34', 'RC_55', 'Grad_5', 'Grad_10', 'Grad_20', 'RSI', 'BB_PctB', 'MACD_Hist', 'Vol_Ratio', 'ATR_Rel', 'Dist_Prev_K10']

# --- 2. HÀM XỬ LÝ DỮ LIỆU DRIVE + REALTIME ---
@st.cache_data(ttl=3600)
def get_integrated_data(symbol):
    try:
        # 1. Tải dữ liệu từ Google Drive
        drive_url = 'https://drive.google.com/uc?id=1xG6J9fBEF_Z4KY3x_frUwnhVTSA6HG2r'
        output = 'historical_data.csv'
        if not os.path.exists(output):
            gdown.download(drive_url, output, quiet=True)
        
        full_df_history = pd.read_csv(output)
        
        # Tìm cột mã CP linh hoạt (ticker/Symbol/MaCP)
        ticker_col = next((c for c in full_df_history.columns if c.lower() in ['ticker', 'symbol', 'macp', 'ma']), None)
        
        if ticker_col:
            df_hist = full_df_history[full_df_history[ticker_col] == symbol].copy()
        else:
            df_hist = full_df_history.copy()

        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        last_date_drive = df_hist['Date'].max()
        
        # 2. Lấy dữ liệu mới từ nguồn ổn định (TCBS) để tránh lỗi 403
        stock = Vnstock().stock(symbol=symbol, source='TCBS')
        df_new = stock.quote.history(start=last_date_drive.strftime('%Y-%m-%d'), 
                                     end=datetime.now().strftime('%Y-%m-%d'))
        
        if df_new is not None and not df_new.empty:
            df_new = df_new.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            df_new['Date'] = pd.to_datetime(df_new['Date'])
            df_final = pd.concat([df_hist, df_new]).drop_duplicates(subset=['Date'], keep='last')
        else:
            df_final = df_hist
            
        return df_final.sort_values('Date').reset_index(drop=True)
    except:
        # Fallback: Nếu lỗi thì lấy thẳng từ Vnstock 300 phiên
        try:
            df = Vnstock().stock(symbol=symbol, source='TCBS').quote.history(start=(datetime.now() - timedelta(days=300)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
            df = df.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            df['Date'] = pd.to_datetime(df['Date'])
            return df.sort_values('Date').reset_index(drop=True)
        except: return pd.DataFrame()

def compute_features(df):
    if len(df) < 60: return pd.DataFrame()
    g = df.copy()
    # Tính các chỉ báo (Giữ nguyên logic của bạn)
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean().fillna(method='bfill')
        g[f'Grad_{n}'] = np.gradient(ma)
    g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
    g['RSI'] = ta.rsi(g['Close'], length=14)
    bb = ta.bbands(g['Close'], length=20, std=2)
    g['BB_Lower'], g['BB_Mid'], g['BB_Upper'], g['BB_PctB'] = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2], bb.iloc[:, 4]
    g['MACD_Hist'] = ta.macd(g['Close']).iloc[:, 1]
    g['ATR_Rel'] = ta.atr(g['High'], g['Low'], g['Close'], length=14) / g['Close']
    ma20 = g['Close'].rolling(20).mean()
    g['Dist_Prev_K10'] = 0.0
    g.loc[g['Close'] >= ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).min()) / g['Close'].rolling(20).min()
    g.loc[g['Close'] < ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).max()) / g['Close'].rolling(20).max()
    return g.dropna().reset_index(drop=True)

def predict_logic(df_c, symbol, target_idx=-1):
    if target_idx == -1:
        d50, d10 = df_c.tail(50), df_c.tail(10)
    else:
        d50 = df_c.iloc[target_idx-49 : target_idx+1]
        d10 = df_c.iloc[target_idx-9 : target_idx+1]
    
    if len(d50) < 50: return None
    scaler = bundle['local_scalers_dict'].get(symbol, bundle['global_scaler'])
    s50 = scaler.transform(d50[FEATS_FULL].values)
    s10 = scaler.transform(d10[FEATS_FULL].values)
    p50 = m50.predict(np.expand_dims(s50, 0), verbose=0)[0]
    p10 = m10.predict(np.expand_dims(s10[:, :17], 0), verbose=0)[0]
    c50, c10 = np.argmax(p50), np.argmax(p10)
    labels = {0: "mua", 1: "ngang", 2: "bán"}
    ensemble = "THEO DÕI"
    if c50 == 0 and c10 == 0: ensemble = "MUA"
    elif c50 == 2 and c10 == 2: ensemble = "BÁN"
    return {'win50': f"{labels[c50]} {p50[c50]:.0%}", 'win10': f"{labels[c10]} {p10[c10]:.0%}", 'ensemble': ensemble, 'c50': c50}

# --- 3. AUTO-SCAN KHI VỪA VÀO APP ---
if 'all_results' not in st.session_state:
    with st.status("🚀 Đang khởi tạo dữ liệu VN30 từ Drive & Real-time...", expanded=True) as status:
        results = []
        for i, sym in enumerate(vn30_symbols):
            st.write(f"⏳ Đang phân tích: {sym}...")
            df_sym = get_integrated_data(sym)
            df_c_sym = compute_features(df_sym)
            if not df_c_sym.empty:
                r = predict_logic(df_c_sym, sym)
                results.append({
                    "Mã": sym, "Giá HT": f"{df_c_sym['Close'].iloc[-1]:,.0f}",
                    "win50": r['win50'], "win10": r['win10'], "ENSEMBLE": r['ensemble']
                })
        st.session_state.all_results = pd.DataFrame(results)
        status.update(label="✅ Đã quét xong 30 mã!", state="complete", expanded=False)

# --- 4. GIAO DIỆN TABS (ĐÃ ĐẢO THỨ TỰ) ---
t1, t2, t3 = st.tabs(["🚀 Tổng hợp VN30", "📊 Đồ thị AI", "🔍 Chi tiết 20 phiên"])

# --- TAB 1: TỔNG HỢP (Hiện ngay khi vào) ---
with t1:
    df_res = st.session_state.all_results
    c1, c2, c3 = st.columns(3)
    with c1: 
        st.success("🟢 PHÂN KHÚC MUA")
        st.dataframe(df_res[df_res['ENSEMBLE'] == "MUA"], use_container_width=True, hide_index=True)
    with c2: 
        st.error("🔴 PHÂN KHÚC BÁN")
        st.dataframe(df_res[df_res['ENSEMBLE'] == "BÁN"], use_container_width=True, hide_index=True)
    with c3: 
        st.warning("🟡 THEO DÕI")
        st.dataframe(df_res[df_res['ENSEMBLE'] == "THEO DÕI"], use_container_width=True, hide_index=True)
    
    if st.button("🔄 Làm mới dữ liệu (Quét lại)"):
        del st.session_state.all_results
        st.rerun()

# --- TAB 2: ĐỒ THỊ ---
with t2:
    sel_sym = st.selectbox("Chọn mã soi kỹ", vn30_symbols)
    df = get_integrated_data(sel_sym)
    df_c = compute_features(df)
    if not df_c.empty:
        chart_df = df_c.tail(100).copy()
        signals = [predict_logic(df_c, sel_sym, target_idx=i) for i in range(len(df_c)-50, len(df_c))]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
        fig.add_trace(go.Candlestick(x=chart_df['Date'], open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name="Giá"), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['BB_Upper'], line=dict(color='rgba(173,216,230,0.3)'), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['BB_Lower'], line=dict(color='rgba(173,216,230,0.3)'), name="BB Lower"), row=1, col=1)
        
        # Marker dự báo
        sig_dates, sig_prices = chart_df['Date'].tail(50), chart_df['Close'].tail(50)
        for i, s in enumerate(signals):
            if s['ensemble'] == "MUA":
                fig.add_trace(go.Scatter(x=[sig_dates.iloc[i]], y=[sig_prices.iloc[i]*0.985], mode="markers", marker=dict(symbol="triangle-up", size=10, color="white", line=dict(width=1, color="black")), showlegend=False), row=1, col=1)
            elif s['ensemble'] == "BÁN":
                fig.add_trace(go.Scatter(x=[sig_dates.iloc[i]], y=[sig_prices.iloc[i]*1.015], mode="markers", marker=dict(symbol="triangle-down", size=10, color="black", line=dict(width=1, color="white")), showlegend=False), row=1, col=1)
            dot_color = "white" if s['c50'] == 0 else ("black" if s['c50'] == 2 else "gray")
            fig.add_trace(go.Scatter(x=[sig_dates.iloc[i]], y=[sig_prices.iloc[i]], mode="markers", marker=dict(symbol="circle", size=5, color=dot_color), showlegend=False), row=1, col=1)

        fig.add_trace(go.Bar(x=chart_df['Date'], y=chart_df['Volume'], name="Volume", marker_color='rgba(100,100,100,0.2)'), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['RSI'], name="RSI", line=dict(color="orange")), row=2, col=1)
        fig.update_layout(height=600, template='plotly_dark', xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: CHI TIẾT ---
with t3:
    sel_sym_hist = st.selectbox("Mã xem lịch sử", vn30_symbols, key="h_box")
    df_h = get_integrated_data(sel_sym_hist)
    df_hc = compute_features(df_h)
    if not df_hc.empty:
        hist = []
        for i in range(len(df_hc)-20, len(df_hc)):
            r = predict_logic(df_hc, sel_sym_hist, target_idx=i)
            hist.append({"Ngày": df_hc['Date'].iloc[i].strftime('%d/%m/%Y'), "Giá": f"{df_hc['Close'].iloc[i]:,.0f}", "win50": r['win50'], "win10": r['win10'], "ENSEMBLE": r['ensemble']})
        st.table(pd.DataFrame(hist[::-1]))

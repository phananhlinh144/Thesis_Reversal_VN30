import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pandas_ta as ta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from vnstock import Vnstock

# ==============================================================================
# 1. CẤU HÌNH & CACHE MODEL
# ==============================================================================
st.set_page_config(page_title="VN30 AI Forecast 2026", layout="wide")

MODEL_WIN50_PATH = 'models_scaling/Full_K10_Win50_Hybrid.keras'
MODEL_WIN10_PATH = 'models_scaling/Baseline_K10_Win10_Hybrid.keras'
SCALER_PATH      = 'models_scaling/smart_scaler_system.pkl'
CSV_PATH         = 'vn30_data_raw.csv'

FINAL_FEATURES = [
    'RC_1', 'RC_2', 'RC_3', 'RC_5', 'RC_8', 'RC_13', 'RC_21', 'RC_34', 'RC_55',
    'Grad_5', 'Grad_10', 'Grad_20', 'RSI', 'BB_PctB', 'MACD_Hist', 'Vol_Ratio', 'ATR_Rel'
]
FEATS_FULL = FINAL_FEATURES + ['Dist_Prev_K10']

@st.cache_resource
def load_ai_system():
    try:
        m50 = tf.keras.models.load_model(MODEL_WIN50_PATH)
        m10 = tf.keras.models.load_model(MODEL_WIN10_PATH)
        scaler_data = joblib.load(SCALER_PATH)
        return m50, m10, scaler_data
    except Exception as e:
        st.error(f"Lỗi tải Model: {e}")
        return None, None, None

model_win50, model_win10, scaler_bundle = load_ai_system()

# ==============================================================================
# 2. HÀM XỬ LÝ DỮ LIỆU (Giữ nguyên logic gốc của bạn)
# ==============================================================================

def compute_features(df):
    g = df.copy()
    if len(g) < 60: return pd.DataFrame()
    
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: 
        g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
    
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean()
        g[f'Grad_{n}'] = np.gradient(ma.fillna(method='bfill'))
        
    g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
    g['RSI'] = ta.rsi(g['Close'], length=14)
    
    bb = ta.bbands(g['Close'], length=20, std=2)
    g['BB_PctB'] = bb.iloc[:, 4]
    g['BB_Upper'] = bb.iloc[:, 2]
    g['BB_Lower'] = bb.iloc[:, 0]
    
    g['MACD_Hist'] = ta.macd(g['Close']).iloc[:, 1]
    g['ATR_Rel'] = ta.atr(g['High'], g['Low'], g['Close'], length=14) / g['Close']
    
    rmin, rmax = g['Close'].rolling(20).min(), g['Close'].rolling(20).max()
    ma20 = g['Close'].rolling(20).mean()
    
    g['Dist_Prev_K10'] = 0.0
    g.loc[g['Close'] >= ma20, 'Dist_Prev_K10'] = (g['Close'] - rmin) / rmin
    g.loc[g['Close'] < ma20, 'Dist_Prev_K10'] = (g['Close'] - rmax) / rmax
    
    return g.dropna().reset_index(drop=True)

def get_data_for_symbol(symbol, fetch_live=True):
    try:
        full_df = pd.read_csv(CSV_PATH)
        df_hist = full_df[full_df['Symbol'] == symbol].copy()
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        df_hist = df_hist.sort_values('Date')
        
        if not fetch_live: return df_hist

        try:
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            live_df = stock.quote.now()
            if not live_df.empty:
                current_price = float(live_df['close'].iloc[0])
                current_vol = float(live_df['volume'].iloc[0])
                current_high = float(live_df['high'].iloc[0])
                current_low = float(live_df['low'].iloc[0])
                
                today = pd.Timestamp(datetime.now().date())
                if df_hist.empty or df_hist.iloc[-1]['Date'].date() < today.date():
                    new_row = pd.DataFrame([{'Date': today, 'Open': current_price, 'High': current_high, 
                                             'Low': current_low, 'Close': current_price, 
                                             'Volume': current_vol, 'Symbol': symbol}])
                    df_hist = pd.concat([df_hist, new_row], ignore_index=True)
                else:
                    idx = df_hist.index[-1]
                    df_hist.at[idx, 'Close'] = current_price
                    df_hist.at[idx, 'High'] = max(df_hist.at[idx, 'High'], current_high)
                    df_hist.at[idx, 'Low'] = min(df_hist.at[idx, 'Low'], current_low)
                    df_hist.at[idx, 'Volume'] = current_vol
        except: pass
        return df_hist
    except: return pd.DataFrame()

def predict_single_row(df_calc, idx_target=-1, symbol=''):
    if len(df_calc) < 55: return None
    end_pos = idx_target + 1 if idx_target != -1 else len(df_calc)
    if end_pos < 50: return None
    
    d50, d10 = df_calc.iloc[end_pos-50 : end_pos], df_calc.iloc[end_pos-10 : end_pos]
    
    scaler = scaler_bundle['local_scalers_dict'].get(symbol, scaler_bundle['global_scaler'])
    # Fix lỗi .values để đồng nhất tên cột
    s50 = scaler.transform(d50[FEATS_FULL])
    s10 = scaler.transform(d10[FEATS_FULL])

    p50 = model_win50.predict(np.expand_dims(s50, axis=0), verbose=0)[0]
    p10 = model_win10.predict(np.expand_dims(s10[:, :17], axis=0), verbose=0)[0] 
    
    cls50, cls10 = np.argmax(p50), np.argmax(p10)
    signal = "NGANG"
    if cls50 == 0 and cls10 == 0: signal = "MUA"
    elif cls50 == 2 and cls10 == 2: signal = "BÁN"
    
    return {
        'Date': df_calc.iloc[end_pos-1]['Date'],
        'Close': df_calc.iloc[end_pos-1]['Close'],
        'Ensemble': signal,
        'Model_50': f"{['mua', 'ngang', 'bán'][cls50]} ({p50[cls50]:.0%})", 
        'Model_10': f"{['mua', 'ngang', 'bán'][cls10]} ({p10[cls10]:.0%})"  
    }

# ==============================================================================
# 3. GIAO DIỆN
# ==============================================================================

st.title("🤖 VN30 AI TRADING SYSTEM")
tab1, tab2, tab3 = st.tabs(["📊 DỰ BÁO CHUNG", "📈 BIỂU ĐỒ CHI TIẾT", "📝 DỮ LIỆU CHI TIẾT"])

vn30_list = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
             'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
             'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']

# TAB 1: 3 BẢNG NGANG
with tab1:
    if st.button("🚀 CHẠY DỰ BÁO TOÀN BỘ VN30"):
        report_data = []
        prog = st.progress(0)
        for i, sym in enumerate(vn30_list):
            df_c = compute_features(get_data_for_symbol(sym))
            res = predict_single_row(df_c, symbol=sym)
            if res:
                report_data.append({'Mã CK': sym, 'Giá': f"{int(res['Close']):,}", 
                                   'Ensemble (AI)': res['Ensemble'], 'Dài': res['Model_50'], 'Ngắn': res['Model_10']})
            prog.progress((i + 1) / len(vn30_list))
        
        final_df = pd.DataFrame(report_data)
        for status, color, title in [('MUA', 'green', '🟢 DANH SÁCH MUA'), 
                                     ('BÁN', 'red', '🔴 DANH SÁCH BÁN'), 
                                     ('NGANG', 'orange', '🟡 TRẠNG THÁI THEO DÕI')]:
            st.subheader(title)
            st.dataframe(final_df[final_df['Ensemble (AI)'] == status], use_container_width=True, hide_index=True)

# TAB 2: BIỂU ĐỒ PAGAN STYLE & VOLUME
with tab2:
    c1, c2 = st.columns([1, 1])
    with c1: selected_sym = st.selectbox("Chọn mã cổ phiếu:", vn30_list)
    with c2: date_range = st.date_input("Khoảng thời gian:", [datetime.now() - timedelta(days=90), datetime.now()])

    if selected_sym:
        df_c = compute_features(get_data_for_symbol(selected_sym))
        mask = (df_c['Date'].dt.date >= date_range[0]) & (df_c['Date'].dt.date <= date_range[1])
        plot_df = df_c.loc[mask]
        
        sig_df = pd.DataFrame([predict_single_row(df_c, idx_target=idx, symbol=selected_sym) for idx in plot_df.index])

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
        
        # Nến không viền
        fig.add_trace(go.Candlestick(x=plot_df['Date'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='Giá', increasing_line_width=0, decreasing_line_width=0), row=1, col=1)
        # Volume mờ phía sau
        fig.add_trace(go.Bar(x=plot_df['Date'], y=plot_df['Volume'], name='Khối lượng', marker_color='rgba(128, 128, 128, 0.15)', showlegend=False), row=1, col=1, secondary_y=True)
        
        if not sig_df.empty:
            # Điểm dự báo Pagan Style (Viền tròn rỗng)
            fig.add_trace(go.Scatter(x=sig_df['Date'], y=sig_df['Close'], mode='markers', marker=dict(symbol='circle-open', size=7, color='black', line=dict(width=1)), name='AI Scan'), row=1, col=1)
            # Mua/Bán
            fig.add_trace(go.Scatter(x=sig_df[sig_df['Ensemble']=='MUA']['Date'], y=sig_df[sig_df['Ensemble']=='MUA']['Close']*0.98, mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='MUA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=sig_df[sig_df['Ensemble']=='BÁN']['Date'], y=sig_df[sig_df['Ensemble']=='BÁN']['Close']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='BÁN'), row=1, col=1)

        fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.update_layout(height=700, xaxis_rangeslider_visible=True, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: CHI TIẾT CÓ MÀU
with tab3:
    sym_tab3 = st.selectbox("Chọn mã chi tiết:", vn30_list)
    if sym_tab3:
        df_c = compute_features(get_data_for_symbol(sym_tab3))
        res_list = [predict_single_row(df_c, idx_target=idx, symbol=sym_tab3) for idx in range(len(df_c)-1, len(df_c)-16, -1)]
        df_table = pd.DataFrame([r for r in res_list if r])
        
        def style_ensemble(val):
            color = {'MUA': '#00CC00', 'BÁN': '#FF0000', 'NGANG': '#FFBB00'}.get(val, 'black')
            return f'color: {color}; font-weight: bold'

        st.dataframe(df_table.style.applymap(style_ensemble, subset=['Ensemble']), use_container_width=True, hide_index=True)

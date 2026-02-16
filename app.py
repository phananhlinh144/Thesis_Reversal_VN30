import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from vnstock import Vnstock # Đảm bảo đã pip install vnstock
import time
import gdown
import os

# --- 1. CẤU HÌNH & LOAD MODEL ---
st.set_page_config(page_title="VN30 AI PRO", layout="wide")

@st.cache_resource
def load_ai_assets():
    # Giả sử các file này nằm cùng thư mục app.py
    m50 = tf.keras.models.load_model('Full_K10_Win50_Hybrid.keras')
    m10 = tf.keras.models.load_model('Baseline_K10_Win10_Hybrid.keras')
    bundle = joblib.load('smart_scaler_system.pkl')
    return m50, m10, bundle

m50, m10, bundle = load_ai_assets()
vn30_symbols = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']
FEATS_FULL = ['RC_1', 'RC_2', 'RC_3', 'RC_5', 'RC_8', 'RC_13', 'RC_21', 'RC_34', 'RC_55', 'Grad_5', 'Grad_10', 'Grad_20', 'RSI', 'BB_PctB', 'MACD_Hist', 'Vol_Ratio', 'ATR_Rel', 'Dist_Prev_K10']

# --- 2. HÀM XỬ LÝ DỮ LIỆU ---
@st.cache_data(ttl=3600)
def get_integrated_data(symbol):
    try:
        # 1. Tải data lịch sử từ Drive (Xương sống của App)
        drive_url = 'https://drive.google.com/uc?id=1xG6J9fBEF_Z4KY3x_frUwnhVTSA6HG2r'
        output = 'historical_data.csv'
        if not os.path.exists(output):
            gdown.download(drive_url, output, quiet=True)
        
        full_df_history = pd.read_csv(output)
        # Chuẩn hóa tên cột mã cổ phiếu
        ticker_col = next((c for c in full_df_history.columns if c.lower() in ['ticker', 'symbol', 'macp', 'ma']), None)
        if ticker_col:
            df_hist = full_df_history[full_df_history[ticker_col] == symbol].copy()
        else:
            df_hist = pd.DataFrame()

        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        last_date_drive = df_hist['Date'].max()
        
        # 2. Lấy data mới (Real-time) từ Vnstock
        # Lưu ý: Vnstock trên Cloud thường bị VCI chặn, ta bọc trong try-except chặt chẽ
        try:
            client = Vnstock()
            stock = client.stock(symbol=symbol, source='VCI')
            # Lấy từ ngày cuối của Drive đến hiện tại
            df_new = stock.quote.history(start=last_date_drive.strftime('%Y-%m-%d'), 
                                         end=datetime.now().strftime('%Y-%m-%d'))
            
            if df_new is not None and not df_new.empty:
                df_new = df_new.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
                df_new['Date'] = pd.to_datetime(df_new['Date'])
                df_final = pd.concat([df_hist, df_new]).drop_duplicates(subset=['Date'], keep='last')
                return df_final.sort_values('Date').reset_index(drop=True)
        except Exception as api_err:
            st.sidebar.warning(f"API {symbol} limit: Dùng dữ liệu Drive")
            
        return df_hist.sort_values('Date').reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame()

def compute_features(df):
    if len(df) < 60: return pd.DataFrame()
    g = df.copy()
    # Logic tính toán giữ nguyên...
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean().fillna(method='ffill').fillna(method='bfill')
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

# --- Predict Logic (Giữ nguyên) ---
def predict_logic(df_c, symbol, target_idx=-1):
    if target_idx == -1: d50, d10 = df_c.tail(50), df_c.tail(10)
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
    return {'win50': f"{labels[c50]} {p50[c50]:.0%}", 'win10': f"{labels[c10]} {p10[c10]:.0%}", 'ENSEMBLE': ensemble, 'c50': c50}

# --- 3. AUTO-SCAN (Chiến thuật tối ưu Cloud) ---
if 'all_results' not in st.session_state:
    with st.status("🚀 Đang khởi tạo dữ liệu VN30...", expanded=True) as status:
        results = []
        progress_bar = st.progress(0)
        for i, sym in enumerate(vn30_symbols):
            # Cập nhật trạng thái
            progress = (i + 1) / len(vn30_symbols)
            progress_bar.progress(progress)
            
            # Nghỉ ngắt quãng để tránh Cloud bị ban IP
            if i > 0 and i % 5 == 0:
                time.sleep(3) 

            df_sym = get_integrated_data(sym)
            if not df_sym.empty:
                df_c_sym = compute_features(df_sym)
                if not df_c_sym.empty:
                    r = predict_logic(df_c_sym, sym)
                    if r:
                        results.append({
                            "Mã": sym, "Giá HT": f"{df_c_sym['Close'].iloc[-1]:,.0f}",
                            "win50": r['win50'], "win10": r['win10'], "ENSEMBLE": r['ENSEMBLE']
                        })
            time.sleep(0.5) # Nghỉ nhẹ giữa các mã
            
        st.session_state.all_results = pd.DataFrame(results, columns=["Mã", "Giá HT", "win50", "win10", "ENSEMBLE"])
        status.update(label="✅ Đã quét xong!", state="complete", expanded=False)

# --- 4. GIAO DIỆN TABS (Giữ nguyên phần hiển thị của bạn) ---
# ... (Phần code Tabs giữ nguyên như cũ)

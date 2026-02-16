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
import warnings

warnings.filterwarnings('ignore')

# --- 1. CẤU HÌNH & LOAD MODEL ---
st.set_page_config(page_title="VN30 AI TRADING", layout="wide", page_icon="📈")

@st.cache_resource
def load_ai_system():
    try:
        m50 = tf.keras.models.load_model('Full_K10_Win50_Hybrid.keras')
        m10 = tf.keras.models.load_model('Baseline_K10_Win10_Hybrid.keras')
        bundle = joblib.load('smart_scaler_system.pkl')
        return m50, m10, bundle
    except Exception as e:
        st.error(f"❌ Lỗi load model/scaler: {e}")
        return None, None, None

m50, m10, bundle = load_ai_system()

VN30_LIST = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
             'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 
             'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']

FEATS_FULL = ['RC_1', 'RC_2', 'RC_3', 'RC_5', 'RC_8', 'RC_13', 'RC_21', 'RC_34', 'RC_55', 
              'Grad_5', 'Grad_10', 'Grad_20', 'RSI', 'BB_PctB', 'MACD_Hist', 'Vol_Ratio', 'ATR_Rel', 'Dist_Prev_K10']

# --- 2. HÀM XỬ LÝ DỮ LIỆU LAI (HYBRID DATA) ---

def get_hybrid_data(symbol):
    """Lấy dữ liệu cũ từ file/API lịch sử và nối với DNSE từ 11/01/2026"""
    try:
        stock = Vnstock().stock(symbol=symbol, source='DNSE')
        
        # 1. Lấy dữ liệu quá khứ (trước 11/01/2026) - Giả định lấy từ nguồn lịch sử chuẩn
        # Ở đây ta lấy từ 1 năm trước đến 10/01/2026
        df_old = stock.quote.history(start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), 
                                     end='2026-01-10')
        
        # 2. Lấy dữ liệu mới từ DNSE (từ 11/01/2026 đến nay)
        df_new = stock.quote.history(start='2026-01-11', 
                                     end=datetime.now().strftime('%Y-%m-%d'))
        
        # Gộp dữ liệu
        df = pd.concat([df_old, df_new], ignore_index=True)
        if df is None or df.empty: return pd.DataFrame()
        
        df = df.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        df['Date'] = pd.to_datetime(df['Date'])
        # Xóa trùng nếu có và sắp xếp
        df = df.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Lỗi fetch data {symbol}: {e}")
        return pd.DataFrame()

def compute_features(df):
    if len(df) < 60: return pd.DataFrame()
    g = df.copy()
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean().fillna(method='bfill')
        g[f'Grad_{n}'] = np.gradient(ma)
    g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
    g['RSI'] = ta.rsi(g['Close'], length=14)
    bb = ta.bbands(g['Close'], length=20, std=2)
    g['BB_PctB'], g['BB_Upper'], g['BB_Lower'] = bb.iloc[:, 4], bb.iloc[:, 2], bb.iloc[:, 0]
    g['MACD_Hist'] = ta.macd(g['Close']).iloc[:, 1]
    g['ATR_Rel'] = ta.atr(g['High'], g['Low'], g['Close'], length=14) / g['Close']
    ma20 = g['Close'].rolling(20).mean()
    g['Dist_Prev_K10'] = 0.0
    g.loc[g['Close'] >= ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).min()) / g['Close'].rolling(20).min()
    g.loc[g['Close'] < ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).max()) / g['Close'].rolling(20).max()
    return g.dropna().reset_index(drop=True)

def predict_single_step(df_feat, symbol, row_idx=-1):
    if len(df_feat) < 55: return None
    end = len(df_feat) + row_idx + 1 if row_idx < 0 else row_idx + 1
    d50 = df_feat.iloc[max(0, end-50):end]
    d10 = df_feat.iloc[max(0, end-10):end]
    if len(d50) < 50: return None
    scaler = bundle['local_scalers_dict'].get(symbol, bundle['global_scaler'])
    s50 = scaler.transform(d50[FEATS_FULL].values)
    s10 = scaler.transform(d10[FEATS_FULL].values)
    p50 = m50.predict(np.expand_dims(s50, 0), verbose=0)[0]
    p10 = m10.predict(np.expand_dims(s10[:, :17], 0), verbose=0)[0]
    c50, c10 = np.argmax(p50), np.argmax(p10)
    signal = "THEO DÕI"
    if c50 == 0 and c10 == 0: signal = "MUA"
    elif c50 == 2 and c10 == 2: signal = "BÁN"
    return {"Date": df_feat.iloc[row_idx]['Date'], "Close": df_feat.iloc[row_idx]['Close'],
            "c50": c50, "p50": p50[c50], "c10": c10, "p10": p10[c10], "Signal": signal}

# --- 3. GIAO DIỆN ---
st.title("🤖 VN30 AI QUANT SYSTEM (11/01 Hybrid)")
tab1, tab2, tab3 = st.tabs(["🚀 DỰ BÁO VN30", "📊 SOI MÃ CHI TIẾT", "📝 LỊCH SỬ DỰ BÁO"])

# TAB 1: QUÉT TOÀN BỘ (Dữ liệu Live nối lịch sử)
with tab1:
    if st.button("⚡ CHẠY QUÉT REAL-TIME VN30", type="primary"):
        results = []
        p_bar = st.progress(0)
        for i, sym in enumerate(VN30_LIST):
            df = get_hybrid_data(sym)
            df_c = compute_features(df)
            res = predict_single_step(df_c, sym, -1)
            if res:
                results.append({"Mã": sym, "Giá": res['Close'], "Win50": f"{res['c50']} ({res['p50']:.0%})", 
                                "Win10": f"{res['c10']} ({res['p10']:.0%})", "ENSEMBLE": res['Signal']})
            p_bar.progress((i+1)/30)
        st.session_state.scan_data = pd.DataFrame(results)
    
    if 'scan_data' in st.session_state:
        st.dataframe(st.session_state.scan_data, use_container_width=True, hide_index=True)

# TAB 2 & 3: XỬ LÝ THEO MÃ
with tab2:
    sel_sym = st.selectbox("Chọn mã phân tích:", VN30_LIST)
    if st.button(f"🔍 Phân tích {sel_sym}"):
        df_raw = get_hybrid_data(sel_sym)
        df_feat = compute_features(df_raw)
        if not df_feat.empty:
            # Lưu vào session để Tab 3 dùng luôn, không cần fetch lại
            st.session_state.current_df_feat = df_feat
            st.session_state.current_sym = sel_sym
            
            # Vẽ Chart (60 phiên gần nhất)
            df_p = df_feat.tail(60).copy()
            # (Phần code vẽ Plotly giữ nguyên như bản trước của bạn...)
            st.success(f"Đã tải dữ liệu {sel_sym}. Chuyển sang Tab 3 để xem lịch sử tùy chọn.")

with tab3:
    st.header("📝 Tra cứu lịch sử dự báo AI")
    if 'current_df_feat' in st.session_state:
        df_feat = st.session_state.current_df_feat
        sym = st.session_state.current_sym
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("Từ ngày:", datetime.now() - timedelta(days=20))
        with col_d2:
            end_date = st.date_input("Đến ngày:", datetime.now())
            
        if st.button("Hiển thị lịch sử"):
            # Lọc dataframe theo ngày
            mask = (df_feat['Date'].dt.date >= start_date) & (df_feat['Date'].dt.date <= end_date)
            df_filtered = df_feat.loc[mask]
            
            if not df_filtered.empty:
                hist_results = []
                for idx in range(len(df_filtered)):
                    # Lấy index thực tế trong df_feat để đảm bảo đủ window 50 phiên trước đó
                    actual_idx = df_filtered.index[idx]
                    res = predict_single_step(df_feat, sym, actual_idx)
                    if res:
                        lbl = {0: 'mua', 1: 'ngang', 2: 'bán'}
                        hist_results.append({
                            "Ngày": res['Date'].strftime('%d/%m/%Y'),
                            "Giá": f"{res['Close']:,.0f}",
                            "win50": f"{lbl[res['c50']]} {res['p50']:.0%}",
                            "win10": f"{lbl[res['c10']]} {res['p10']:.0%}",
                            "ENSEMBLE": res['Signal']
                        })
                
                st.table(pd.DataFrame(hist_results[::-1])) # Hiển thị ngày mới nhất lên đầu
            else:
                st.warning("Không có dữ liệu trong khoảng ngày này.")
    else:
        st.info("Hãy chọn mã và bấm Phân tích ở Tab 2 trước.")

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pandas_ta as ta
import time
from datetime import datetime, timedelta
from vnstock import Vnstock

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="VN30 AI Trading Bot",
    page_icon="📈",
    layout="wide"
)

# --- 1. LOAD MODEL & CACHE (Chỉ load 1 lần để web chạy nhanh) ---
@st.cache_resource
def load_ai_models():
    # Lưu ý: Khi up lên GitHub, hãy để file model cùng thư mục với app.py
    # Hoặc sửa đường dẫn này cho đúng với cấu trúc folder trên GitHub của bạn
    try:
        m_win50 = tf.keras.models.load_model('Full_K10_Win50_Hybrid.keras')
        m_win10 = tf.keras.models.load_model('Baseline_K10_Win10_Hybrid.keras')
        scaler_data = joblib.load('smart_scaler_system.pkl')
        return m_win50, m_win10, scaler_data
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        return None, None, None

model_win50, model_win10, scaler_bundle = load_ai_models()

if scaler_bundle:
    global_scaler = scaler_bundle['global_scaler']
    local_scalers = scaler_bundle['local_scalers_dict']

# --- DANH SÁCH VN30 ---
VN30_LIST = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
             'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
             'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']

FINAL_FEATURES = [
    'RC_1', 'RC_2', 'RC_3', 'RC_5', 'RC_8', 'RC_13', 'RC_21', 'RC_34', 'RC_55',
    'Grad_5', 'Grad_10', 'Grad_20', 'RSI', 'BB_PctB', 'MACD_Hist', 'Vol_Ratio', 'ATR_Rel'
]
FEATS_FULL = FINAL_FEATURES + ['Dist_Prev_K10']

# --- 2. HÀM XỬ LÝ DỮ LIỆU (Giữ nguyên logic của bạn) ---
def get_data_efficient(symbol):
    try:
        stock = Vnstock().stock(symbol=symbol, source='VCI') 
        df = stock.quote.history(start=(datetime.now() - timedelta(days=300)).strftime('%Y-%m-%d'), 
                                 end=datetime.now().strftime('%Y-%m-%d'))
        
        if df is None or df.empty: return pd.DataFrame()

        df = df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 
                                'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols_to_numeric: df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Real-time update logic
        try:
            live_df = stock.quote.now()
            if not live_df.empty:
                current_price = float(live_df['close'].iloc[0])
                current_vol   = float(live_df['volume'].iloc[0])
                current_high  = float(live_df['high'].iloc[0])
                current_low   = float(live_df['low'].iloc[0])
                
                if current_high == 0: current_high = current_price
                if current_low == 0: current_low = current_price

                today_date = pd.Timestamp(datetime.now().date())
                last_hist_date = df.iloc[-1]['Date']

                if last_hist_date.date() == today_date.date():
                    last_idx = df.index[-1]
                    df.at[last_idx, 'Close'] = current_price
                    df.at[last_idx, 'High']  = max(df.at[last_idx, 'High'], current_high)
                    df.at[last_idx, 'Low']   = min(df.at[last_idx, 'Low'], current_low)
                    df.at[last_idx, 'Volume'] = current_vol
                else:
                    new_row = {
                        'Date': today_date, 'Open': current_price, 'High': current_high,
                        'Low': current_low, 'Close': current_price, 'Volume': current_vol
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        except:
            pass
        return df
    except:
        return pd.DataFrame()

def compute_features_inference(df):
    g = df.copy()
    if len(g) < 60: return pd.DataFrame()
    
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: 
        g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
    
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean()
        g[f'Grad_{n}'] = np.gradient(ma.fillna(method='bfill'))
        
    g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
    g['RSI'] = ta.rsi(g['Close'], length=14)
    g['BB_PctB'] = ta.bbands(g['Close'], length=20, std=2).iloc[:, 4]
    g['MACD_Hist'] = ta.macd(g['Close']).iloc[:, 1]
    g['ATR_Rel'] = ta.atr(g['High'], g['Low'], g['Close'], length=14) / g['Close']
    
    rmin = g['Close'].rolling(20).min()
    rmax = g['Close'].rolling(20).max()
    ma20 = g['Close'].rolling(20).mean()
    
    g['Dist_Prev_K10'] = 0.0
    mask_up = g['Close'] >= ma20
    mask_down = g['Close'] < ma20
    g.loc[mask_up, 'Dist_Prev_K10'] = (g['Close'] - rmin) / rmin
    g.loc[mask_down, 'Dist_Prev_K10'] = (g['Close'] - rmax) / rmax
    
    return g.dropna().reset_index(drop=True)

def process_prediction(df_calc, symbol, target_idx=-1):
    if len(df_calc) < 55: return None
    
    if target_idx == -1:
        d50 = df_calc.iloc[-50:].copy()
        d10 = df_calc.iloc[-10:].copy()
        current_date = df_calc.iloc[-1]['Date']
        current_price = df_calc.iloc[-1]['Close']
    else:
        end_pos = target_idx + 1
        d50 = df_calc.iloc[end_pos-50 : end_pos].copy()
        d10 = df_calc.iloc[end_pos-10 : end_pos].copy()
        current_date = df_calc.iloc[target_idx]['Date']
        current_price = df_calc.iloc[target_idx]['Close']

    if len(d50) < 50 or len(d10) < 10: return None

    scaler = local_scalers.get(symbol, global_scaler)
    try:
        s50 = scaler.transform(d50[FEATS_FULL].values)
        s10 = scaler.transform(d10[FEATS_FULL].values)
    except:
        s50 = global_scaler.transform(d50[FEATS_FULL].values)
        s10 = global_scaler.transform(d10[FEATS_FULL].values)

    p50_raw = model_win50.predict(np.expand_dims(s50, axis=0), verbose=0)[0]
    p10_raw = model_win10.predict(np.expand_dims(s10[:, :17], axis=0), verbose=0)[0]

    cls50, cls10 = np.argmax(p50_raw), np.argmax(p10_raw)
    avg_prob = (p50_raw[cls50] + p10_raw[cls10]) / 2

    signal = "THEO DÕI"
    if cls50 == 0 and cls10 == 0: signal = "MUA"
    elif cls50 == 2 and cls10 == 2: signal = "BÁN"

    return {
        'Symbol': symbol,
        'Date': current_date,
        'Close': current_price,
        'Prob': avg_prob,
        'Signal': signal
    }

# --- 3. GIAO DIỆN WEB ---
st.title("🤖 VN30 AI Trading Dashboard")
st.markdown(f"**Cập nhật lúc:** {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")

# Sidebar
st.sidebar.header("Bộ điều khiển")
mode = st.sidebar.radio("Chế độ", ["Quét toàn thị trường", "Soi mã cụ thể"])

if mode == "Quét toàn thị trường":
    if st.sidebar.button("🚀 BẮT ĐẦU QUÉT"):
        st.write("### ⏳ Đang quét tín hiệu Real-time...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        for i, sym in enumerate(VN30_LIST):
            status_text.text(f"Đang phân tích: {sym} ({i+1}/{len(VN30_LIST)})")
            
            df = get_data_efficient(sym)
            if not df.empty:
                df_c = compute_features_inference(df)
                res = process_prediction(df_c, sym)
                if res: results.append(res)
            
            progress_bar.progress((i + 1) / len(VN30_LIST))
        
        status_text.text("✅ Hoàn tất!")
        progress_bar.empty()
        
        # Phân loại
        df_res = pd.DataFrame(results)
        if not df_res.empty:
            df_mua = df_res[df_res['Signal'] == 'MUA'].sort_values('Prob', ascending=False)
            df_ban = df_res[df_res['Signal'] == 'BÁN'].sort_values('Prob', ascending=False)
            df_theo_doi = df_res[df_res['Signal'] == 'THEO DÕI'].sort_values('Prob', ascending=False).head(10)

            col1, col2 = st.columns(2)
            with col1:
                st.success("### 🟢 KHUYẾN NGHỊ MUA")
                if not df_mua.empty:
                    st.dataframe(df_mua[['Symbol', 'Close', 'Prob']].style.format({"Close": "{:,.0f}", "Prob": "{:.1%}"}))
                else:
                    st.write("Chưa có tín hiệu Mua.")

            with col2:
                st.error("### 🔴 KHUYẾN NGHỊ BÁN")
                if not df_ban.empty:
                    st.dataframe(df_ban[['Symbol', 'Close', 'Prob']].style.format({"Close": "{:,.0f}", "Prob": "{:.1%}"}))
                else:
                    st.write("Chưa có tín hiệu Bán.")
            
            st.warning("### 🟡 TOP THEO DÕI (Prob cao nhất)")
            st.dataframe(df_theo_doi[['Symbol', 'Close', 'Prob']].style.format({"Close": "{:,.0f}", "Prob": "{:.1%}"}))

elif mode == "Soi mã cụ thể":
    selected_sym = st.sidebar.selectbox("Chọn mã cổ phiếu", VN30_LIST)
    if st.button(f"🔍 Phân tích {selected_sym}"):
        with st.spinner(f"Đang tải dữ liệu {selected_sym}..."):
            df = get_data_efficient(selected_sym)
            if not df.empty:
                df_c = compute_features_inference(df)
                
                # Hiện giá hiện tại
                curr_price = df_c.iloc[-1]['Close']
                st.metric(label=f"Giá {selected_sym}", value=f"{curr_price:,.0f} VND")
                
                # Vẽ biểu đồ giá nhỏ
                st.line_chart(df_c.set_index('Date')['Close'].tail(50))

                # Dự báo 5 phiên gần nhất
                st.write("### 📅 Lịch sử tín hiệu AI (5 phiên gần nhất)")
                hist_res = []
                for i in range(4, -1, -1):
                    idx = len(df_c) - 1 - i
                    if idx < 0: continue
                    r = process_prediction(df_c, selected_sym, target_idx=idx)
                    if r:
                        hist_res.append({
                            'Ngày': r['Date'].strftime('%d/%m'),
                            'Giá': r['Close'],
                            'Tín hiệu AI': r['Signal'],
                            'Độ tin cậy': r['Prob']
                        })
                
                df_hist = pd.DataFrame(hist_res)
                
                # Tô màu bảng kết quả
                def color_signal(val):
                    color = 'green' if val == 'MUA' else 'red' if val == 'BÁN' else 'orange'
                    return f'color: {color}; font-weight: bold'
                
                st.dataframe(df_hist.style.applymap(color_signal, subset=['Tín hiệu AI'])
                             .format({"Giá": "{:,.0f}", "Độ tin cậy": "{:.1%}"}))
            else:
                st.error("Không lấy được dữ liệu.")
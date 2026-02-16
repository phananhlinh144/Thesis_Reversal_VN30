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
# 1. CẤU HÌNH & CACHE MODEL (Chỉ load 1 lần để nhẹ máy)
# ==============================================================================
st.set_page_config(page_title="VN30 AI Forecast", layout="wide")

# Đường dẫn model
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
# 2. HÀM XỬ LÝ DỮ LIỆU (OPTIMIZED)
# ==============================================================================

def compute_features(df):
    """Tính toán chỉ báo giống hệt Jupyter"""
    g = df.copy()
    if len(g) < 60: return pd.DataFrame()
    
    # Return Change
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: 
        g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
    
    # Gradient
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean()
        g[f'Grad_{n}'] = np.gradient(ma.fillna(method='bfill'))
        
    g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
    g['RSI'] = ta.rsi(g['Close'], length=14)
    
    # Bollinger Bands
    bb = ta.bbands(g['Close'], length=20, std=2)
    g['BB_PctB'] = bb.iloc[:, 4] # %B
    g['BB_Upper'] = bb.iloc[:, 2] # Upper
    g['BB_Lower'] = bb.iloc[:, 0] # Lower
    
    g['MACD_Hist'] = ta.macd(g['Close']).iloc[:, 1]
    g['ATR_Rel'] = ta.atr(g['High'], g['Low'], g['Close'], length=14) / g['Close']
    
    # Dist Prev K10
    rmin = g['Close'].rolling(20).min()
    rmax = g['Close'].rolling(20).max()
    ma20 = g['Close'].rolling(20).mean()
    
    g['Dist_Prev_K10'] = 0.0
    mask_up = g['Close'] >= ma20
    mask_down = g['Close'] < ma20
    g.loc[mask_up, 'Dist_Prev_K10'] = (g['Close'] - rmin) / rmin
    g.loc[mask_down, 'Dist_Prev_K10'] = (g['Close'] - rmax) / rmax
    
    return g.dropna().reset_index(drop=True)

def get_data_for_symbol(symbol, fetch_live=True):
    """
    1. Đọc từ file CSV raw (nhanh).
    2. Nếu cần (fetch_live=True), gọi API lấy 1 dòng giá hiện tại ghép vào đuôi.
    """
    try:
        # 1. Đọc file CSV có sẵn
        full_df = pd.read_csv(CSV_PATH)
        df_hist = full_df[full_df['Symbol'] == symbol].copy()
        
        # Format cột chuẩn
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        df_hist = df_hist.sort_values('Date')
        
        if not fetch_live:
            return df_hist

        # 2. Lấy giá Real-time (Chỉ 1 request nhẹ)
        try:
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            live_df = stock.quote.now()
            
            if not live_df.empty:
                current_price = float(live_df['close'].iloc[0])
                current_vol = float(live_df['volume'].iloc[0])
                current_high = float(live_df['high'].iloc[0])
                current_low = float(live_df['low'].iloc[0])
                
                if current_high == 0: current_high = current_price
                if current_low == 0: current_low = current_price
                
                today = pd.Timestamp(datetime.now().date())
                
                # Logic ghép nối: Nếu ngày cuối trong file < hôm nay thì thêm dòng mới
                if df_hist.empty or df_hist.iloc[-1]['Date'].date() < today.date():
                    new_row = pd.DataFrame([{
                        'Date': today,
                        'Open': current_price,
                        'High': current_high,
                        'Low': current_low,
                        'Close': current_price,
                        'Volume': current_vol,
                        'Symbol': symbol
                    }])
                    df_hist = pd.concat([df_hist, new_row], ignore_index=True)
                else:
                    # Nếu file đã có dòng hôm nay rồi (chạy buổi chiều) thì update giá
                    idx = df_hist.index[-1]
                    df_hist.at[idx, 'Close'] = current_price
                    df_hist.at[idx, 'High'] = max(df_hist.at[idx, 'High'], current_high)
                    df_hist.at[idx, 'Low'] = min(df_hist.at[idx, 'Low'], current_low)
                    df_hist.at[idx, 'Volume'] = current_vol

        except Exception as e:
            pass # Mất mạng hoặc lỗi API thì dùng dữ liệu cũ
            
        return df_hist
    except Exception as e:
        return pd.DataFrame()

def predict_single_row(df_calc, idx_target=-1, symbol=''):
    """Dự báo cho 1 điểm dữ liệu cụ thể"""
    if len(df_calc) < 55: return None
    
    # Lấy window dữ liệu
    end_pos = idx_target + 1 if idx_target != -1 else len(df_calc)
    if end_pos < 50: return None
    
    d50 = df_calc.iloc[end_pos-50 : end_pos]
    d10 = df_calc.iloc[end_pos-10 : end_pos]
    
    current_date = df_calc.iloc[end_pos-1]['Date']
    current_close = df_calc.iloc[end_pos-1]['Close']
    
    # Scaling
    global_scaler = scaler_bundle['global_scaler']
    local_scalers = scaler_bundle['local_scalers_dict']
    scaler = local_scalers.get(symbol, global_scaler)
    
    try:
        s50 = scaler.transform(d50[FEATS_FULL].values)
        s10 = scaler.transform(d10[FEATS_FULL].values)
    except:
        s50 = global_scaler.transform(d50[FEATS_FULL].values)
        s10 = global_scaler.transform(d10[FEATS_FULL].values)

    # Predict
    p50 = model_win50.predict(np.expand_dims(s50, axis=0), verbose=0)[0]
    p10 = model_win10.predict(np.expand_dims(s10[:, :17], axis=0), verbose=0)[0] # Model Win10 chỉ dùng 17 features
    
    cls50 = np.argmax(p50)
    cls10 = np.argmax(p10)
    
    # Logic nhãn
    # 0: Mua, 1: Giữ, 2: Bán (Tùy model training, giả định theo code cũ)
    # Ensemble Logic
    signal = "NGANG"
    if cls50 == 0 and cls10 == 0: signal = "MUA"
    elif cls50 == 2 and cls10 == 2: signal = "BÁN"
    
    return {
        'Date': current_date,
        'Close': current_close,
        'Ensemble': signal, # VIẾT HOA
        'Model_50': f"{['mua', 'ngang', 'bán'][cls50]} ({p50[cls50]:.0%})", # viết thường + %
        'Model_10': f"{['mua', 'ngang', 'bán'][cls10]} ({p10[cls10]:.0%})"  # viết thường + %
    }

# ==============================================================================
# 3. GIAO DIỆN CHÍNH (STREAMLIT)
# ==============================================================================

st.title("🤖 VN30 AI TRADING SYSTEM")

tab1, tab2, tab3 = st.tabs(["📊 DỰ BÁO CHUNG", "📈 BIỂU ĐỒ CHI TIẾT", "📝 DỮ LIỆU CHI TIẾT"])

vn30_list = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
             'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
             'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']

# ------------------------------------------------------------------------------
# TAB 1: DỰ BÁO CHUNG (Quét 30 mã)
# ------------------------------------------------------------------------------
with tab1:
    st.header("Tổng quan thị trường hôm nay")
    st.info("Bấm nút bên dưới để quét toàn bộ 30 mã (Mất khoảng 30s-1p)")
    
    if st.button("🚀 CHẠY DỰ BÁO TOÀN BỘ VN30"):
        progress_bar = st.progress(0)
        report_data = []
        
        for i, sym in enumerate(vn30_list):
            # Lấy data và tính toán
            df = get_data_for_symbol(sym, fetch_live=True) # Có fetch live
            df_c = compute_features(df)
            res = predict_single_row(df_c, idx_target=-1, symbol=sym)
            
            if res:
                report_data.append({
                    'Mã CK': sym,
                    'Giá': f"{int(res['Close']):,}",
                    'Ensemble (AI)': res['Ensemble'],
                    'Model Dài (Win50)': res['Model_50'],
                    'Model Ngắn (Win10)': res['Model_10']
                })
            
            progress_bar.progress((i + 1) / len(vn30_list))
            time.sleep(0.1) # Nghỉ cực ngắn để UI mượt hơn
            
        st.success("Đã quét xong!")
        
        # Phân loại để hiển thị đẹp
        final_df = pd.DataFrame(report_data)
        
        col_mua, col_ban, col_ngang = st.columns(3)
        
        with col_mua:
            st.markdown("### 🟢 KHUYẾN NGHỊ MUA")
            df_mua = final_df[final_df['Ensemble (AI)'] == 'MUA']
            if not df_mua.empty:
                st.dataframe(df_mua, hide_index=True)
            else:
                st.write("Không có mã nào.")

        with col_ban:
            st.markdown("### 🔴 KHUYẾN NGHỊ BÁN")
            df_ban = final_df[final_df['Ensemble (AI)'] == 'BÁN']
            if not df_ban.empty:
                st.dataframe(df_ban, hide_index=True)
            else:
                st.write("Không có mã nào.")

        with col_ngang:
            st.markdown("### 🟡 TRẠNG THÁI NGANG")
            df_ngang = final_df[final_df['Ensemble (AI)'] == 'NGANG']
            if not df_ngang.empty:
                st.dataframe(df_ngang, hide_index=True)
            else:
                st.write("Không có mã nào.")

# ------------------------------------------------------------------------------
# TAB 2: BIỂU ĐỒ (Chỉ load mã được chọn)
# ------------------------------------------------------------------------------
with tab2:
    selected_sym = st.selectbox("Chọn mã cổ phiếu:", vn30_list, key='chart_select')
    
    if selected_sym:
        with st.spinner(f"Đang tải dữ liệu {selected_sym}..."):
            df = get_data_for_symbol(selected_sym, fetch_live=True)
            df_c = compute_features(df)
            
            # Dự báo quá khứ (30 ngày gần nhất để vẽ lên biểu đồ)
            signals = []
            lookback_plot = 60 # Vẽ 60 nến
            start_idx = max(55, len(df_c) - lookback_plot)
            
            for idx in range(start_idx, len(df_c)):
                res = predict_single_row(df_c, idx_target=idx, symbol=selected_sym)
                if res:
                    signals.append(res)
            
            # --- VẼ BIỂU ĐỒ ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3])

            # 1. Giá nến & Bollinger Bands
            plot_df = df_c.iloc[start_idx:]
            
            # BB Band Area
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['BB_Upper'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['BB_Lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)', showlegend=False), row=1, col=1)
            
            # Candlestick
            fig.add_trace(go.Candlestick(x=plot_df['Date'],
                            open=plot_df['Open'], high=plot_df['High'],
                            low=plot_df['Low'], close=plot_df['Close'],
                            name='Giá'), row=1, col=1)

            # 2. Mũi tên dự báo (Ensemble)
            # Lọc các điểm Mua/Bán từ signals
            sig_df = pd.DataFrame(signals)
            if not sig_df.empty:
                buy_pts = sig_df[sig_df['Ensemble'] == 'MUA']
                sell_pts = sig_df[sig_df['Ensemble'] == 'BÁN']
                
                # Mũi tên Trắng (Mua) - Trong Plotly dùng marker tam giác
                fig.add_trace(go.Scatter(x=buy_pts['Date'], y=buy_pts['Close'] * 0.98, 
                                         mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'),
                                         name='AI Báo Mua'), row=1, col=1)
                
                # Mũi tên Đen (Bán)
                fig.add_trace(go.Scatter(x=sell_pts['Date'], y=sell_pts['Close'] * 1.02, 
                                         mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
                                         name='AI Báo Bán'), row=1, col=1)
                
                # Chấm tròn (Dự báo thực tế - Ở đây mình để chấm tròn cho các điểm dự báo bất kể KQ)
                fig.add_trace(go.Scatter(x=sig_df['Date'], y=sig_df['Close'],
                                         mode='markers', marker=dict(size=4, color='black'),
                                         name='Điểm Dự Báo'), row=1, col=1)

            # 3. RSI & Volume
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            fig.update_layout(title=f"Biểu đồ phân tích {selected_sym}", xaxis_rangeslider_visible=False, height=600)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: CHI TIẾT DỰ BÁO (Table)
# ------------------------------------------------------------------------------
with tab3:
    col_sel, col_day = st.columns([1, 2])
    with col_sel:
        sym_tab3 = st.selectbox("Chọn mã:", vn30_list, key='tab3_select')
    with col_day:
        days_back = st.slider("Xem lại bao nhiêu ngày?", 5, 30, 10)
        
    if sym_tab3:
        # Lấy data (Nếu đã load ở tab 2 thì nó có cache nhẹ của streamlit, nếu không thì load mới)
        df = get_data_for_symbol(sym_tab3, fetch_live=True)
        df_c = compute_features(df)
        
        table_data = []
        # Lấy N ngày cuối cùng
        start_idx = max(55, len(df_c) - days_back)
        
        for idx in range(len(df_c)-1, start_idx-1, -1): # Duyệt ngược từ mới nhất về cũ
            res = predict_single_row(df_c, idx_target=idx, symbol=sym_tab3)
            if res:
                table_data.append({
                    'Ngày': res['Date'].strftime('%d-%m-%Y'),
                    'Giá Đóng': f"{int(res['Close']):,}",
                    'ENSEMBLE': res['Ensemble'], # HOA
                    'Win50 (Dài)': res['Model_50'], # thường + %
                    'Win10 (Ngắn)': res['Model_10'] # thường + %
                })
                
        st.write(f"### Chi tiết tín hiệu {sym_tab3} ({days_back} phiên gần nhất)")
        st.dataframe(pd.DataFrame(table_data))


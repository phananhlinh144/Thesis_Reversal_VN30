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
    """Tải model và scaler, chỉ thực hiện 1 lần duy nhất"""
    try:
        m50 = tf.keras.models.load_model(MODEL_WIN50_PATH)
        m10 = tf.keras.models.load_model(MODEL_WIN10_PATH)
        scaler_data = joblib.load(SCALER_PATH)
        return m50, m10, scaler_data
    except Exception as e:
        st.error(f"⚠️ Lỗi nghiêm trọng khi tải Model: {e}")
        return None, None, None

model_win50, model_win10, scaler_bundle = load_ai_system()

# ==============================================================================
# 2. HÀM XỬ LÝ DỮ LIỆU & TÍNH TOÁN CHỈ BÁO
# ==============================================================================

def compute_features(df):
    """Tính toán toán bộ hệ thống chỉ báo kỹ thuật phục vụ AI"""
    g = df.copy()
    if len(g) < 60: 
        return pd.DataFrame()
    
    # --- 1. Return Change (RC) ---
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: 
        g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
    
    # --- 2. Gradient (Sửa lỗi fillna(method='bfill') của Pandas mới) ---
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean()
        # Sử dụng .bfill() thay cho method='bfill' để tránh Future Warning
        g[f'Grad_{n}'] = np.gradient(ma.bfill())
        
    # --- 3. Chỉ báo động lượng & Biến động ---
    g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
    g['RSI'] = ta.rsi(g['Close'], length=14)
    
    # Bollinger Bands
    bb = ta.bbands(g['Close'], length=20, std=2)
    g['BB_PctB'] = bb.iloc[:, 4]  # %B
    g['BB_Upper'] = bb.iloc[:, 2] # Upper
    g['BB_Lower'] = bb.iloc[:, 0] # Lower
    
    # MACD & ATR
    g['MACD_Hist'] = ta.macd(g['Close']).iloc[:, 1]
    g['ATR_Rel'] = ta.atr(g['High'], g['Low'], g['Close'], length=14) / g['Close']
    
    # --- 4. Dist Prev K10 (Logic vùng giá) ---
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
    """Lấy dữ liệu từ CSV và cập nhật giá Real-time từ Vnstock"""
    try:
        # Đọc dữ liệu lịch sử
        full_df = pd.read_csv(CSV_PATH)
        # Sửa lỗi KeyError: Ép tên cột về chữ HOA đầu để đồng nhất
        full_df.columns = [c.capitalize() for c in full_df.columns]
        
        df_hist = full_df[full_df['Symbol'] == symbol].copy()
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        df_hist = df_hist.sort_values('Date')
        
        if not fetch_live:
            return df_hist

        # Lấy giá hiện tại (Real-time)
        try:
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            live_df = stock.quote.now()
            
            if not live_df.empty:
                # Map dữ liệu Vnstock (thường là chữ thường) sang biến
                c_p = float(live_df['close'].iloc[0])
                c_v = float(live_df['volume'].iloc[0])
                c_h = float(live_df['high'].iloc[0])
                c_l = float(live_df['low'].iloc[0])
                
                if c_h == 0: c_h = c_p
                if c_l == 0: c_l = c_p
                
                today = pd.Timestamp(datetime.now().date())
                
                # Nếu chưa có nến ngày hôm nay trong CSV
                if df_hist.empty or df_hist.iloc[-1]['Date'].date() < today.date():
                    new_row = pd.DataFrame([{
                        'Date': today, 'Open': c_p, 'High': c_h, 'Low': c_l,
                        'Close': c_p, 'Volume': c_v, 'Symbol': symbol
                    }])
                    df_hist = pd.concat([df_hist, new_row], ignore_index=True)
                else:
                    # Nếu đã có (đang trong phiên giao dịch), cập nhật nến hiện tại
                    idx = df_hist.index[-1]
                    df_hist.at[idx, 'Close'] = c_p
                    df_hist.at[idx, 'High'] = max(df_hist.at[idx, 'High'], c_h)
                    df_hist.at[idx, 'Low'] = min(df_hist.at[idx, 'Low'], c_l)
                    df_hist.at[idx, 'Volume'] = c_v
        except:
            pass # Lỗi API thì dùng data lịch sử
            
        return df_hist
    except:
        return pd.DataFrame()

def predict_single_row(df_calc, idx_target=-1, symbol=''):
    """Chạy dự báo AI cho một điểm thời gian cụ thể"""
    if len(df_calc) < 55: return None
    
    end_pos = idx_target + 1 if idx_target != -1 else len(df_calc)
    if end_pos < 50: return None
    
    d50 = df_calc.iloc[end_pos-50 : end_pos]
    d10 = df_calc.iloc[end_pos-10 : end_pos]
    
    # Lấy Scaler cho mã tương ứng
    global_scaler = scaler_bundle['global_scaler']
    local_scalers = scaler_bundle['local_scalers_dict']
    scaler = local_scalers.get(symbol, global_scaler)
    
    try:
        s50 = scaler.transform(d50[FEATS_FULL].values)
        s10 = scaler.transform(d10[FEATS_FULL].values)
        
        p50 = model_win50.predict(np.expand_dims(s50, axis=0), verbose=0)[0]
        # Model Win10 chỉ lấy 17 features đầu (bỏ Dist_Prev_K10 nếu cần)
        p10 = model_win10.predict(np.expand_dims(s10[:, :17], axis=0), verbose=0)[0]
        
        cls50, cls10 = np.argmax(p50), np.argmax(p10)
        
        # Ensemble Logic
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
    except:
        return None

# ==============================================================================
# 3. GIAO DIỆN STREAMLIT
# ==============================================================================
vn30_list = [
    'ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE'
]

tab1, tab2, tab3 = st.tabs(["📊 DỰ BÁO CHUNG", "📈 BIỂU ĐỒ CHI TIẾT", "📝 DỮ LIỆU CHI TIẾT"])

# ------------------------------------------------------------------------------
# TAB 1: 3 BẢNG NGANG
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("🔥 Hệ thống quét tín hiệu VN30")
    if st.button("🚀 CHẠY QUÉT TOÀN BỘ THỊ TRƯỜNG"):
        report_data = []
        bar = st.progress(0)
        for i, sym in enumerate(vn30_list):
            df_c = compute_features(get_data_for_symbol(sym))
            res = predict_single_row(df_c, symbol=sym)
            if res:
                report_data.append({
                    'Mã': sym, 'Giá': f"{int(res['Close']):,}",
                    'AI': res['Ensemble'], 'Dài': res['Model_50'], 'Ngắn': res['Model_10']
                })
            bar.progress((i + 1) / len(vn30_list))
        
        f_df = pd.DataFrame(report_data)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("🟢 DANH SÁCH MUA")
            st.table(f_df[f_df['AI'] == 'MUA'][['Mã', 'Giá', 'Dài']])
        with col2:
            st.error("🔴 DANH SÁCH BÁN")
            st.table(f_df[f_df['AI'] == 'BÁN'][['Mã', 'Giá', 'Dài']])
        with col3:
            st.warning("🟡 THEO DÕI (NGANG)")
            st.table(f_df[f_df['AI'] == 'NGANG'][['Mã', 'Giá', 'Dài']])

# ------------------------------------------------------------------------------
# TAB 2: BIỂU ĐỒ CHUYÊN SÂU
# ------------------------------------------------------------------------------
with tab2:
    c_sym, c_type, c_range = st.columns([1.5, 1.5, 2])
    with c_sym: s_sym = st.selectbox("Chọn mã phân tích:", vn30_list, key='s2')
    with c_type: s_type = st.radio("Kiểu giá:", ["Nến Nhật", "Đường"], horizontal=True)
    with c_range: s_lookback = st.slider("Số phiên hiển thị:", 30, 200, 100)
    
    if s_sym:
        df_full = compute_features(get_data_for_symbol(s_sym))
        plot_df = df_full.tail(s_lookback).copy()
        
        # Chạy dự báo cho từng phiên trong plot_df để vẽ mũi tên
        signals = []
        for idx in range(len(df_full) - len(plot_df), len(df_full)):
            r = predict_single_row(df_full, idx_target=idx, symbol=s_sym)
            if r: signals.append(r)
        sig_df = pd.DataFrame(signals)

        # Tạo Subplots: Row 1 (Giá + Vol), Row 2 (RSI)
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, 
            vertical_spacing=0.08, row_heights=[0.75, 0.25],
            subplot_titles=(f"Phân tích Kỹ thuật & AI: {s_sym}", "Chỉ số RSI (14)")
        )

        # 1. Vẽ Giá (Nến hoặc Đường)
        if s_type == "Nến Nhật":
            fig.add_trace(go.Candlestick(
                x=plot_df['Date'], open=plot_df['Open'], high=plot_df['High'],
                low=plot_df['Low'], close=plot_df['Close'], name='Nến'
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Close'], name='Giá Đóng', line=dict(color='#1f77b4')), row=1, col=1)

        # 2. Vẽ Volume (Nằm chung bảng giá nhưng dùng trục phụ hoặc cột)
        fig.add_trace(go.Bar(
            x=plot_df['Date'], y=plot_df['Volume'], name='Khối lượng',
            marker_color='rgba(150, 150, 150, 0.3)', showlegend=False
        ), row=1, col=1)

        # 3. Vẽ Mũi tên AI (Chỉ hiện MUA/BÁN)
        if not sig_df.empty:
            buys = sig_df[sig_df['Ensemble'] == 'MUA']
            sells = sig_df[sig_df['Ensemble'] == 'BÁN']
            fig.add_trace(go.Scatter(
                x=buys['Date'], y=buys['Close'] * 0.97, mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='green', line=dict(width=2, color='white')),
                name='AI BÁO MUA'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=sells['Date'], y=sells['Close'] * 1.03, mode='markers',
                marker=dict(symbol='triangle-down', size=15, color='red', line=dict(width=2, color='white')),
                name='AI BÁO BÁN'
            ), row=1, col=1)

        # 4. Vẽ RSI
        fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['RSI'], name='RSI', line=dict(color='purple', width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: BẢNG DỮ LIỆU & COLOR MAPPING
# ------------------------------------------------------------------------------
with tab3:
    c1, c2 = st.columns([1, 2])
    with c1: s_tab3 = st.selectbox("Chọn mã:", vn30_list, key='s3')
    with c2: s_days = st.slider("Xem lại số phiên:", 5, 50, 15)
    
    if s_tab3:
        df_c = compute_features(get_data_for_symbol(s_tab3))
        res_list = []
        # Chạy ngược từ ngày mới nhất về quá khứ
        for i in range(len(df_c)-1, len(df_c)-1-s_days, -1):
            if i < 50: break
            r = predict_single_row(df_c, idx_target=i, symbol=s_tab3)
            if r:
                res_list.append({
                    'Ngày': r['Date'].strftime('%d/%m/%Y'),
                    'Giá Đóng': f"{r['Close']:.2f}",
                    'Ensemble': r['Ensemble'],
                    'Win50_Dài': r['Model_50'],
                    'Win10_Ngắn': r['Model_10']
                })
        
        df_final = pd.DataFrame(res_list)
        
        # Hàm đổi màu cột Ensemble
        def color_signal(val):
            if val == 'MUA': color = '#d4edda'; txt = 'green' # Xanh
            elif val == 'BÁN': color = '#f8d7da'; txt = 'red' # Đỏ
            else: color = '#fff3cd'; txt = '#856404'         # Vàng
            return f'background-color: {color}; color: {txt}; font-weight: bold'

        st.dataframe(
            df_final.style.map(color_signal, subset=['Ensemble']),
            use_container_width=True, height=500
        )

# ==============================================================================
# KẾT THÚC FILE - ĐẢM BẢO LOGIC KHÔNG BỊ CẮT XÉN
# ==============================================================================

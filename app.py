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
# 1. CẤU HÌNH & CSS & CACHE MODEL
# ==============================================================================
st.set_page_config(page_title="VN30 AI PRO TRADING", layout="wide", page_icon="📈")

# CSS tùy chỉnh để làm đẹp giao diện và bảng
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; gap: 1px;}
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 2px solid #ff4b4b;}
    /* Chỉnh màu header bảng */
    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

# Đường dẫn file (Đảm bảo cấu trúc thư mục đúng trên GitHub/Local)
MODEL_WIN50_PATH = 'models_scaling/Full_K10_Win50_Hybrid.keras'
MODEL_WIN10_PATH = 'models_scaling/Baseline_K10_Win10_Hybrid.keras'
SCALER_PATH      = 'models_scaling/smart_scaler_system.pkl'
CSV_PATH         = 'vn30_data_raw.csv'

# Danh sách Features chuẩn (QUAN TRỌNG: Thứ tự phải đúng như lúc Train)
FINAL_FEATURES = [
    'RC_1', 'RC_2', 'RC_3', 'RC_5', 'RC_8', 'RC_13', 'RC_21', 'RC_34', 'RC_55',
    'Grad_5', 'Grad_10', 'Grad_20', 'RSI', 'BB_PctB', 'MACD_Hist', 'Vol_Ratio', 'ATR_Rel'
]
FEATS_FULL = FINAL_FEATURES + ['Dist_Prev_K10']

@st.cache_resource
def load_ai_system():
    """Load Model Keras và Scaler Joblib một lần duy nhất"""
    try:
        m50 = tf.keras.models.load_model(MODEL_WIN50_PATH)
        m10 = tf.keras.models.load_model(MODEL_WIN10_PATH)
        scaler_data = joblib.load(SCALER_PATH)
        return m50, m10, scaler_data
    except Exception as e:
        st.error(f"❌ Lỗi nghiêm trọng khi tải Model/Scaler: {e}")
        return None, None, None

model_win50, model_win10, scaler_bundle = load_ai_system()

VN30_LIST = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
             'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
             'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']

# ==============================================================================
# 2. HÀM XỬ LÝ DỮ LIỆU & FEATURE ENGINEERING
# ==============================================================================

def compute_features(df):
    """
    Tính toán các chỉ báo kỹ thuật.
    Input: DataFrame (Date, Open, High, Low, Close, Volume)
    Output: DataFrame với các cột Feature đầy đủ, bỏ các dòng NaN đầu tiên.
    """
    if df is None or len(df) < 60: 
        return pd.DataFrame()
    
    g = df.copy()
    
    # 1. Return Change (RC)
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: 
        g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
    
    # 2. Gradient (Đạo hàm xu hướng)
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean()
        # fillna bfill để tránh lỗi NaN ở đầu khi tính gradient
        g[f'Grad_{n}'] = np.gradient(ma.fillna(method='bfill'))
        
    # 3. Volume Ratio
    g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
    
    # 4. RSI
    g['RSI'] = ta.rsi(g['Close'], length=14)
    
    # 5. Bollinger Bands
    bb = ta.bbands(g['Close'], length=20, std=2)
    # pandas_ta trả về tên cột kiểu BBL_20_2.0, BJM... lấy đúng index
    if bb is not None:
        g['BB_PctB'] = bb.iloc[:, 4] # %B
        g['BB_Upper'] = bb.iloc[:, 2] # Upper Band
        g['BB_Lower'] = bb.iloc[:, 0] # Lower Band
    
    # 6. MACD
    macd = ta.macd(g['Close'])
    if macd is not None:
        g['MACD_Hist'] = macd.iloc[:, 1] # Histogram
    
    # 7. ATR Relative
    atr = ta.atr(g['High'], g['Low'], g['Close'], length=14)
    g['ATR_Rel'] = atr / g['Close']
    
    # 8. Distance to Previous K10 (Logic tùy chỉnh)
    # Tính khoảng cách tới Min/Max của 20 phiên trước đó tùy theo vị trí giá so với MA20
    rmin = g['Close'].rolling(20).min()
    rmax = g['Close'].rolling(20).max()
    ma20 = g['Close'].rolling(20).mean()
    
    g['Dist_Prev_K10'] = 0.0
    mask_up = g['Close'] >= ma20
    mask_down = g['Close'] < ma20
    
    # Tránh chia cho 0
    g.loc[mask_up, 'Dist_Prev_K10'] = (g['Close'] - rmin) / (rmin + 1e-9)
    g.loc[mask_down, 'Dist_Prev_K10'] = (g['Close'] - rmax) / (rmax + 1e-9)
    
    # Xóa các dòng NaN do Rolling tạo ra (55 dòng đầu)
    g = g.dropna().reset_index(drop=True)
    return g

def get_data_for_symbol(symbol, fetch_live=True):
    """
    Kết hợp dữ liệu lịch sử từ CSV và dữ liệu Real-time từ Vnstock
    """
    try:
        # 1. Đọc CSV Local
        try:
            full_df = pd.read_csv(CSV_PATH)
            df_hist = full_df[full_df['Symbol'] == symbol].copy()
            df_hist['Date'] = pd.to_datetime(df_hist['Date'])
            df_hist = df_hist.sort_values('Date')
        except:
            df_hist = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'])

        if not fetch_live:
            return df_hist

        # 2. Fetch Live Data
        try:
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            live_df = stock.quote.now()
            
            if not live_df.empty:
                # Parse dữ liệu live
                cur_close = float(live_df['close'].iloc[0])
                cur_vol = float(live_df['volume'].iloc[0])
                cur_high = float(live_df['high'].iloc[0])
                cur_low = float(live_df['low'].iloc[0])
                
                # Fix lỗi High/Low = 0 đầu phiên
                if cur_high == 0: cur_high = cur_close
                if cur_low == 0: cur_low = cur_close
                
                today = pd.Timestamp(datetime.now().date())
                
                # Logic Merge: Nếu ngày cuối trong hist < hôm nay -> Thêm dòng mới
                # Nếu ngày cuối == hôm nay -> Update giá
                if df_hist.empty or df_hist.iloc[-1]['Date'].date() < today.date():
                    new_row = pd.DataFrame([{
                        'Date': today,
                        'Open': cur_close, # Giả định Open = Close hiện tại nếu mới mở
                        'High': cur_high,
                        'Low': cur_low,
                        'Close': cur_close,
                        'Volume': cur_vol,
                        'Symbol': symbol
                    }])
                    df_hist = pd.concat([df_hist, new_row], ignore_index=True)
                else:
                    idx = df_hist.index[-1]
                    df_hist.at[idx, 'Close'] = cur_close
                    df_hist.at[idx, 'High'] = max(df_hist.at[idx, 'High'], cur_high)
                    df_hist.at[idx, 'Low'] = min(df_hist.at[idx, 'Low'], cur_low)
                    df_hist.at[idx, 'Volume'] = cur_vol
                    
        except Exception as e:
            # Nếu lỗi mạng, vẫn trả về dữ liệu lịch sử
            pass
            
        return df_hist.reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame()

# ==============================================================================
# 3. HÀM DỰ BÁO (CORE AI) - Đã fix lỗi Warning sklearn
# ==============================================================================

def predict_single_row(df_calc, idx_target=-1, symbol=''):
    """
    Dự báo cho 1 dòng dữ liệu (tại idx_target).
    """
    # Kiểm tra độ dài dữ liệu
    if len(df_calc) < 60: return None
    
    # Xác định vị trí cắt dữ liệu
    # Nếu idx_target = -1 (mới nhất), end_pos = len(df).
    # Nếu idx_target = 100, end_pos = 101 (để iloc lấy đến 100)
    end_pos = idx_target + 1 if idx_target != -1 else len(df_calc)
    
    # Đảm bảo đủ 50 phiên quá khứ
    if end_pos < 50: return None
    
    # Cắt Window dữ liệu
    # LƯU Ý: Giữ nguyên dạng DataFrame để Scaler nhận diện tên cột -> Fix lỗi Warning
    d50_df = df_calc.iloc[end_pos-50 : end_pos][FEATS_FULL] 
    d10_df = df_calc.iloc[end_pos-10 : end_pos][FEATS_FULL]
    
    current_info = df_calc.iloc[end_pos-1]
    
    # Lấy Scaler phù hợp
    scaler = scaler_bundle['local_scalers_dict'].get(symbol, scaler_bundle['global_scaler'])
    
    # Transform (Truyền DataFrame vào transform)
    try:
        s50 = scaler.transform(d50_df) # Input là DataFrame -> OK
        s10 = scaler.transform(d10_df)
    except Exception:
        # Fallback nếu scaler lỗi (hiếm gặp nếu đúng tên cột)
        s50 = scaler_bundle['global_scaler'].transform(d50_df)
        s10 = scaler_bundle['global_scaler'].transform(d10_df)
        
    # Predict (Input của Model Keras là Numpy Array 3D: [batch, timesteps, features])
    # Expand dims từ (50, 18) -> (1, 50, 18)
    p50 = model_win50.predict(np.expand_dims(s50, axis=0), verbose=0)[0]
    
    # Model Win10 chỉ dùng 17 features đầu (bỏ Dist_Prev_K10) -> check lại lúc train
    # Giả sử model Win10 train với 17 features:
    p10 = model_win10.predict(np.expand_dims(s10[:, :17], axis=0), verbose=0)[0]
    
    # Lấy class có xác suất cao nhất
    c50 = np.argmax(p50)
    c10 = np.argmax(p10)
    
    # Logic Ensemble (Kết hợp)
    # 0: Mua, 1: Ngang, 2: Bán
    signal = "NGANG"
    if c50 == 0 and c10 == 0: signal = "MUA"
    elif c50 == 2 and c10 == 2: signal = "BÁN"
    
    return {
        'Date': current_info['Date'],
        'Close': current_info['Close'],
        'High': current_info['High'], 
        'Low': current_info['Low'],
        'Open': current_info['Open'],
        'Volume': current_info['Volume'],
        'RSI': current_info['RSI'],
        'BB_Upper': current_info['BB_Upper'],
        'BB_Lower': current_info['BB_Lower'],
        'Ensemble': signal,
        'Raw_50': c50, 'Prob_50': p50[c50],
        'Raw_10': c10, 'Prob_10': p10[c10]
    }

# ==============================================================================
# 4. GIAO DIỆN CHÍNH (STREAMLIT UI)
# ==============================================================================

st.title("🤖 VN30 AI QUANT SYSTEM")

# Tạo Tabs
tab1, tab2, tab3 = st.tabs(["📊 DỰ BÁO TOÀN THỊ TRƯỜNG", "📈 BIỂU ĐỒ CHUYÊN SÂU", "📝 LỊCH SỬ TÍN HIỆU"])

# --- TAB 1: DASHBOARD TỔNG HỢP ---
with tab1:
    st.subheader("Quét tín hiệu VN30 Real-time")
    
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_scan = st.button("🚀 BẮT ĐẦU QUÉT", type="primary")
    
    if run_scan:
        results = []
        progress_bar = st.progress(0)
        status_txt = st.empty()
        
        start_time = time.time()
        
        for i, sym in enumerate(VN30_LIST):
            status_txt.text(f"Đang xử lý {sym} ({i+1}/30)...")
            
            # Lấy data và predict dòng cuối cùng
            df = get_data_for_symbol(sym, fetch_live=True)
            df_c = compute_features(df)
            res = predict_single_row(df_c, idx_target=-1, symbol=sym)
            
            if res:
                # Format dữ liệu để hiển thị
                lbl_map = {0: 'mua', 1: 'ngang', 2: 'bán'}
                results.append({
                    "Mã": sym,
                    "Giá": f"{res['Close']:,.0f}",
                    "Tín hiệu": res['Ensemble'],
                    "W50": f"{lbl_map[res['Raw_50']]} {res['Prob_50']:.0%}",
                    "W10": f"{lbl_map[res['Raw_10']]} {res['Prob_10']:.0%}"
                })
            
            progress_bar.progress((i+1)/30)
            time.sleep(0.05) # Delay nhẹ tránh nghẽn UI
            
        progress_bar.empty()
        status_txt.success(f"Hoàn thành trong {time.time() - start_time:.2f}s")
        
        # Chia 3 cột hiển thị
        df_res = pd.DataFrame(results)
        
        if not df_res.empty:
            c_buy, c_sell, c_hold = st.columns(3)
            
            with c_buy:
                st.markdown("### 🟢 KHUYẾN NGHỊ MUA")
                df_buy = df_res[df_res['Tín hiệu'] == 'MUA'][['Mã', 'Giá', 'W50', 'W10']]
                if not df_buy.empty:
                    st.dataframe(df_buy, hide_index=True, use_container_width=True)
                else:
                    st.info("Không có mã mua.")
            
            with c_sell:
                st.markdown("### 🔴 KHUYẾN NGHỊ BÁN")
                df_sell = df_res[df_res['Tín hiệu'] == 'BÁN'][['Mã', 'Giá', 'W50', 'W10']]
                if not df_sell.empty:
                    st.dataframe(df_sell, hide_index=True, use_container_width=True)
                else:
                    st.info("Không có mã bán.")
            
            with c_hold:
                st.markdown("### 🟡 TRẠNG THÁI NGANG")
                df_hold = df_res[df_res['Tín hiệu'] == 'NGANG'][['Mã', 'Giá', 'W50', 'W10']]
                if not df_hold.empty:
                    st.dataframe(df_hold, hide_index=True, use_container_width=True)
                else:
                    st.info("Không có mã ngang.")

# --- TAB 2: BIỂU ĐỒ PHÂN TÍCH ---
with tab2:
    # 1. Controls
    c_sel1, c_sel2, c_sel3, c_sel4 = st.columns([1, 1, 1, 1])
    with c_sel1:
        selected_sym = st.selectbox("Chọn mã:", VN30_LIST, key='chart_sym')
    with c_sel2:
        start_date = st.date_input("Từ ngày:", datetime.now() - timedelta(days=90))
    with c_sel3:
        end_date = st.date_input("Đến ngày:", datetime.now())
    with c_sel4:
        chart_type = st.radio("Kiểu biểu đồ:", ["Nến (Candle)", "Đường (Line)"], horizontal=True)

    if st.button("Vẽ biểu đồ"):
        with st.spinner(f"Đang phân tích {selected_sym}..."):
            # Lấy data
            df = get_data_for_symbol(selected_sym, fetch_live=True)
            df_c = compute_features(df)
            
            # Lọc theo ngày
            mask = (df_c['Date'].dt.date >= start_date) & (df_c['Date'].dt.date <= end_date)
            df_plot = df_c.loc[mask].copy()
            
            if len(df_plot) > 10:
                # Chạy dự báo lại cho khoảng thời gian này để lấy tín hiệu vẽ
                # Lưu ý: Cần loop qua từng điểm trong df_plot để predict (mô phỏng quá khứ)
                
                # Tìm index tương ứng trong df gốc
                indices = df_plot.index
                signals_data = []
                
                for idx in indices:
                    # Chỉ predict nếu đủ dữ liệu quá khứ (idx >= 55)
                    pred = predict_single_row(df_c, idx_target=idx, symbol=selected_sym)
                    if pred:
                        signals_data.append(pred)
                
                df_sigs = pd.DataFrame(signals_data)
                
                # --- VẼ PLOTLY ---
                # Tạo subplot: Row 1 (Giá + Vol), Row 2 (RSI)
                fig = make_subplots(
                    rows=2, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.05, 
                    row_heights=[0.75, 0.25],
                    specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
                )
                
                # 1. BIỂU ĐỒ GIÁ (Row 1 - Primary Y)
                if "Nến" in chart_type:
                    fig.add_trace(go.Candlestick(
                        x=df_plot['Date'], open=df_plot['Open'], high=df_plot['High'],
                        low=df_plot['Low'], close=df_plot['Close'], name='Giá'
                    ), row=1, col=1, secondary_y=False)
                else:
                    fig.add_trace(go.Scatter(
                        x=df_plot['Date'], y=df_plot['Close'], mode='lines', 
                        line=dict(color='blue', width=2), name='Giá Đóng'
                    ), row=1, col=1, secondary_y=False)

                # BB Bands
                fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['BB_Upper'], line=dict(color='gray', width=1), name='BB Upper', showlegend=False), row=1, col=1, secondary_y=False)
                fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['BB_Lower'], line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(200,200,200,0.1)', name='BB Lower', showlegend=False), row=1, col=1, secondary_y=False)

                # 2. VOLUME (Row 1 - Secondary Y - chung bảng)
                # Tô màu volume xanh/đỏ
                colors_vol = ['green' if c >= o else 'red' for c, o in zip(df_plot['Close'], df_plot['Open'])]
                fig.add_trace(go.Bar(
                    x=df_plot['Date'], y=df_plot['Volume'], 
                    marker_color=colors_vol, opacity=0.3, name='Volume'
                ), row=1, col=1, secondary_y=True)

                # 3. RSI (Row 2)
                fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                # 4. TÍN HIỆU AI (Mũi tên) & PAGAN CIRCLES (Chấm rỗng)
                if not df_sigs.empty:
                    # Lọc tín hiệu MUA
                    buys = df_sigs[df_sigs['Ensemble'] == 'MUA']
                    fig.add_trace(go.Scatter(
                        x=buys['Date'], y=buys['Low']*0.99, 
                        mode='markers', marker=dict(symbol='arrow-up', size=12, color='green'),
                        name='AI Mua'
                    ), row=1, col=1, secondary_y=False)
                    
                    # Lọc tín hiệu BÁN
                    sells = df_sigs[df_sigs['Ensemble'] == 'BÁN']
                    fig.add_trace(go.Scatter(
                        x=sells['Date'], y=sells['High']*1.01, 
                        mode='markers', marker=dict(symbol='arrow-down', size=12, color='red'),
                        name='AI Bán'
                    ), row=1, col=1, secondary_y=False)
                    
                    # PAGAN CIRCLES: Chấm tròn rỗng tại các điểm dự báo
                    # Đại diện cho vị trí "Look Back / Look Forward"
                    fig.add_trace(go.Scatter(
                        x=df_sigs['Date'], y=df_sigs['Close'],
                        mode='markers', 
                        marker=dict(symbol='circle-open', size=6, color='black', line=dict(width=1)),
                        name='Điểm Dự Báo'
                    ), row=1, col=1, secondary_y=False)

                # Layout Tinh chỉnh
                fig.update_layout(
                    height=700, 
                    title=f"Biểu đồ kỹ thuật & Tín hiệu AI: {selected_sym}",
                    xaxis_rangeslider_visible=False,
                    yaxis2=dict(showgrid=False, overlaying='y', side='right', range=[0, df_plot['Volume'].max()*4]), # Vol thấp xuống dưới
                    legend=dict(orientation="h", y=1.02)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Dữ liệu trong khoảng thời gian này quá ít hoặc không đủ 60 phiên để tính toán.")

# --- TAB 3: CHI TIẾT LỊCH SỬ ---
with tab3:
    c_list, c_slider = st.columns([1, 2])
    with c_list:
        sym_t3 = st.selectbox("Chọn mã xem lịch sử:", VN30_LIST, key='hist_sym')
    with c_slider:
        days_back = st.slider("Số phiên nhìn lại:", 5, 60, 20)
        
    if sym_t3:
        # Lấy data
        df = get_data_for_symbol(sym_t3, fetch_live=True)
        df_c = compute_features(df)
        
        hist_data = []
        # Loop ngược từ ngày mới nhất về quá khứ
        loop_range = range(len(df_c)-1, max(54, len(df_c)-days_back-1), -1)
        
        for idx in loop_range:
            res = predict_single_row(df_c, idx_target=idx, symbol=sym_t3)
            if res:
                lbl_map = {0: 'mua', 1: 'ngang', 2: 'bán'}
                hist_data.append({
                    "Ngày": res['Date'].strftime('%d/%m/%Y'),
                    "Giá Đóng": f"{res['Close']:,.0f}",
                    "ENSEMBLE": res['Ensemble'],
                    "Win50 (Dài)": f"{lbl_map[res['Raw_50']]} ({res['Prob_50']:.0%})",
                    "Win10 (Ngắn)": f"{lbl_map[res['Raw_10']]} ({res['Prob_10']:.0%})"
                })
        
        df_hist_show = pd.DataFrame(hist_data)
        
        # Hàm tô màu cho Pandas Styler
        def color_ensemble_text(val):
            color = 'black' # Mặc định
            if val == 'MUA': color = '#28a745' # Xanh lá
            elif val == 'BÁN': color = '#dc3545' # Đỏ
            elif val == 'NGANG': color = '#ffc107' # Vàng cam
            return f'color: {color}; font-weight: bold'

        if not df_hist_show.empty:
            st.write(f"### Lịch sử tín hiệu {sym_t3}")
            # Áp dụng màu sắc
            st.dataframe(
                df_hist_show.style.map(color_ensemble_text, subset=['ENSEMBLE']),
                use_container_width=True,
                height=500
            )
        else:
            st.info("Chưa đủ dữ liệu lịch sử để hiển thị.")

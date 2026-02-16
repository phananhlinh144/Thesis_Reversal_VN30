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

# CSS tùy chỉnh
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; gap: 1px;}
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 2px solid #ff4b4b;}
    /* Ẩn index của bảng */
    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

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
    if df is None or len(df) < 60: 
        return pd.DataFrame()
    
    g = df.copy()
    
    # 1. Return Change
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: 
        g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
    
    # 2. Gradient (Fix lỗi Deprecated fillna)
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean()
        # Thay thế fillna(method='bfill') bằng bfill()
        g[f'Grad_{n}'] = np.gradient(ma.bfill())
        
    g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
    g['RSI'] = ta.rsi(g['Close'], length=14)
    
    bb = ta.bbands(g['Close'], length=20, std=2)
    if bb is not None:
        g['BB_PctB'] = bb.iloc[:, 4]
        g['BB_Upper'] = bb.iloc[:, 2]
        g['BB_Lower'] = bb.iloc[:, 0]
    
    macd = ta.macd(g['Close'])
    if macd is not None:
        g['MACD_Hist'] = macd.iloc[:, 1]
    
    atr = ta.atr(g['High'], g['Low'], g['Close'], length=14)
    g['ATR_Rel'] = atr / g['Close']
    
    # 3. Distance to Previous K10
    rmin = g['Close'].rolling(20).min()
    rmax = g['Close'].rolling(20).max()
    ma20 = g['Close'].rolling(20).mean()
    
    g['Dist_Prev_K10'] = 0.0
    mask_up = g['Close'] >= ma20
    mask_down = g['Close'] < ma20
    
    g.loc[mask_up, 'Dist_Prev_K10'] = (g['Close'] - rmin) / (rmin + 1e-9)
    g.loc[mask_down, 'Dist_Prev_K10'] = (g['Close'] - rmax) / (rmax + 1e-9)
    
    # 4. TÍNH TOÁN LABEL PAGAN K10 (REALITY CHECK)
    # Nhìn trước 10 ngày và sau 10 ngày (Window = 21, Center = True)
    # Chỉ tính được cho quá khứ (cách hiện tại ít nhất 10 ngày)
    g['Rolling_Min_10'] = g['Close'].rolling(window=21, center=True).min()
    g['Rolling_Max_10'] = g['Close'].rolling(window=21, center=True).max()
    
    g['Is_Pagan_Bottom'] = (g['Close'] == g['Rolling_Min_10'])
    g['Is_Pagan_Top'] = (g['Close'] == g['Rolling_Max_10'])

    g = g.dropna(subset=FINAL_FEATURES).reset_index(drop=True)
    return g

def get_data_for_symbol(symbol, fetch_live=True):
    try:
        try:
            full_df = pd.read_csv(CSV_PATH)
            df_hist = full_df[full_df['Symbol'] == symbol].copy()
            df_hist['Date'] = pd.to_datetime(df_hist['Date'])
            df_hist = df_hist.sort_values('Date')
        except:
            df_hist = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'])

        if not fetch_live:
            return df_hist

        try:
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            live_df = stock.quote.now()
            
            if not live_df.empty:
                cur_close = float(live_df['close'].iloc[0])
                cur_vol = float(live_df['volume'].iloc[0])
                cur_high = float(live_df['high'].iloc[0])
                cur_low = float(live_df['low'].iloc[0])
                
                if cur_high == 0: cur_high = cur_close
                if cur_low == 0: cur_low = cur_close
                
                today = pd.Timestamp(datetime.now().date())
                
                if df_hist.empty or df_hist.iloc[-1]['Date'].date() < today.date():
                    new_row = pd.DataFrame([{
                        'Date': today, 'Open': cur_close, 'High': cur_high,
                        'Low': cur_low, 'Close': cur_close, 'Volume': cur_vol, 'Symbol': symbol
                    }])
                    df_hist = pd.concat([df_hist, new_row], ignore_index=True)
                else:
                    idx = df_hist.index[-1]
                    df_hist.at[idx, 'Close'] = cur_close
                    df_hist.at[idx, 'High'] = max(df_hist.at[idx, 'High'], cur_high)
                    df_hist.at[idx, 'Low'] = min(df_hist.at[idx, 'Low'], cur_low)
                    df_hist.at[idx, 'Volume'] = cur_vol
                    
        except Exception:
            pass
            
        return df_hist.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

# ==============================================================================
# 3. HÀM DỰ BÁO (CORE AI)
# ==============================================================================

def predict_single_row(df_calc, idx_target=-1, symbol=''):
    if len(df_calc) < 60: return None
    end_pos = idx_target + 1 if idx_target != -1 else len(df_calc)
    if end_pos < 50: return None
    
    d50_df = df_calc.iloc[end_pos-50 : end_pos][FEATS_FULL] 
    d10_df = df_calc.iloc[end_pos-10 : end_pos][FEATS_FULL]
    current_info = df_calc.iloc[end_pos-1]
    
    scaler = scaler_bundle['local_scalers_dict'].get(symbol, scaler_bundle['global_scaler'])
    
    try:
        s50 = scaler.transform(d50_df)
        s10 = scaler.transform(d10_df)
    except Exception:
        s50 = scaler_bundle['global_scaler'].transform(d50_df)
        s10 = scaler_bundle['global_scaler'].transform(d10_df)
        
    p50 = model_win50.predict(np.expand_dims(s50, axis=0), verbose=0)[0]
    p10 = model_win10.predict(np.expand_dims(s10[:, :17], axis=0), verbose=0)[0]
    
    c50 = np.argmax(p50)
    c10 = np.argmax(p10)
    
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
        # Thông tin Pagan
        'Is_Pagan_Top': current_info.get('Is_Pagan_Top', False),
        'Is_Pagan_Bottom': current_info.get('Is_Pagan_Bottom', False),
        'Ensemble': signal,
        'Raw_50': c50, 'Prob_50': p50[c50],
        'Raw_10': c10, 'Prob_10': p10[c10]
    }

# ==============================================================================
# 4. GIAO DIỆN CHÍNH
# ==============================================================================

st.title("🤖 VN30 AI QUANT SYSTEM")
tab1, tab2, tab3 = st.tabs(["📊 TÍN HIỆU THỊ TRƯỜNG", "📈 BIỂU ĐỒ CHUYÊN SÂU", "📝 LỊCH SỬ"])

# --- TAB 1: HIỂN THỊ DẠNG BẢNG NGANG (THEO DÒNG) ---
with tab1:
    st.subheader("Quét tín hiệu VN30 Real-time")
    if st.button("🚀 BẮT ĐẦU QUÉT", type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_txt = st.empty()
        
        for i, sym in enumerate(VN30_LIST):
            status_txt.text(f"Đang xử lý {sym} ({i+1}/30)...")
            df = get_data_for_symbol(sym, fetch_live=True)
            df_c = compute_features(df)
            res = predict_single_row(df_c, idx_target=-1, symbol=sym)
            
            if res:
                lbl_map = {0: 'mua', 1: 'ngang', 2: 'bán'}
                results.append({
                    "Mã": sym, "Giá": f"{res['Close']:,.0f}",
                    "Tín hiệu": res['Ensemble'],
                    "W50 (Trend)": f"{lbl_map[res['Raw_50']]} {res['Prob_50']:.0%}",
                    "W10 (Momentum)": f"{lbl_map[res['Raw_10']]} {res['Prob_10']:.0%}"
                })
            progress_bar.progress((i+1)/30)
            time.sleep(0.02)
        
        progress_bar.empty()
        status_txt.success("Hoàn thành!")
        
        df_res = pd.DataFrame(results)
        
        if not df_res.empty:
            # Thay vì chia cột, ta in từng bảng theo chiều dọc
            
            # 1. BẢNG MUA
            st.markdown("### 🟢 KHUYẾN NGHỊ MUA")
            df_buy = df_res[df_res['Tín hiệu'] == 'MUA']
            if not df_buy.empty:
                st.dataframe(df_buy, hide_index=True, width='stretch')
            else:
                st.caption("Không có mã khuyến nghị Mua.")
                
            st.divider() # Đường kẻ ngang phân cách
            
            # 2. BẢNG BÁN
            st.markdown("### 🔴 KHUYẾN NGHỊ BÁN")
            df_sell = df_res[df_res['Tín hiệu'] == 'BÁN']
            if not df_sell.empty:
                st.dataframe(df_sell, hide_index=True, width='stretch')
            else:
                st.caption("Không có mã khuyến nghị Bán.")
                
            st.divider()
            
            # 3. BẢNG NGANG
            st.markdown("### 🟡 TRẠNG THÁI SIDEWAY/NGANG")
            df_hold = df_res[df_res['Tín hiệu'] == 'NGANG']
            if not df_hold.empty:
                st.dataframe(df_hold, hide_index=True, width='stretch')
            else:
                st.caption("Không có mã đi ngang.")

# --- TAB 2: BIỂU ĐỒ & PAGAN CIRCLES ---
with tab2:
    # --- THÊM CỘT 4 ĐỂ CHỌN LOẠI BIỂU ĐỒ ---
    c_sel1, c_sel2, c_sel3, c_sel4 = st.columns([1, 1, 1, 1])
    with c_sel1: selected_sym = st.selectbox("Chọn mã:", VN30_LIST, key='chart_sym')
    with c_sel2: start_date = st.date_input("Từ ngày:", datetime.now() - timedelta(days=120))
    with c_sel3: end_date = st.date_input("Đến ngày:", datetime.now())
    with c_sel4: chart_type = st.selectbox("Loại biểu đồ:", ["Nến (Candles)", "Đường (Line)"], key='chart_type')

    if st.button("Vẽ biểu đồ phân tích"):
        with st.spinner(f"Đang xử lý {selected_sym}..."):
            df = get_data_for_symbol(selected_sym, fetch_live=True)
            df_c = compute_features(df)
            
            mask = (df_c['Date'].dt.date >= start_date) & (df_c['Date'].dt.date <= end_date)
            df_plot = df_c.loc[mask].copy()
            
            # --- FIX CHART GAPS: CHUYỂN NGÀY THÀNH STRING (CATEGORY) ---
            # Plotly sẽ vẽ các chuỗi này liên tiếp nhau, không quan tâm khoảng cách thời gian -> Xóa Gap
            df_plot['Date_Str'] = df_plot['Date'].dt.strftime('%d/%m')
            
            if len(df_plot) > 10:
                indices = df_plot.index
                signals_data = []
                for idx in indices:
                    pred = predict_single_row(df_c, idx_target=idx, symbol=selected_sym)
                    if pred: signals_data.append(pred)
                
                df_sigs = pd.DataFrame(signals_data)
                # Map lại Date_Str cho tín hiệu để khớp với trục X biểu đồ
                df_sigs['Date_Str'] = df_sigs['Date'].dt.strftime('%d/%m')

                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                    row_heights=[0.75, 0.25], specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
                )
                
                # 1. BIỂU ĐỒ GIÁ (NẾN HOẶC ĐƯỜNG) - XỬ LÝ LOGIC TẠI ĐÂY
                if chart_type == "Nến (Candles)":
                    fig.add_trace(go.Candlestick(
                        x=df_plot['Date_Str'], open=df_plot['Open'], high=df_plot['High'],
                        low=df_plot['Low'], close=df_plot['Close'], name='Giá'
                    ), row=1, col=1, secondary_y=False)
                else:
                    fig.add_trace(go.Scatter(
                        x=df_plot['Date_Str'], y=df_plot['Close'], 
                        mode='lines', line=dict(color='#00F0FF', width=2), name='Giá (Line)'
                    ), row=1, col=1, secondary_y=False)

                # BB Bands (GIỮ NGUYÊN)
                fig.add_trace(go.Scatter(x=df_plot['Date_Str'], y=df_plot['BB_Upper'], line=dict(color='gray', width=1), name='BB Upper', showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_plot['Date_Str'], y=df_plot['BB_Lower'], line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(200,200,200,0.1)', name='BB Lower', showlegend=False), row=1, col=1)

                # Volume (GIỮ NGUYÊN)
                colors_vol = ['green' if c >= o else 'red' for c, o in zip(df_plot['Close'], df_plot['Open'])]
                fig.add_trace(go.Bar(x=df_plot['Date_Str'], y=df_plot['Volume'], marker_color=colors_vol, opacity=0.3, name='Volume'), row=1, col=1, secondary_y=True)

                # RSI (GIỮ NGUYÊN)
                fig.add_trace(go.Scatter(x=df_plot['Date_Str'], y=df_plot['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                if not df_sigs.empty:
                    # AI Signals (GIỮ NGUYÊN)
                    buys = df_sigs[df_sigs['Ensemble'] == 'MUA']
                    sells = df_sigs[df_sigs['Ensemble'] == 'BÁN']
                    
                    # Logic vẽ marker vẫn hoạt động tốt trên cả Nến và Đường vì dùng chung trục X (Date_Str)
                    fig.add_trace(go.Scatter(
                        x=buys['Date_Str'], y=buys['Low']*0.99, mode='markers', 
                        marker=dict(symbol='triangle-up', size=10, color='lime'), name='AI Mua'
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=sells['Date_Str'], y=sells['High']*1.01, mode='markers', 
                        marker=dict(symbol='triangle-down', size=10, color='red'), name='AI Bán'
                    ), row=1, col=1)

                    # --- PAGAN K10 CIRCLES (GIỮ NGUYÊN) ---
                    pagan_tops = df_sigs[df_sigs['Is_Pagan_Top'] == True]
                    pagan_bots = df_sigs[df_sigs['Is_Pagan_Bottom'] == True]

                    fig.add_trace(go.Scatter(
                        x=pagan_tops['Date_Str'], y=pagan_tops['High']*1.005,
                        mode='markers',
                        marker=dict(symbol='circle-open', size=20, color='black', line=dict(width=3)),
                        name='Đỉnh (K10)'
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(
                        x=pagan_bots['Date_Str'], y=pagan_bots['Low']*0.995,
                        mode='markers',
                        marker=dict(symbol='circle-open', size=20, color='blue', line=dict(width=3)),
                        name='Đáy (K10)'
                    ), row=1, col=1)

                fig.update_layout(
                    height=700, 
                    title=f"{selected_sym} - AI Prediction & Pagan Labels",
                    xaxis_rangeslider_visible=False,
                    xaxis=dict(type='category'), # Bắt buộc để xóa gap
                    yaxis2=dict(showgrid=False, overlaying='y', side='right', range=[0, df_plot['Volume'].max()*4]),
                    legend=dict(orientation="h", y=1.02)
                )
                
                st.plotly_chart(fig, width='stretch')
            else:
                st.warning("Dữ liệu không đủ để vẽ.")

# --- TAB 3: HISTORY ---
with tab3:
    c_list, c_slider = st.columns([1, 2])
    with c_list: sym_t3 = st.selectbox("Mã:", VN30_LIST, key='hist_sym')
    with c_slider: days_back = st.slider("Số phiên:", 5, 60, 20)
    
    if sym_t3:
        df = get_data_for_symbol(sym_t3, fetch_live=True)
        df_c = compute_features(df)
        hist_data = []
        loop_range = range(len(df_c)-1, max(54, len(df_c)-days_back-1), -1)
        
        for idx in loop_range:
            res = predict_single_row(df_c, idx_target=idx, symbol=sym_t3)
            if res:
                lbl_map = {0: 'mua', 1: 'ngang', 2: 'bán'}
                hist_data.append({
                    "Ngày": res['Date'].strftime('%d/%m/%Y'),
                    "Giá": f"{res['Close']:,.0f}",
                    "Tín hiệu AI": res['Ensemble'],
                    "Win50": f"{lbl_map[res['Raw_50']]} ({res['Prob_50']:.0%})",
                    "Win10": f"{lbl_map[res['Raw_10']]} ({res['Prob_10']:.0%})"
                })
        
        df_hist_show = pd.DataFrame(hist_data)
        def color_signal(val):
            if val == 'MUA': return 'color: #28a745; font-weight: bold'
            if val == 'BÁN': return 'color: #dc3545; font-weight: bold'
            return 'color: #ffc107'

        if not df_hist_show.empty:
            st.dataframe(df_hist_show.style.map(color_signal, subset=['Tín hiệu AI']), width='stretch')

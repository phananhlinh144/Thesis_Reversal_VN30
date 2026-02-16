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

# Tắt warning
warnings.filterwarnings('ignore')

# --- 1. CẤU HÌNH & LOAD MODEL ---
st.set_page_config(page_title="VN30 AI TRADING", layout="wide", page_icon="📈")

# CSS tùy chỉnh để làm đẹp bảng
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; gap: 1px;}
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 2px solid #ff4b4b;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ai_system():
    # Load model và scaler (đảm bảo file nằm cùng thư mục với app.py)
    try:
        m50 = tf.keras.models.load_model('Full_K10_Win50_Hybrid.keras')
        m10 = tf.keras.models.load_model('Baseline_K10_Win10_Hybrid.keras')
        bundle = joblib.load('smart_scaler_system.pkl')
        return m50, m10, bundle
    except Exception as e:
        st.error(f"❌ Lỗi load model/scaler: {e}. Hãy kiểm tra lại file .keras và .pkl")
        return None, None, None

m50, m10, bundle = load_ai_system()

VN30_LIST = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
             'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 
             'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']

FEATS_BASE = ['RC_1', 'RC_2', 'RC_3', 'RC_5', 'RC_8', 'RC_13', 'RC_21', 'RC_34', 'RC_55', 
              'Grad_5', 'Grad_10', 'Grad_20', 'RSI', 'BB_PctB', 'MACD_Hist', 'Vol_Ratio', 'ATR_Rel']
FEATS_FULL = FEATS_BASE + ['Dist_Prev_K10']

# --- 2. HÀM XỬ LÝ DỮ LIỆU ---

def get_data_dnse(symbol):
    """Lấy dữ liệu từ DNSE cho 1 mã"""
    try:
        stock = Vnstock().stock(symbol=symbol, source='DNSE')
        # Lấy dư ra 365 ngày để tính chỉ báo cho mượt
        df = stock.quote.history(start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), 
                                 end=datetime.now().strftime('%Y-%m-%d'))
        if df is None or df.empty: return pd.DataFrame()
        
        # Chuẩn hóa tên cột
        df = df.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date').reset_index(drop=True)
    except:
        return pd.DataFrame()

def compute_features(df):
    """Tính toán chỉ báo kỹ thuật giống Jupyter"""
    if len(df) < 60: return pd.DataFrame()
    g = df.copy()
    
    # Rate of Change
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: 
        g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
        
    # Gradients
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean().fillna(method='bfill')
        g[f'Grad_{n}'] = np.gradient(ma)
        
    # Technical Indicators
    g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
    g['RSI'] = ta.rsi(g['Close'], length=14)
    bb = ta.bbands(g['Close'], length=20, std=2)
    g['BB_PctB'] = bb.iloc[:, 4] # %B
    g['BB_Upper'] = bb.iloc[:, 2] # Upper Band cho biểu đồ
    g['BB_Lower'] = bb.iloc[:, 0] # Lower Band cho biểu đồ
    g['MACD_Hist'] = ta.macd(g['Close']).iloc[:, 1]
    g['ATR_Rel'] = ta.atr(g['High'], g['Low'], g['Close'], length=14) / g['Close']
    
    # Distance to Previous K10
    ma20 = g['Close'].rolling(20).mean()
    g['Dist_Prev_K10'] = 0.0
    mask_up = g['Close'] >= ma20
    mask_down = g['Close'] < ma20
    g.loc[mask_up, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).min()) / g['Close'].rolling(20).min()
    g.loc[mask_down, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).max()) / g['Close'].rolling(20).max()
    
    return g.dropna().reset_index(drop=True)

def predict_single_step(df_feat, symbol, row_idx=-1):
    """Dự báo cho 1 điểm thời gian cụ thể"""
    if len(df_feat) < 55: return None
    
    # Cắt dữ liệu tại thời điểm row_idx
    if row_idx == -1:
        d50 = df_feat.iloc[-50:]
        d10 = df_feat.iloc[-10:]
    else:
        end = row_idx + 1
        d50 = df_feat.iloc[end-50:end]
        d10 = df_feat.iloc[end-10:end]
        
    if len(d50) < 50: return None
    
    # Scaler
    scaler = bundle['local_scalers_dict'].get(symbol, bundle['global_scaler'])
    try:
        s50 = scaler.transform(d50[FEATS_FULL].values)
        s10 = scaler.transform(d10[FEATS_FULL].values)
    except:
        s50 = bundle['global_scaler'].transform(d50[FEATS_FULL].values)
        s10 = bundle['global_scaler'].transform(d10[FEATS_FULL].values)
        
    # Predict
    p50 = m50.predict(np.expand_dims(s50, 0), verbose=0)[0]
    p10 = m10.predict(np.expand_dims(s10[:, :17], 0), verbose=0)[0]
    
    c50, c10 = np.argmax(p50), np.argmax(p10)
    prob50, prob10 = p50[c50], p10[c10]
    
    # Logic Ensemble
    signal = "THEO DÕI" # Default (Ngang)
    if c50 == 0 and c10 == 0: signal = "MUA"
    elif c50 == 2 and c10 == 2: signal = "BÁN"
    
    return {
        "Date": df_feat.iloc[row_idx]['Date'],
        "Close": df_feat.iloc[row_idx]['Close'],
        "c50": c50, "p50": prob50,
        "c10": c10, "p10": prob10,
        "Signal": signal
    }

# --- 3. GIAO DIỆN ---
st.title("🤖 VN30 AI QUANT TRADING SYSTEM")

tab1, tab2, tab3 = st.tabs(["🚀 DỰ BÁO TOÀN THỊ TRƯỜNG", "📊 BIỂU ĐỒ & SOI MÃ", "📝 LỊCH SỬ TÍN HIỆU"])

# ================= TAB 1: QUÉT 30 MÃ =================
with tab1:
    col1, col2 = st.columns([1, 4])
    with col1:
        btn_scan = st.button("⚡ QUÉT VN30 (DNSE)", type="primary")
    
    if 'scan_data' not in st.session_state:
        st.session_state.scan_data = None

    if btn_scan:
        results = []
        progress_bar = st.progress(0)
        status = st.empty()
        
        for i, sym in enumerate(VN30_LIST):
            status.text(f"Đang xử lý {sym} ({i+1}/30)...")
            df = get_data_dnse(sym)
            df_c = compute_features(df)
            
            res = predict_single_step(df_c, sym, -1) # Dự báo phiên mới nhất
            if res:
                lbl_map = {0: 'mua', 1: 'ngang', 2: 'bán'}
                
                # Format text theo yêu cầu
                win50_txt = f"{lbl_map[res['c50']]} {res['p50']:.0%}"
                win10_txt = f"{lbl_map[res['c10']]} {res['p10']:.0%}"
                ens_txt = res['Signal'] # Đã là HOA (MUA/BÁN/THEO DÕI)
                
                results.append({
                    "Mã": sym,
                    "Giá": res['Close'],
                    "Win50": win50_txt,
                    "Win10": win10_txt,
                    "ENSEMBLE": ens_txt
                })
            
            progress_bar.progress((i+1)/30)
            time.sleep(0.1) # Nhẹ nhàng với API
            
        st.session_state.scan_data = pd.DataFrame(results)
        status.success("Đã quét xong!")
        progress_bar.empty()

    if st.session_state.scan_data is not None:
        df_show = st.session_state.scan_data
        
        # Hàm tô màu
        def style_rows(val):
            color = 'black'
            if val == 'MUA': color = '#28a745' # Xanh lá
            elif val == 'BÁN': color = '#dc3545' # Đỏ
            elif val == 'THEO DÕI': color = '#ffc107' # Vàng
            return f'color: {color}; font-weight: bold'

        st.dataframe(
            df_show.style.applymap(lambda x: style_rows(x) if x in ['MUA', 'BÁN', 'THEO DÕI'] else None, subset=['ENSEMBLE'])
                         .format({"Giá": "{:,.0f}"}),
            use_container_width=True, 
            height=800,
            hide_index=True
        )

# ================= TAB 2: BIỂU ĐỒ & TAB 3: CHI TIẾT =================
# Logic: Chỉ fetch data 1 lần cho cả 2 tab này khi chọn mã

# Selector nằm bên ngoài tab hoặc đầu tab 2
with tab2:
    selected_sym = st.selectbox("🔍 Chọn mã cổ phiếu:", VN30_LIST)
    
    if st.button(f"Phân tích chi tiết {selected_sym}"):
        with st.spinner(f"Đang tải dữ liệu {selected_sym} từ DNSE..."):
            df_stock = get_data_dnse(selected_sym)
            df_features = compute_features(df_stock)
            
            if len(df_features) > 60:
                # --- CHẠY BACKTEST NHANH 60 NGÀY QUA ĐỂ LẤY TÍN HIỆU VẼ ---
                history_preds = []
                # Lấy 60 ngày cuối để vẽ chart, nhưng cần chạy predict cho từng ngày
                # Loop ngược từ hiện tại về quá khứ
                loop_range = range(len(df_features)-1, len(df_features)-61, -1)
                
                for idx in loop_range:
                    if idx < 55: break
                    p = predict_single_step(df_features, selected_sym, idx)
                    if p:
                        history_preds.append(p)
                
                # Convert sang DF để dễ xử lý
                df_preds = pd.DataFrame(history_preds).sort_values('Date').reset_index(drop=True)
                
                # Merge lại với dữ liệu giá để vẽ
                df_plot = df_features.tail(60).copy()
                df_plot = df_plot.merge(df_preds[['Date', 'Signal', 'c50', 'c10']], on='Date', how='left')
                
                # --- VẼ CHART (PLOTLY) ---
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
                                    specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]])

                # 1. Candlestick & BBands
                fig.add_trace(go.Candlestick(x=df_plot['Date'], open=df_plot['Open'], high=df_plot['High'],
                                             low=df_plot['Low'], close=df_plot['Close'], name='Giá'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['BB_Upper'], line=dict(color='gray', width=1), name='BB Upper'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['BB_Lower'], line=dict(color='gray', width=1), name='BB Lower', fill='tonexty'), row=1, col=1)

                # --- VẼ TÍN HIỆU (MŨI TÊN ENSEMBLE) ---
                # Mua: Mũi tên xanh hướng lên dưới đáy nến
                buy_sig = df_plot[df_plot['Signal'] == 'MUA']
                fig.add_trace(go.Scatter(x=buy_sig['Date'], y=buy_sig['Low'] * 0.99, mode='markers', 
                                         marker=dict(symbol='arrow-up', size=10, color='green'), name='AI MUA'), row=1, col=1)
                
                # Bán: Mũi tên đỏ hướng xuống trên đỉnh nến
                sell_sig = df_plot[df_plot['Signal'] == 'BÁN']
                fig.add_trace(go.Scatter(x=sell_sig['Date'], y=sell_sig['High'] * 1.01, mode='markers', 
                                         marker=dict(symbol='arrow-down', size=10, color='red'), name='AI BÁN'), row=1, col=1)

                # --- VẼ TÍN HIỆU LẺ (CHẤM TRÒN) ---
                # Win50 (Model dài hạn): Chấm tròn nhỏ
                # 0=Mua (Xanh), 2=Bán (Đỏ)
                m50_buy = df_plot[df_plot['c50'] == 0]
                fig.add_trace(go.Scatter(x=m50_buy['Date'], y=m50_buy['Low']*0.98, mode='markers',
                                         marker=dict(symbol='circle', size=6, color='lightgreen'), name='Win50 Mua'), row=1, col=1)
                m50_sell = df_plot[df_plot['c50'] == 2]
                fig.add_trace(go.Scatter(x=m50_sell['Date'], y=m50_sell['High']*1.02, mode='markers',
                                         marker=dict(symbol='circle', size=6, color='pink'), name='Win50 Bán'), row=1, col=1)

                # 2. Volume
                colors = ['red' if c < o else 'green' for o, c in zip(df_plot['Open'], df_plot['Close'])]
                fig.add_trace(go.Bar(x=df_plot['Date'], y=df_plot['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

                # 3. RSI
                fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['RSI'], line=dict(color='orange'), name='RSI'), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

                fig.update_layout(height=800, xaxis_rangeslider_visible=False, title=f"Biểu đồ kỹ thuật & Tín hiệu AI: {selected_sym}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Lưu data dự báo vào session state để dùng cho Tab 3
                st.session_state.history_df = df_preds.sort_values('Date', ascending=False)
                
            else:
                st.error("Dữ liệu không đủ để phân tích.")

# ================= TAB 3: LỊCH SỬ CHI TIẾT =================
with tab3:
    if 'history_df' in st.session_state and st.session_state.history_df is not None:
        st.subheader(f"📋 Lịch sử tín hiệu AI: {selected_sym}")
        
        # Lấy data từ Tab 2 đã tính
        df_hist = st.session_state.history_df.head(20).copy() # Lấy 20 ngày gần nhất
        
        # Format lại bảng hiển thị
        display_data = []
        lbl_map = {0: 'mua', 1: 'ngang', 2: 'bán'}
        
        for _, row in df_hist.iterrows():
            display_data.append({
                "Ngày": row['Date'].strftime('%d/%m/%Y'),
                "Giá đóng": f"{row['Close']:,.0f}",
                "Win50 (Dài)": f"{lbl_map[row['c50']]} {row['p50']:.0%}",
                "Win10 (Ngắn)": f"{lbl_map[row['c10']]} {row['p10']:.0%}",
                "ENSEMBLE": row['Signal']
            })
            
        df_display = pd.DataFrame(display_data)
        
        # Hàm tô màu cho bảng lịch sử
        def style_hist(val):
            if val == 'MUA': return 'color: green; font-weight: bold'
            if val == 'BÁN': return 'color: red; font-weight: bold'
            return 'color: orange'

        st.table(df_display.style.applymap(style_hist, subset=['ENSEMBLE']))
    else:
        st.info("👈 Vui lòng chọn mã và bấm 'Phân tích' ở Tab 2 trước.")

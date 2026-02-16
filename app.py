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

# --- 1. CẤU HÌNH & KHỞI TẠO ---
st.set_page_config(page_title="VN30 AI TRADING SYSTEM", layout="wide", page_icon="📈")
warnings.filterwarnings('ignore')

# CSS tùy chỉnh để làm đẹp giao diện
st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    .signal-buy {color: #00ff00; font-weight: bold;}
    .signal-sell {color: #ff4b4b; font-weight: bold;}
    .signal-hold {color: #ffa500; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Danh sách VN30
VN30_SYMBOLS = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
                'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 
                'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']

FEATS_FULL = [
    'RC_1', 'RC_2', 'RC_3', 'RC_5', 'RC_8', 'RC_13', 'RC_21', 'RC_34', 'RC_55',
    'Grad_5', 'Grad_10', 'Grad_20', 'RSI', 'BB_PctB', 'MACD_Hist', 'Vol_Ratio', 'ATR_Rel', 
    'Dist_Prev_K10'
]

# --- 2. LOAD MODEL (CACHE ĐỂ TĂNG TỐC) ---
@st.cache_resource
def load_ai_system():
    try:
        # Đường dẫn file model (Cập nhật đúng đường dẫn của bạn)
        m50 = tf.keras.models.load_model('Full_K10_Win50_Hybrid.keras')
        m10 = tf.keras.models.load_model('Baseline_K10_Win10_Hybrid.keras')
        bundle = joblib.load('smart_scaler_system.pkl')
        return m50, m10, bundle
    except Exception as e:
        st.error(f"❌ Lỗi load model: {e}. Hãy kiểm tra lại đường dẫn file .keras và .pkl")
        return None, None, None

m50, m10, bundle = load_ai_system()

# --- 3. XỬ LÝ DỮ LIỆU (REAL-TIME LOGIC TỪ JUPYTER) ---
def get_data_efficient(symbol):
    try:
        stock = Vnstock().stock(symbol=symbol, source='VCI')
        # Lấy lịch sử 300 ngày
        df = stock.quote.history(start=(datetime.now() - timedelta(days=300)).strftime('%Y-%m-%d'), 
                                 end=datetime.now().strftime('%Y-%m-%d'))
        
        if df is None or df.empty: return pd.DataFrame()

        df = df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 
                                'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # --- REAL-TIME UPDATE ---
        try:
            live_df = stock.quote.now()
            if not live_df.empty:
                cp = float(live_df['close'].iloc[0])
                cv = float(live_df['volume'].iloc[0])
                ch = float(live_df['high'].iloc[0]) if float(live_df['high'].iloc[0]) > 0 else cp
                cl = float(live_df['low'].iloc[0]) if float(live_df['low'].iloc[0]) > 0 else cp
                
                today = pd.Timestamp(datetime.now().date())
                last_date = df.iloc[-1]['Date']

                if last_date.date() == today.date():
                    idx = df.index[-1]
                    df.at[idx, 'Close'] = cp
                    df.at[idx, 'High'] = max(df.at[idx, 'High'], ch)
                    df.at[idx, 'Low'] = min(df.at[idx, 'Low'], cl)
                    df.at[idx, 'Volume'] = cv
                else:
                    new_row = {'Date': today, 'Open': cp, 'High': ch, 'Low': cl, 'Close': cp, 'Volume': cv}
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        except:
            pass # Bỏ qua nếu lỗi realtime API
        
        return df
    except:
        return pd.DataFrame()

def compute_features(df):
    if len(df) < 60: return pd.DataFrame()
    g = df.copy()
    
    # Return features
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: 
        g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
    
    # Gradient features
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean().fillna(method='bfill')
        g[f'Grad_{n}'] = np.gradient(ma)
        
    g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
    g['RSI'] = ta.rsi(g['Close'], length=14)
    
    bb = ta.bbands(g['Close'], length=20, std=2)
    g['BB_PctB'] = bb.iloc[:, 4]
    g['BB_Upper'] = bb.iloc[:, 2] # Cho vẽ biểu đồ
    g['BB_Lower'] = bb.iloc[:, 0] # Cho vẽ biểu đồ
    
    g['MACD_Hist'] = ta.macd(g['Close']).iloc[:, 1]
    g['ATR_Rel'] = ta.atr(g['High'], g['Low'], g['Close'], length=14) / g['Close']
    
    # Custom Feature Dist
    ma20 = g['Close'].rolling(20).mean()
    g['Dist_Prev_K10'] = 0.0
    g.loc[g['Close'] >= ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).min()) / g['Close'].rolling(20).min()
    g.loc[g['Close'] < ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).max()) / g['Close'].rolling(20).max()
    
    return g.dropna().reset_index(drop=True)

def run_prediction(df, symbol, target_idx=-1):
    # Cắt data tại thời điểm target_idx
    if target_idx == -1:
        d_slice = df
    else:
        # Nếu target_idx là 0 (quá khứ xa nhất), logic python slice sẽ lỗi nếu không handle
        if target_idx >= len(df): return None
        d_slice = df.iloc[:target_idx+1]
        
    if len(d_slice) < 50: return None

    # Lấy 50 nến cuối của lát cắt đó
    last_50 = d_slice.tail(50)
    last_10 = d_slice.tail(10)
    
    scaler = bundle['local_scalers_dict'].get(symbol, bundle['global_scaler'])
    
    try:
        s50 = scaler.transform(last_50[FEATS_FULL].values)
        s10 = scaler.transform(last_10[FEATS_FULL].values)
    except:
        s50 = bundle['global_scaler'].transform(last_50[FEATS_FULL].values)
        s10 = bundle['global_scaler'].transform(last_10[FEATS_FULL].values)

    # Dự báo
    p50_raw = m50.predict(np.expand_dims(s50, 0), verbose=0)[0]
    p10_raw = m10.predict(np.expand_dims(s10[:, :17], 0), verbose=0)[0]

    c50, c10 = np.argmax(p50_raw), np.argmax(p10_raw)
    prob50, prob10 = p50_raw[c50], p10_raw[c10]

    labels = {0: "MUA", 1: "HOLD", 2: "BÁN"}
    ens_label = "THEO DÕI"
    if c50 == 0 and c10 == 0: ens_label = "MUA"
    elif c50 == 2 and c10 == 2: ens_label = "BÁN"
    elif c50 == 1 and c10 == 1: ens_label = "NGANG"
    
    return {
        "Date": d_slice.iloc[-1]['Date'],
        "Close": d_slice.iloc[-1]['Close'],
        "Win50_Lbl": labels[c50].lower(), "Win50_Prob": prob50,
        "Win10_Lbl": labels[c10].lower(), "Win10_Prob": prob10,
        "Ensemble": ens_label
    }

# --- 4. GIAO DIỆN CHÍNH ---
st.title("🤖 VN30 AI PRO TRADING SYSTEM")

# Sidebar chọn mã
st.sidebar.header("Cấu hình")
selected_symbol = st.sidebar.selectbox("Chọn cổ phiếu", VN30_SYMBOLS)
days_lookback = st.sidebar.slider("Số ngày xem lại (Tab 3)", 5, 20, 10)

# Load data cho mã được chọn (dùng cho Tab 1 và Tab 3)
with st.spinner(f"Đang tải dữ liệu {selected_symbol}..."):
    df_main = get_data_efficient(selected_symbol)
    df_main_c = compute_features(df_main)

# TABS
tab1, tab2, tab3 = st.tabs(["📊 Đồ Thị & Chỉ Báo", "🚀 Dự Báo Toàn Thị Trường", "📝 Chi Tiết Lịch Sử AI"])

# ================= TAB 1: ĐỒ THỊ =================
with tab1:
    if not df_main_c.empty:
        st.subheader(f"Biểu đồ kỹ thuật & Tín hiệu AI: {selected_symbol}")
        
        # Chạy dự báo quá khứ để lấy điểm vẽ (Scan ngược 50 phiên gần nhất để vẽ lên biểu đồ)
        scan_len = 60 
        ai_signals = []
        # Chỉ chạy loop dự báo nếu data đủ dài
        if len(df_main_c) > 55:
            start_idx = len(df_main_c) - scan_len if len(df_main_c) > scan_len else 55
            for i in range(start_idx, len(df_main_c)):
                res = run_prediction(df_main_c, selected_symbol, target_idx=i)
                if res: ai_signals.append(res)
        
        df_sig = pd.DataFrame(ai_signals)

        # Plotly Subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
                            specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]])

        # 1. Nến & BB
        fig.add_trace(go.Candlestick(x=df_main_c['Date'], open=df_main_c['Open'], high=df_main_c['High'],
                                     low=df_main_c['Low'], close=df_main_c['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_main_c['Date'], y=df_main_c['BB_Upper'], line=dict(color='gray', width=1), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_main_c['Date'], y=df_main_c['BB_Lower'], line=dict(color='gray', width=1), name='BB Lower', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

        # 2. AI Markers (Mũi tên & Chấm)
        if not df_sig.empty:
            # Mũi tên cho Ensemble (MUA/BÁN)
            buys = df_sig[df_sig['Ensemble'] == 'MUA']
            sells = df_sig[df_sig['Ensemble'] == 'BÁN']
            
            # Mũi tên MUA (Xanh, hướng lên)
            fig.add_trace(go.Scatter(x=buys['Date'], y=buys['Close']*0.98, mode='markers', 
                                     marker=dict(symbol='triangle-up', size=12, color='#00FF00'), name='AI Mua'), row=1, col=1)
            # Mũi tên BÁN (Đỏ, hướng xuống)
            fig.add_trace(go.Scatter(x=sells['Date'], y=sells['Close']*1.02, mode='markers', 
                                     marker=dict(symbol='triangle-down', size=12, color='#FF0000'), name='AI Bán'), row=1, col=1)
            
            # Chấm tròn (Dự báo thực tế tại mỗi điểm) - Biểu thị điểm AI có đưa ra nhận định (bất kể Buy/Sell/Hold)
            fig.add_trace(go.Scatter(x=df_sig['Date'], y=df_sig['Close'], mode='markers',
                                     marker=dict(symbol='circle', size=4, color='white', opacity=0.5), name='AI Scan Point'), row=1, col=1)

        # 3. Volume
        colors = ['red' if o > c else 'green' for o, c in zip(df_main_c['Open'], df_main_c['Close'])]
        fig.add_trace(go.Bar(x=df_main_c['Date'], y=df_main_c['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

        # 4. RSI
        fig.add_trace(go.Scatter(x=df_main_c['Date'], y=df_main_c['RSI'], line=dict(color='#FFD700'), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=800, template='plotly_dark', xaxis_rangeslider_visible=False, title_text=f"AI Analysis: {selected_symbol}")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Không đủ dữ liệu để vẽ biểu đồ.")

# ================= TAB 2: DỰ BÁO TOÀN VN30 =================
with tab2:
    st.write("### 📡 Bảng tín hiệu Real-time VN30")
    if st.button("🔄 Quét toàn bộ thị trường ngay"):
        results_scan = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, sym in enumerate(VN30_SYMBOLS):
            status_text.text(f"Đang phân tích: {sym}...")
            d = get_data_efficient(sym)
            dc = compute_features(d)
            
            if not dc.empty:
                res = run_prediction(dc, sym, target_idx=-1) # Lấy nến cuối cùng
                if res:
                    results_scan.append({
                        "Mã": sym,
                        "Giá": f"{res['Close']:,.0f}",
                        "Win50": f"{res['Win50_Lbl']} ({res['Win50_Prob']:.0%})",
                        "Win10": f"{res['Win10_Lbl']} ({res['Win10_Prob']:.0%})",
                        "ENSEMBLE": res['Ensemble']
                    })
            progress_bar.progress((i + 1) / 30)
            time.sleep(0.1) # Nhẹ nhàng với API

        status_text.success("✅ Đã quét xong!")
        df_res = pd.DataFrame(results_scan)

        if not df_res.empty:
            # Chia cột hiển thị
            col_buy, col_sell, col_side = st.columns(3)
            
            with col_buy:
                st.success("🟢 KHUYẾN NGHỊ MUA")
                df_buy = df_res[df_res['ENSEMBLE'] == 'MUA']
                st.dataframe(df_buy, hide_index=True, use_container_width=True)

            with col_sell:
                st.error("🔴 KHUYẾN NGHỊ BÁN")
                df_sell = df_res[df_res['ENSEMBLE'] == 'BÁN']
                st.dataframe(df_sell, hide_index=True, use_container_width=True)

            with col_side:
                st.warning("🟡 SIDEWAY / THEO DÕI")
                df_side = df_res[~df_res['ENSEMBLE'].isin(['MUA', 'BÁN'])]
                st.dataframe(df_side, hide_index=True, use_container_width=True)

# ================= TAB 3: CHI TIẾT LỊCH SỬ =================
with tab3:
    st.subheader(f"📝 Nhật ký dự báo AI: {selected_symbol}")
    st.write(f"Dữ liệu {days_lookback} phiên gần nhất")

    if not df_main_c.empty and len(df_main_c) > 60:
        history_data = []
        # Loop ngược từ hiện tại về quá khứ
        for i in range(days_lookback):
            idx = len(df_main_c) - 1 - i
            res = run_prediction(df_main_c, selected_symbol, target_idx=idx)
            if res:
                # Format theo yêu cầu: win thường + %, Ensemble HOA
                row = {
                    "Ngày": res['Date'].strftime('%d/%m/%Y'),
                    "Giá Đóng": f"{res['Close']:,.0f}",
                    "Model Win50": f"{res['Win50_Lbl']} ({res['Win50_Prob']:.0%})",
                    "Model Win10": f"{res['Win10_Lbl']} ({res['Win10_Prob']:.0%})",
                    "ENSEMBLE": res['Ensemble'] # Đã viết HOA ở hàm process
                }
                history_data.append(row)
        
        df_hist = pd.DataFrame(history_data)
        
        # Style tô màu cho bảng
        def highlight_ensemble(val):
            color = ''
            if val == 'MUA': color = 'background-color: #1a472a; color: #4ade80' # Xanh đậm
            elif val == 'BÁN': color = 'background-color: #4a1a1a; color: #f87171' # Đỏ đậm
            return color

        st.dataframe(df_hist.style.applymap(highlight_ensemble, subset=['ENSEMBLE']), 
                     use_container_width=True, hide_index=True)
    else:
        st.info("Chưa đủ dữ liệu để hiển thị lịch sử.")

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

# --- 1. CẤU HÌNH & LOAD ASSETS ---
st.set_page_config(page_title="VN30 AI ENSEMBLE PRO", layout="wide")

@st.cache_resource
def load_ai_system():
    # Điều chỉnh đường dẫn file cho khớp với hosting của bạn
    m50 = tf.keras.models.load_model('Full_K10_Win50_Hybrid.keras')
    m10 = tf.keras.models.load_model('Baseline_K10_Win10_Hybrid.keras')
    bundle = joblib.load('smart_scaler_system.pkl')
    return m50, m10, bundle

m50, m10, bundle = load_ai_system()
vn30_symbols = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']
FEATS_FULL = ['RC_1', 'RC_2', 'RC_3', 'RC_5', 'RC_8', 'RC_13', 'RC_21', 'RC_34', 'RC_55', 'Grad_5', 'Grad_10', 'Grad_20', 'RSI', 'BB_PctB', 'MACD_Hist', 'Vol_Ratio', 'ATR_Rel', 'Dist_Prev_K10']

# --- 2. CORE LOGIC (GIỮ NGUYÊN TỪ JUPYTER) ---
def get_data_dnse(symbol):
    try:
        stock = Vnstock().stock(symbol=symbol, source='DNSE')
        df = stock.quote.history(start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), 
                                 end=datetime.now().strftime('%Y-%m-%d'))
        if df is None or df.empty: return pd.DataFrame()
        df = df.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date').reset_index(drop=True)
    except: return pd.DataFrame()

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
    g['BB_Lower'], g['BB_Mid'], g['BB_Upper'], g['BB_PctB'] = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2], bb.iloc[:, 4]
    g['MACD_Hist'] = ta.macd(g['Close']).iloc[:, 1]
    g['ATR_Rel'] = ta.atr(g['High'], g['Low'], g['Close'], length=14) / g['Close']
    ma20 = g['Close'].rolling(20).mean()
    g['Dist_Prev_K10'] = 0.0
    g.loc[g['Close'] >= ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).min()) / g['Close'].rolling(20).min()
    g.loc[g['Close'] < ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).max()) / g['Close'].rolling(20).max()
    return g.dropna().reset_index(drop=True)

def predict_ensemble(df_c, symbol, target_idx=-1):
    if target_idx == -1: d50, d10 = df_c.tail(50), df_c.tail(10)
    else:
        end = target_idx + 1
        d50, d10 = df_c.iloc[end-50:end], df_c.iloc[end-10:end]
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
    
    return {
        'date': df_c.iloc[target_idx]['Date'],
        'close': df_c.iloc[target_idx]['Close'],
        'win50': f"{labels[c50]} {p50[c50]:.0%}",
        'win10': f"{labels[c10]} {p10[c10]:.0%}",
        'ENSEMBLE': ensemble,
        'c50': c50, 'c10': c10
    }

# --- 3. GIAO DIỆN 3 TABS ---
t1, t2, t3 = st.tabs(["🚀 Tổng hợp VN30", "📊 Đồ thị AI", "🔍 Chi tiết 20 phiên"])

# --- TAB 1: TỔNG HỢP VN30 ---
with t1:
    st.subheader("Báo cáo AI Ensemble - Toàn bộ VN30")
    if st.button("📡 Chạy quét 30 mã (Dữ liệu hôm nay)"):
        results = []
        bar = st.progress(0)
        for i, sym in enumerate(vn30_symbols):
            df = get_data_dnse(sym)
            df_c = compute_features(df)
            if not df_c.empty:
                res = predict_ensemble(df_c, sym)
                if res:
                    results.append({"Mã": sym, "Giá HT": f"{res['close']:,.0f}", "win50": res['win50'], "win10": res['win10'], "ENSEMBLE": res['ENSEMBLE']})
            bar.progress((i+1)/30)
            time.sleep(0.1)
        st.session_state.all_res = pd.DataFrame(results)

    if 'all_res' in st.session_state:
        df_show = st.session_state.all_res
        c1, c2, c3 = st.columns(3)
        with c1: st.success("🟢 MUA"); st.table(df_show[df_show['ENSEMBLE']=="MUA"][["Mã", "Giá HT", "ENSEMBLE"]])
        with c2: st.error("🔴 BÁN"); st.table(df_show[df_show['ENSEMBLE']=="BÁN"][["Mã", "Giá HT", "ENSEMBLE"]])
        with c3: st.warning("🟡 THEO DÕI"); st.table(df_show[df_show['ENSEMBLE']=="THEO DÕI"][["Mã", "Giá HT", "ENSEMBLE"]])
        st.write("### Chi tiết xác suất lẻ")
        st.dataframe(df_show, use_container_width=True, hide_index=True)

# --- TAB 2: ĐỒ THỊ AI ---
with t2:
    sel_sym = st.selectbox("Chọn mã soi kỹ", vn30_symbols, key="box_tab2")
    df_plot = get_data_dnse(sel_sym)
    df_c = compute_features(df_plot)
    
    if not df_c.empty:
        # Dự báo 50 phiên cuối để vẽ marker
        plot_data = df_c.tail(100).copy()
        signals = []
        for i in range(len(df_c)-50, len(df_c)):
            signals.append(predict_ensemble(df_c, sel_sym, target_idx=i))
        
        # Plotly Subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.2, 0.3])
        
        # 1. Giá + BB
        fig.add_trace(go.Candlestick(x=plot_data['Date'], open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name="Giá"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['BB_Upper'], line=dict(color='gray', dash='dash'), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['BB_Lower'], line=dict(color='gray', dash='dash'), name="BB Lower"), row=1, col=1)
        
        # Markers: Chấm tròn (Win50) & Mũi tên (Ensemble)
        sig_df = pd.DataFrame([s for s in signals if s is not None])
        # Win50: Trắng (Mua), Đen (Bán)
        fig.add_trace(go.Scatter(x=sig_df[sig_df['c50']==0]['date'], y=sig_df[sig_df['c50']==0]['close'], mode='markers', marker=dict(color='white', size=8), name="AI50: Mua"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sig_df[sig_df['c50']==2]['date'], y=sig_df[sig_df['c50']==2]['close'], mode='markers', marker=dict(color='black', size=8), name="AI50: Bán"), row=1, col=1)
        
        # Ensemble: Mũi tên
        fig.add_trace(go.Scatter(x=sig_df[sig_df['ENSEMBLE']=="MUA"]['date'], y=sig_df[sig_df['ENSEMBLE']=="MUA"]['close']*0.97, mode='markers', marker=dict(symbol='triangle-up', color='lime', size=12), name="Ensemble: MUA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sig_df[sig_df['ENSEMBLE']=="BÁN"]['date'], y=sig_df[sig_df['ENSEMBLE']=="BÁN"]['close']*1.03, mode='markers', marker=dict(symbol='triangle-down', color='red', size=12), name="Ensemble: BÁN"), row=1, col=1)
        
        # 2. Volume
        fig.add_trace(go.Bar(x=plot_data['Date'], y=plot_data['Volume'], name="Volume"), row=2, col=1)
        
        # 3. RSI
        fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['RSI'], line=dict(color='yellow'), name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", row=3, col=1); fig.add_hline(y=30, line_dash="dot", row=3, col=1)
        
        fig.update_layout(height=800, template='plotly_dark', xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: CHI TIẾT 20 PHIÊN ---
with t3:
    sel_sym_h = st.selectbox("Chọn mã xem lịch sử", vn30_symbols, key="box_tab3")
    df_h = get_data_dnse(sel_sym_h)
    df_hc = compute_features(df_h)
    if not df_hc.empty:
        st.write(f"### Lịch sử dự báo AI: {sel_sym_h} (20 phiên gần nhất)")
        hist_results = []
        # Chạy ngược từ ngày mới nhất về 20 phiên trước
        for i in range(len(df_hc)-20, len(df_hc)):
            r = predict_ensemble(df_hc, sel_sym_h, target_idx=i)
            if r:
                hist_results.append({
                    "Ngày": r['date'].strftime('%d/%m/%Y'),
                    "Giá Đóng": f"{r['close']:,.0f}",
                    "win50": r['win50'],
                    "win10": r['win10'],
                    "ENSEMBLE": r['ENSEMBLE']
                })
        # Đảo ngược danh sách để ngày mới nhất lên đầu
        st.table(pd.DataFrame(hist_results[::-1]))

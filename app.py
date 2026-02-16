import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- LOGIC ENSEMBLE ---
def get_ensemble_signal(p50, p10):
    r50 = np.argmax(p50)
    r10 = np.argmax(p10)
    
    # Ưu tiên xu hướng dài hạn làm nền tảng
    if r50 == 0 and r10 == 0: return "MUA MẠNH 💎", "Mua"
    if r50 == 0: return "MUA (Đợi điểm vào) 🟢", "Mua"
    if r50 == 2: return "BÁN 🔴", "Bán"
    if r10 == 2: return "CẨN TRỌNG 🟡", "Ngang"
    return "THEO DÕI ⚪", "Ngang"

# --- TAB 1: BẢNG TỔNG HỢP THEO NHÓM ---
with tab_scan:
    if st.session_state.scan_results is not None:
        df_res = st.session_state.scan_results
        
        # Tạo thêm cột phân loại Ensemble
        # (Giả sử bạn đã chạy prediction và lưu vào session_state)
        
        c_mua, c_ngang, c_ban = st.columns(3)
        
        with c_mua:
            st.success("🟢 DANH MỤC MUA")
            # Filter và hiển thị bảng Mua
            
        with c_ngang:
            st.warning("🟡 THEO DÕI")
            
        with c_ban:
            st.error("🔴 DANH MỤC BÁN")

# --- TAB 2: CHI TIẾT & BIỂU ĐỒ KỸ THUẬT ---
def draw_pro_chart(df, symbol, signal):
    # Tính Bollinger Bands
    bb = ta.bbands(df['Close'], length=20, std=2)
    df = pd.concat([df, bb], axis=1)
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.5, 0.2, 0.3])

    # 1. Candlestick + BB
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Giá'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BBU_20_2.0'], line=dict(color='rgba(173, 216, 230, 0.4)'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BBL_20_2.0'], line=dict(color='rgba(173, 216, 230, 0.4)'), fill='tonexty', name='BB Lower'), row=1, col=1)

    # Thêm Mũi tên dự báo Ensemble
    last_date = df['Date'].iloc[-1]
    last_price = df['Close'].iloc[-1]
    
    arrow_color = "green" if "MUA" in signal else ("red" if "BÁN" in signal else "gray")
    ay = -40 if "MUA" in signal else 40
    
    fig.add_annotation(x=last_date, y=last_price, text=f"Dự báo: {signal}",
                       showarrow=True, arrowhead=2, arrowcolor=arrow_color, ay=ay, row=1, col=1)

    # 2. RSI
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

    # 3. Volume
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='orange'), row=3, col=1)

    fig.update_layout(height=800, template='plotly_dark', xaxis_rangeslider_visible=False)
    return fig

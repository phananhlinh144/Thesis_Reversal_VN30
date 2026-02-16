import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
import warnings
import os
from datetime import datetime
from vnstock import *

# Tắt các cảnh báo
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. CẤU HÌNH ---
MODEL_WIN50_PATH = 'Full_K10_Win50_Hybrid.keras'
MODEL_WIN10_PATH = 'Baseline_K10_Win10_Hybrid.keras'
SCALER_PATH      = 'smart_scaler_system.pkl'
HISTORY_CSV_PATH = 'vn30_data_raw.csv' 
OUTPUT_CSV_PATH  = 'vn30_signals.csv'

FEATS_FULL = [
    'RC_1', 'RC_2', 'RC_3', 'RC_5', 'RC_8', 'RC_13', 'RC_21', 'RC_34', 'RC_55',
    'Grad_5', 'Grad_10', 'Grad_20', 'RSI', 'BB_PctB', 'MACD_Hist', 'Vol_Ratio', 'ATR_Rel', 'Dist_Prev_K10'
]

# --- 2. LOAD MODELS ---
print("⏳ Đang khởi tạo hệ thống AI...")
try:
    model_win50 = tf.keras.models.load_model(MODEL_WIN50_PATH)
    model_win10 = tf.keras.models.load_model(MODEL_WIN10_PATH)
    scaler_bundle = joblib.load(SCALER_PATH)
    local_scalers = scaler_bundle['local_scalers_dict']
    global_scaler = scaler_bundle['global_scaler']
    print("✅ Load Model thành công.")
except Exception as e:
    print(f"❌ Lỗi Load Model: {e}")
    exit()

# --- 3. HÀM TÍNH TOÁN (KHÔNG DÙNG PANDAS_TA) ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_features(df):
    if len(df) < 60: return pd.DataFrame()
    g = df.copy()
    g = g.ffill().bfill()
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: 
        g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
        
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean().bfill()
        g[f'Grad_{n}'] = np.gradient(ma)
        
    g['Vol_Ratio'] = g['Volume'] / g['Volume'].rolling(20).mean()
    
    g['RSI'] = compute_rsi(g['Close'], 14)
    ma20 = g['Close'].rolling(20).mean()
    std20 = g['Close'].rolling(20).std()
    g['BB_PctB'] = (g['Close'] - (ma20 - 2*std20)) / ((ma20 + 2*std20) - (ma20 - 2*std20))
    
    exp12 = g['Close'].ewm(span=12, adjust=False).mean()
    exp26 = g['Close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    g['MACD_Hist'] = macd - macd.ewm(span=9, adjust=False).mean()
    
    tr = pd.concat([g['High']-g['Low'], abs(g['High']-g['Close'].shift()), abs(g['Low']-g['Close'].shift())], axis=1).max(axis=1)
    g['ATR_Rel'] = tr.rolling(14).mean() / g['Close']
    
    g['Dist_Prev_K10'] = 0.0
    g.loc[g['Close'] >= ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).min()) / g['Close'].rolling(20).min()
    g.loc[g['Close'] < ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).max()) / g['Close'].rolling(20).max()
    return g.dropna().reset_index(drop=True)

def get_hybrid_data(symbol):
    try:
        full_hist = pd.read_csv(HISTORY_CSV_PATH)
        # Ép kiểu Date ngay lập tức
        full_hist['Date'] = pd.to_datetime(full_hist['Date']) 
        df_old = full_hist[full_hist['Symbol'] == symbol].sort_values('Date')
        df_new = stock_historical_data(symbol=symbol, start_date="2026-01-11", end_date=datetime.now().strftime('%Y-%m-%d'), resolution='1D', type='stock', source='VCI')
        if df_new is not None and not df_new.empty:
            df_new = df_new.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            df_final = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=['Date'])
        else: df_final = df_old
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']: df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        return df_final.sort_values('Date').reset_index(drop=True)
    except: return pd.DataFrame()

def predict_at_index(df_feat, symbol, idx=-1):
    actual_idx = len(df_feat) + idx
    d50, d10 = df_feat.iloc[actual_idx-49 : actual_idx+1], df_feat.iloc[actual_idx-9 : actual_idx+1]
    scaler = local_scalers.get(symbol, global_scaler)
    s50, s10 = scaler.transform(d50[FEATS_FULL].values), scaler.transform(d10[FEATS_FULL].values)
    p50, p10 = model_win50.predict(np.expand_dims(s50, 0), verbose=0)[0], model_win10.predict(np.expand_dims(s10[:, :17], 0), verbose=0)[0]
    c50, c10 = np.argmax(p50), np.argmax(p10)
    sig = "MUA" if (c50==0 and c10==0) else ("BÁN" if (c50==2 and c10==2) else "THEO DÕI")
    return {"Mã": symbol, "Ngày": pd.to_datetime(df_feat.iloc[actual_idx]['Date']).strftime('%Y-%m-%d'), "Giá": int(df_feat.iloc[actual_idx]['Close']), "ENSEMBLE": sig}

if __name__ == "__main__":
    vn30 = ['ACB','BCM','BID','CTG','DGC','FPT','GAS','GVR','HDB','HPG','LPB','MSN','MBB','MWG','PLX','SAB','SHB','SSB','SSI','STB','TCB','TPB','VCB','VIC','VHM','VIB','VJC','VNM','VPB','VRE']
    final_output = []
    for sym in vn30:
        df = get_hybrid_data(sym)
        df_f = compute_features(df)
        if not df_f.empty:
            for j in range(-5, 0):
                try: final_output.append(predict_at_index(df_f, sym, j))
                except: continue
        time.sleep(1.5)
    if final_output:
        pd.DataFrame(final_output).to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"✅ Đã tạo file {OUTPUT_CSV_PATH}")
        print(f"📊 Đã dự báo xong. Tổng số dòng kết quả: {len(final_output)}")


import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
import warnings
import os
import gspread
from datetime import datetime
from vnstock import * warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. CẤU HÌNH ---
MODEL_WIN50_PATH = 'Full_K10_Win50_Hybrid.keras'
MODEL_WIN10_PATH = 'Baseline_K10_Win10_Hybrid.keras'
SCALER_PATH      = 'smart_scaler_system.pkl'
HISTORY_CSV_PATH = 'vn30_data_raw.csv' 
GOOGLE_SHEET_NAME = "VN30_AI"

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

# --- 3. HÀM TỰ TÍNH INDICATORS (THAY THẾ PANDAS_TA) ---

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
    
    # 1. Returns Change
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: 
        g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
        
    # 2. Gradient của MA
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean().bfill()
        g[f'Grad_{n}'] = np.gradient(ma)
    
    # 3. Vol Ratio & RSI (Tự tính bằng Pandas)
    g['Vol_Ratio'] = g['Volume'] / g['Volume'].rolling(20).mean()
    g['RSI'] = compute_rsi(g['Close'], 14)
    
    # 4. Bollinger Bands %B (Tự tính)
    ma20 = g['Close'].rolling(20).mean()
    std20 = g['Close'].rolling(20).std()
    upper = ma20 + (std20 * 2)
    lower = ma20 - (std20 * 2)
    g['BB_PctB'] = (g['Close'] - lower) / (upper - lower)
    
    # 5. MACD Hist (Tự tính)
    exp12 = g['Close'].ewm(span=12, adjust=False).mean()
    exp26 = g['Close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    g['MACD_Hist'] = macd - signal
    
    # 6. ATR Relative (Tự tính đơn giản)
    high_low = g['High'] - g['Low']
    high_close = np.abs(g['High'] - g['Close'].shift())
    low_close = np.abs(g['Low'] - g['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    g['ATR_Rel'] = tr.rolling(14).mean() / g['Close']
    
    # 7. Dist_Prev_K10
    g['Dist_Prev_K10'] = 0.0
    g.loc[g['Close'] >= ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).min()) / g['Close'].rolling(20).min()
    g.loc[g['Close'] < ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).max()) / g['Close'].rolling(20).max()

    return g.dropna().reset_index(drop=True)

# --- (Các hàm get_hybrid_data, predict_at_index, push_to_sheets giữ nguyên như của bạn) ---

def get_hybrid_data(symbol):
    try:
        full_hist = pd.read_csv(HISTORY_CSV_PATH)
        full_hist['Date'] = pd.to_datetime(full_hist['Date'])
        df_old = full_hist[full_hist['Symbol'] == symbol].sort_values('Date')
        start_date = "2026-01-11"
        end_date = datetime.now().strftime('%Y-%m-%d')
        try:
            df_new = stock_historical_data(symbol=symbol, start_date=start_date, end_date=end_date, 
                                           resolution='1D', type='stock', source='VCI')
            if df_new is not None and not df_new.empty:
                df_new = df_new.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
                df_new['Date'] = pd.to_datetime(df_new['Date'])
                df_final = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_final = df_old
        except:
            df_final = df_old
        df_final = df_final.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        return df_final
    except Exception as e:
        print(f"⚠️ Lỗi xử lý {symbol}: {e}")
        return pd.DataFrame()

def predict_at_index(df_feat, symbol, idx=-1):
    actual_idx = len(df_feat) + idx
    if actual_idx < 50: return None
    d50 = df_feat.iloc[actual_idx-49 : actual_idx+1]
    d10 = df_feat.iloc[actual_idx-9 : actual_idx+1]
    scaler = local_scalers.get(symbol, global_scaler)
    s50 = scaler.transform(d50[FEATS_FULL].values)
    s10 = scaler.transform(d10[FEATS_FULL].values)
    p50_raw = model_win50.predict(np.expand_dims(s50, 0), verbose=0)[0]
    p10_raw = model_win10.predict(np.expand_dims(s10[:, :17], 0), verbose=0)[0]
    c50, c10 = np.argmax(p50_raw), np.argmax(p10_raw)
    signal = "THEO DÕI"
    if c50 == 0 and c10 == 0: signal = "MUA"
    elif c50 == 2 and c10 == 2: signal = "BÁN"
    labels = {0: 'Tăng', 1: 'Ngang', 2: 'Giảm'}
    return {"Mã": symbol, "Ngày": df_feat.iloc[actual_idx]['Date'].strftime('%Y-%m-%d'), "Giá": int(df_feat.iloc[actual_idx]['Close']), "Win50": labels[c50], "Win10": labels[c10], "ENSEMBLE": signal}

def push_to_sheets(final_output):
    try:
        print("📤 Đang đẩy dữ liệu lên Google Sheets...")
        gc = gspread.service_account(filename='credentials.json')
        sh = gc.open(GOOGLE_SHEET_NAME)
        wks = sh.get_all_worksheets()[0]
        df_final = pd.DataFrame(final_output)
        wks.clear()
        wks.update([df_final.columns.values.tolist()] + df_final.values.tolist())
        print("✅ Đã cập nhật kết quả lên Google Sheets thành công!")
    except Exception as e:
        print(f"❌ Lỗi đẩy Sheets: {e}")

if __name__ == "__main__":
    vn30 = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
            'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 
            'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']
    final_output = []
    LOOKBACK = 15 
    print(f"🚀 Bắt đầu quét dữ liệu VN30...")
    for i, sym in enumerate(vn30):
        print(f"\r⏳ [{i+1}/30] Đang xử lý: {sym:<5}", end="")
        df = get_hybrid_data(sym)
        if df.empty: continue
        df_feat = compute_features(df)
        if df_feat.empty: continue
        for j in range(-LOOKBACK, 0):
            try:
                res = predict_at_index(df_feat, sym, idx=j)
                if res: final_output.append(res)
            except: continue
        time.sleep(1.0) 
    if final_output:
        push_to_sheets(final_output)

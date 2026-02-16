import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pandas_ta as ta
import time
import warnings
import os
from datetime import datetime, timedelta
from vnstock import * # Sử dụng vnstock phiên bản cũ

# Tắt các cảnh báo để log sạch sẽ
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
MODEL_WIN50_PATH = 'Full_K10_Win50_Hybrid.keras'
MODEL_WIN10_PATH = 'Baseline_K10_Win10_Hybrid.keras'
SCALER_PATH      = 'smart_scaler_system.pkl'
HISTORY_CSV_PATH = 'vn30_data_raw.csv' 

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

# --- 3. HÀM XỬ LÝ DỮ LIỆU ---

def get_hybrid_data(symbol):
    """Đọc dữ liệu từ file csv (đến 10/1) và nối thêm từ VCI bằng vnstock cũ"""
    try:
        # 1. Đọc dữ liệu lịch sử từ file csv (Dữ liệu bạn đã gửi)
        full_hist = pd.read_csv(HISTORY_CSV_PATH)
        full_hist['Date'] = pd.to_datetime(full_hist['Date'])
        df_old = full_hist[full_hist['Symbol'] == symbol].sort_values('Date')
        
        # 2. Lấy dữ liệu mới từ nguồn VCI (vnstock cũ dùng hàm stock_historical_data)
        start_date = "2026-01-11"
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Lưu ý: vnstock cũ lấy dữ liệu theo định dạng 'YYYY-MM-DD'
            df_new = stock_historical_data(symbol=symbol, 
                                           start_date=start_date, 
                                           end_date=end_date, 
                                           resolution='1D', 
                                           type='stock', 
                                           source='VCI')
            
            if df_new is not None and not df_new.empty:
                # Chuẩn hóa tên cột vnstock cũ về dạng chung
                df_new = df_new.rename(columns={'time':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
                df_new['Date'] = pd.to_datetime(df_new['Date'])
                df_final = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_final = df_old
        except:
            df_final = df_old

        df_final = df_final.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
        
        # Ép kiểu dữ liệu số để tránh lỗi tính toán indicators
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
            
        return df_final
    except Exception as e:
        print(f"⚠️ Lỗi xử lý {symbol}: {e}")
        return pd.DataFrame()

def compute_features(df):
    if len(df) < 60: return pd.DataFrame()
    g = df.copy()
    g = g.ffill().bfill()
    
    # Tính toán Returns Change
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55]: 
        g[f'RC_{n}'] = g['Close'].pct_change(n) * 100
        
    # Tính toán Gradient của các đường MA
    for n in [5, 10, 20]:
        ma = g['Close'].rolling(window=n).mean().bfill()
        g[f'Grad_{n}'] = np.gradient(ma)
    
    # Chỉ báo kỹ thuật từ pandas_ta
    g['Vol_Ratio'] = g['Volume'] / ta.sma(g['Volume'], length=20)
    g['RSI'] = ta.rsi(g['Close'], length=14)
    
    bb = ta.bbands(g['Close'], length=20, std=2)
    # Tìm cột có tên chứa 'B' (thường là BBP_20_2.0)
    pctb_col = [c for c in bb.columns if c.startswith('BBP')]
    if pctb_col:
        g['BB_PctB'] = bb[pctb_col[0]]
    else:
        g['BB_PctB'] = bb.iloc[:, 4] # Quay lại cách cũ nếu không tìm thấy
    
    g['MACD_Hist'] = ta.macd(g['Close']).iloc[:, 1]
    g['ATR_Rel'] = ta.atr(g['High'], g['Low'], g['Close'], length=14) / g['Close']
    
    # Khoảng cách so với nến K10 trước đó
    ma20 = g['Close'].rolling(20).mean()
    g['Dist_Prev_K10'] = 0.0
    g.loc[g['Close'] >= ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).min()) / g['Close'].rolling(20).min()
    g.loc[g['Close'] < ma20, 'Dist_Prev_K10'] = (g['Close'] - g['Close'].rolling(20).max()) / g['Close'].rolling(20).max()

    g = g.dropna()
    
    if len(g) < 55:
        print(f"⚠️ Cảnh báo: Dữ liệu sau khi tính toán quá ít dòng.")
    return g.reset_index(drop=True)

def predict_at_index(df_feat, symbol, idx=-1):
    actual_idx = len(df_feat) + idx if idx < 0 else idx
    if actual_idx < 50: return None

    # Slice cửa sổ 50 phiên và 10 phiên
    d50 = df_feat.iloc[actual_idx-49 : actual_idx+1]
    d10 = df_feat.iloc[actual_idx-9 : actual_idx+1]

    # Scaling dữ liệu
    scaler = local_scalers.get(symbol, global_scaler)
    s50 = scaler.transform(d50[FEATS_FULL].values)
    s10 = scaler.transform(d10[FEATS_FULL].values)

    # Dự báo từ 2 model Hybrid
    p50_raw = model_win50.predict(np.expand_dims(s50, 0), verbose=0)[0]
    p10_raw = model_win10.predict(np.expand_dims(s10[:, :17], 0), verbose=0)[0]

    c50, c10 = np.argmax(p50_raw), np.argmax(p10_raw)
    
    signal = "THEO DÕI"
    if c50 == 0 and c10 == 0: signal = "MUA"
    elif c50 == 2 and c10 == 2: signal = "BÁN"
    
    labels = {0: 'Tăng', 1: 'Ngang', 2: 'Giảm'}

    return {
        "Mã": symbol,
        "Ngày": df_feat.iloc[actual_idx]['Date'].strftime('%Y-%m-%d'),
        "Giá": int(df_feat.iloc[actual_idx]['Close']),
        "Win50": f"{labels[c50]} ({p50_raw[c50]:.0%})",
        "Win10": f"{labels[c10]} ({p10_raw[c10]:.0%})",
        "ENSEMBLE": signal
    }

# --- 4. CHƯƠNG TRÌNH CHÍNH ---

if __name__ == "__main__":
    vn30 = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
            'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB', 
            'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']
    
    final_output = []
    LOOKBACK = 20 # Số phiên lịch sử để hiển thị trên Web
    
    print(f"🚀 Bắt đầu quét dữ liệu Hybrid (vnstock cũ)...")
    for i, sym in enumerate(vn30):
        print(f"\r⏳ [{i+1}/30] Đang xử lý: {sym:<5}", end="")
        df = get_hybrid_data(sym)
        if df.empty: continue
        
        df_feat = compute_features(df)
        if df_feat.empty: continue
        
        # Lưu kết quả 20 phiên gần nhất
        for j in range(-LOOKBACK, 0):
            try:
                res = predict_at_index(df_feat, sym, idx=j)
                if res: final_output.append(res)
            except: continue
            
        time.sleep(1.7) # Nghỉ để không bị firewall chặn IP
        
    if final_output:
        pd.DataFrame(final_output).to_csv('vn30_signals.csv', index=False, encoding='utf-8-sig')
        print(f"\n✅ Hệ thống đã cập nhật vn30_signals.csv thành công!")



import pandas as pd
import time
import os
from datetime import datetime, timedelta
from vnstock import Vnstock

client = Vnstock()
CSV_FILE = 'vn30_data_raw.csv'

def get_new_data(symbol, start_date):
    try:
        ticker = client.stock(symbol=symbol)
        end_d = datetime.now().strftime('%Y-%m-%d')
        
        df_temp = ticker.quote.history(start=start_date, end=end_d)
        
        if df_temp is not None and not df_temp.empty:
            temp = df_temp.copy()
            
            # 1. ĐỔI TÊN trực tiếp 'time' thành 'Date' (để không bị sinh thêm cột)
            temp = temp.rename(columns={
                'time': 'Date',
                'open': 'Open', 
                'high': 'High', 
                'low': 'Low', 
                'close': 'Close', 
                'volume': 'Volume'
            })
            
            # 2. Định dạng lại ngày và gán mã chứng khoán
            temp['Date'] = pd.to_datetime(temp['Date']).dt.strftime('%Y-%m-%d')
            temp['Symbol'] = symbol
            
            # 3. Lọc bỏ dữ liệu trước 2018 nếu lỡ có bị dính
            temp = temp[temp['Date'] >= '2018-01-01']
            
            # 4. CHỈ GIỮ ĐÚNG 7 CỘT CHUẨN
            return temp[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]
        return pd.DataFrame()
    except Exception as e:
        print(f"\n❌ Lỗi mã {symbol}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    vn30_symbols = ['ACB','BCM','BID','CTG','DGC','FPT','GAS','GVR','HDB','HPG','LPB','MSN','MBB','MWG','PLX','SAB','SHB','SSB','SSI','STB','TCB','TPB','VCB','VIC','VHM','VIB','VJC','VNM','VPB','VRE']

    if os.path.exists(CSV_FILE):
        old_df = pd.read_csv(CSV_FILE)
        
        # Dọn dẹp file cũ nếu lỡ bị dư cột Date/time trước đó
        valid_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']
        
        # Nếu file cũ có 'time' mà chưa có 'Date' thì đổi tên luôn
        if 'time' in old_df.columns and 'Date' not in old_df.columns:
            old_df = old_df.rename(columns={'time': 'Date'})
            
        # Chỉ giữ lại các cột chuẩn, bỏ mọi cột rác
        old_df = old_df[[c for c in valid_cols if c in old_df.columns]]
        
        old_df['Date'] = pd.to_datetime(old_df['Date']).dt.strftime('%Y-%m-%d')
        latest_date_str = old_df['Date'].max()
        start_date_dt = datetime.strptime(latest_date_str, '%Y-%m-%d') + timedelta(days=1)
        start_date = start_date_dt.strftime('%Y-%m-%d')
    else:
        old_df = pd.DataFrame()
        start_date = "2026-01-11"

    print(f"🚀 Lấy tiếp dữ liệu từ: {start_date}")

    new_data_list = []
    for i, sym in enumerate(vn30_symbols):
        print(f"📡 {sym}...", end='\r')
        df_new = get_new_data(sym, start_date)
        
        if not df_new.empty:
            new_data_list.append(df_new)
        
        # Nghỉ đúng 2 giây mỗi mã để tránh bị chặn
        time.sleep(2)

    if new_data_list:
        all_new_df = pd.concat(new_data_list, ignore_index=True)
        final_df = pd.concat([old_df, all_new_df], ignore_index=True)
        
        # Xóa trùng và sắp xếp chuẩn
        final_df = final_df.drop_duplicates(subset=['Date', 'Symbol'], keep='last')
        final_df = final_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]
        final_df = final_df.sort_values(by=['Symbol', 'Date'])
        
        # Lưu file đè lên file cũ
        final_df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"\n✅ Cập nhật file sạch thành công (Chỉ còn 1 cột Date)!")
    else:
        print("\n☕ Không có gì mới để cập nhật.")

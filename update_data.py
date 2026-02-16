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
            # Tạo cột Date chuẩn, bỏ cột 'time' gốc của vnstock
            temp['Date'] = pd.to_datetime(temp['time']).dt.strftime('%Y-%m-%d')
            temp['Symbol'] = symbol
            temp = temp.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'})
            
            # CHỈ LẤY ĐÚNG 7 CỘT CHUẨN
            return temp[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]
        return pd.DataFrame()
    except Exception as e:
        print(f"\n❌ Lỗi mã {symbol}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    vn30_symbols = ['ACB','BCM','BID','CTG','DGC','FPT','GAS','GVR','HDB','HPG','LPB','MSN','MBB','MWG','PLX','SAB','SHB','SSB','SSI','STB','TCB','TPB','VCB','VIC','VHM','VIB','VJC','VNM','VPB','VRE']

    if os.path.exists(CSV_FILE):
        old_df = pd.read_csv(CSV_FILE)
        
        # DỌN DẸP FILE CŨ: Nếu file cũ bị dư cột, chỉ giữ lại những cột mình cần
        # Cách này sẽ bỏ cái cột "ngày đằng trước" mà bạn nói
        valid_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']
        
        # Nếu file cũ có cột 'time' mà chưa có 'Date', mình đổi tên luôn
        if 'time' in old_df.columns and 'Date' not in old_df.columns:
            old_df = old_df.rename(columns={'time': 'Date'})
            
        # Giữ lại đúng các cột cần thiết, bỏ mọi cột rác/cột trống
        existing_cols = [c for c in valid_cols if c in old_df.columns]
        old_df = old_df[existing_cols]
        
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
        if i > 0 and i % 10 == 0:
            print(f"\n⏳ Nghỉ 1,7s...")
            time.sleep(1,7)
        
        print(f"📡 {sym}...", end='\r')
        df_new = get_new_data(sym, start_date)
        if not df_new.empty:
            new_data_list.append(df_new)
        time.sleep(1.7)

    if new_data_list:
        all_new_df = pd.concat(new_data_list, ignore_index=True)
        # Gộp và làm sạch lần cuối
        final_df = pd.concat([old_df, all_new_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['Date', 'Symbol'], keep='last')
        
        # Đảm bảo thứ tự cột luôn cố định: Date đứng đầu, Symbol đứng cuối
        final_df = final_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]
        final_df = final_df.sort_values(by=['Symbol', 'Date'])
        
        final_df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"\n✅ Đã cập nhật xong file sạch!")
    else:
        print("\n☕ Không có gì mới.")

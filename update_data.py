import pandas as pd
import time
import os
from datetime import datetime, timedelta
from vnstock import Vnstock

# Khởi tạo client theo chuẩn mới nhất bạn dùng
client = Vnstock()

# --- CẤU HÌNH ---
CSV_FILE = 'vn30_data_raw.csv'

def get_new_data(symbol, start_date):
    try:
        ticker = client.stock(symbol=symbol)
        # start_date truyền vào là ngày tiếp theo sau ngày cuối trong CSV
        end_d = datetime.now().strftime('%Y-%m-%d')
        
        # Nếu ngày bắt đầu lớn hơn hôm nay thì không cần lấy
        if start_date > end_d:
            return pd.DataFrame()

        df_temp = ticker.quote.history(start=start_date, end=end_d)
        
        if df_temp is not None and not df_temp.empty:
            temp = df_temp.copy()
            temp['Date'] = pd.to_datetime(temp['time']).dt.strftime('%Y-%m-%d')
            temp['Symbol'] = symbol
            # Đổi tên khớp với file raw cũ
            temp = temp.rename(columns={'open':'Open', 'high':'High', 
                                        'low':'Low', 'close':'Close', 'volume':'Volume'})
            # Chỉ lấy các cột cần thiết để nối vào file cũ
            return temp[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
        return pd.DataFrame()
    except Exception as e:
        print(f"\n❌ Lỗi mã {symbol}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    vn30_symbols = [
        'ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
        'LPB', 'MSN', 'MBB', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
        'TCB', 'TPB', 'VCB', 'VIC', 'VHM', 'VIB', 'VJC', 'VNM', 'VPB', 'VRE']

    # 1. Đọc file cũ để tìm ngày cuối cùng
    if os.path.exists(CSV_FILE):
        old_df = pd.read_csv(CSV_FILE)
        old_df['Date'] = pd.to_datetime(old_df['Date']).dt.strftime('%Y-%m-%d')
        latest_date_str = old_df['Date'].max()
        # Ngày bắt đầu lấy mới là ngày tiếp theo
        start_date_dt = datetime.strptime(latest_date_str, '%Y-%m-%d') + timedelta(days=1)
        start_date = start_date_dt.strftime('%Y-%m-%d')
    else:
        # Nếu chưa có file thì mặc định lấy sau 10/1/2026
        old_df = pd.DataFrame()
        start_date = "2026-01-11"

    print(f"📅 Ngày cuối trong file: {latest_date_str if os.path.exists(CSV_FILE) else 'N/A'}")
    print(f"🚀 Bắt đầu lấy dữ liệu từ ngày: {start_date}")

    new_data_list = []
    
    for i, sym in enumerate(vn30_symbols):
        # Cứ sau mỗi 10 mã thì nghỉ 65 giây như bạn yêu cầu
        if i > 0 and i % 10 == 0:
            print(f"\n⏳ Đã xong {i} mã. Nghỉ 65s để tránh bị chặn...")
            time.sleep(65)
        
        print(f"📡 Đang tải: {sym}...       ", end='\r')
        
        df_new = get_new_data(sym, start_date)
        
        if not df_new.empty:
            new_data_list.append(df_new)
        
        # Nghỉ nhẹ 1.7s giữa các mã
        time.sleep(1.7)

    # 2. Nối dữ liệu và lưu
    if new_data_list:
        all_new_df = pd.concat(new_data_list, ignore_index=True)
        final_df = pd.concat([old_df, all_new_df], ignore_index=True)
        
        # Xóa trùng và sắp xếp
        final_df = final_df.drop_duplicates(subset=['Date', 'Symbol'], keep='last')
        final_df = final_df.sort_values(by=['Symbol', 'Date'])
        
        final_df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"\n✅ Thành công! Đã nối thêm {len(all_new_df)} dòng dữ liệu mới.")
    else:
        print("\n☕ Không có dữ liệu mới để cập nhật.")

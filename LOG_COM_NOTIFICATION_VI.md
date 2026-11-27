# Chế độ log: gửi tín hiệu mã mới qua COM

## Chức năng
- Chỉ trong `product.mode: log`, mỗi barcode mới trên dòng log hoàn chỉnh
  `Read Barcode : ...` gọi `COMOutput.send_new_code()`.
- Payload chính xác: ASCII `n` + CR + LF, tức `b"n\\r\\n"` (hex `6E 0D 0A`).
- Dùng chung cổng `com_output.port`, baudrate và cơ chế retry hiện có.
- Quét lại cùng barcode vẫn gửi tín hiệu mới; nhiều dòng trong một lần poll
  được thông báo lần lượt, không chỉ lấy dòng cuối.
- Khi đang đợi CSV hợp lệ ở khởi động, dòng barcode mới vẫn được thông báo.
  Tín hiệu xác nhận đã đọc barcode, KHÔNG xác nhận CSV hợp lệ hoặc kết quả OK.
- Không gửi thông báo cho lịch sử log đã có lúc khởi động. Chỉ khôi phục
  mã sản phẩm cuối cùng để giữ chức năng cũ.
- Dòng chưa có LF được giữ lại đến khi ghi hoàn chỉnh.
- Chuyển file theo ngày trong lúc chạy: đọc và thông báo các dòng trong file mới.
- Kết quả kiểm tra qua COM, reset batch và loại bỏ kết quả AI cũ giữ nguyên.
- `com_output.enabled: false` không gửi. Khi COM lỗi, cảnh báo và tiếp tục
  theo cơ chế xử lý lỗi sẵn có, không coi đó là đã xác nhận thiết bị nhận được.

## Cấu hình
Sửa các trường dưới đây trong `config/config.yaml` hiện có, giữ nguyên các
phần cấu hình khác. COM5 và 9600 chỉ là ví dụ, cần dùng giá trị của thiết bị:

```yaml
product:
  mode: log
  log_root_dir: 'D:/anh'
  log_filename: WH_AVIS_System.log
  log_date_folder_format: '%Y%m%d'
  log_poll_interval: 0.5

com_output:
  enabled: true
  port: COM5
  baudrate: 9600
  retry_count: 3
```

Gói nguồn gốc không có `config/config.yaml`; bản cập nhật không tự tạo hoặc
ghi đè cấu hình triển khai của bạn. Sao lưu và giữ lại thư mục config/model.

## File thay đổi
- `main.py`: mở COM trước reader, kết nối callback nhận barcode với gửi `n`.
- `modules/log_input.py`: callback theo từng scan, xử lý dòng ghi dở.
- `modules/com_output.py`: thêm `send_new_code()`.
- `tests/test_log_com_notification.py`: kiểm thử mới.
- Hai test harness có sẵn: bổ sung biến CAPTURE_STATS_INTERVAL bị thiếu
  trong môi trường AST mô phỏng; không thay logic kiểm tra.
- `LOG_COM_TEST_RESULTS.txt`: kết quả kiểm thử offline.

## Kiểm thử
```bash
python -m unittest discover -s tests -v
```

COM được mô phỏng để kiểm tra đúng byte, chưa kiểm chứng phần cứng Windows/PLC.
Không có ACK giao thức từ thiết bị; retry khi lỗi có thể tạo thông báo trùng.
Polling chỉ thấy dữ liệu còn tồn tại khi đọc; không đảm bảo bắt được file bị
ghi đè/xóa quá nhanh. Gửi COM đồng bộ trên main thread như kết quả kiểm tra
hiện có; timeout/retry có thể làm tăng thời gian phản hồi GUI/log.

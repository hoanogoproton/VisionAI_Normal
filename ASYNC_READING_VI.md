# Triển khai thu nhận ảnh độc lập với AI

Bản sửa này dựa trên mã nguồn AiVisionUPLOAD.zip bạn cung cấp. Ba ZIP nguồn có
main.py giống nhau. Không dùng watchdog. Chưa kiểm thử trên phần mềm tạo ảnh,
Windows, model, COM/PLC hay cấu hình sản xuất thực tế của bạn.

## 1. Luồng mới

```text
Mỗi thư mục input (một worker thu nhận riêng):
    os.scandir → chọn phiên bản mới → mở nguồn chia sẻ → đọc bytes → đóng nguồn
    → FIFO RAM có giới hạn

Một worker AI dùng ThreadPoolExecutor(max_workers=1):
    lấy packet → kiểm tra marker → cv2.imdecode → ACK hoặc NACK
    → chạy logic YOLO/ROI/đo dây hiện có

Luồng chính:
    cập nhật GUI → kiểm tra Future.done() → nhận kết quả → giữ trình tự GUI/COM
```

Decode và inference chạy cùng executor một luồng, không đồng thời sử dụng cache
model. GUI chỉ gọi result() sau done(); không đợi một future đang chạy. Thu nhận
tiếp tục khi AI đang xử lý. Ghi log, lưu kết quả, gửi COM và đọc barcode hiện vẫn
có thao tác đồng bộ trên luồng chính; chúng không giữ handle ảnh nguồn. Đây không
phải bảo đảm thời gian thực hoặc GUI không bao giờ bị chậm bởi I/O khác.

## 2. File cần thay đồng bộ

- main.py: coordinator bất đồng bộ, FIFO theo camera, xử lý scan/product epoch,
  suppress kết quả job cũ, kiểm tra lỗi thu nhận và shutdown.
- modules/image_watcher.py: worker riêng, một lượt scandir, lịch quét thật, RAM
  FIFO, ACK/NACK, retry hữu hạn, thống kê, lỗi quá tải được giữ đến khi dừng.
- modules/shared_image_reader.py: CreateFileW với GENERIC_READ và
  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, OPEN_EXISTING;
  đóng fd trong finally trước khi giải mã; metadata Win32 trên chính handle.
- modules/ai_worker.py: job giải mã và inference, không gọi GUI/COM hay mở nguồn.
- modules/config_loader.py: bổ sung cấu hình acquisition mặc định.
- modules/log_input.py: chỉ thay cách mở log sang shared handle. Không thiết kế
  lại parser, barcode normalization, polling hoặc event semantics.
- main.spec: bổ sung hiddenimports.
- ACQUISITION_CONFIG_SNIPPET.yaml, tests/, tools/probe_shared_read.py:
  cấu hình mẫu, kiểm thử và probe không sử dụng AI/COM/PLC.
- LOG_IMAGE_RESET.md, README.md: cập nhật tài liệu.

Đã đối chiếu AST: run_detection_for_cam, emit_batch_result,
try_load_product_config, merge_cam_results, merge_updates_for_ng_rois,
get_ng_cameras và append_history_csv giữ nguyên logic so với nguồn ban đầu.
Tuy nhiên lịch xử lý đã thay đổi, nên vẫn cần kiểm tra tín hiệu thực tế.

API ImageWatcher đã đổi: dữ liệu là ImagePacket/bytes, không còn temp_path.
Không được chỉ thay ImageWatcher mà giữ main.py cũ.

## 3. Cài đặt an toàn

1. Sao lưu phần mềm hiện tại và giải nén vào thư mục thử nghiệm riêng.
2. ZIP nguồn KHÔNG có config/config.yaml, camera_config.csv, CSV sản phẩm,
   models/*.pt hay EXE. Giữ/copy cấu hình và model thực tế của bạn vào bản thử.
   Không tạo tham số camera/ROI/COM từ các ví dụ thay cho cấu hình sản xuất.
3. Gói đầy đủ chứa mã gốc cộng bản sửa. Gói patch chứa các file cần thay/bổ sung.
   Copy toàn bộ nội dung AiVisionUPLOAD/ trong patch vào thư mục dự án, giữ
   cấu trúc modules/, tests/, tools/. Không trộn với bản shared_read cũ.
4. Ghép mục acquisition ở ACQUISITION_CONFIG_SNIPPET.yaml vào config hiện có.
   Không thêm thì dùng mặc định. acquisition.poll_interval điều khiển reader;
   camera.poll_interval cũ không còn được dùng cho thu nhận.
5. output_dir và log_dir phải nằm ngoài cây thư mục input. Bản sửa không tạo
   ảnh tạm. camera_config_loader có thể vẫn tạo thư mục theo cấu hình cũ,
   nhưng bộ thu nhận không copy hoặc cleanup ảnh tại đó.
6. Chạy probe trước, sau đó thử tích hợp trong chế độ không nối/phát COM/PLC.
   Chỉ đưa lên dây chuyền sau khi kiểm tra batch, OK/NG và tín hiệu thực tế.

Từ thư mục dự án:

```powershell
python -m unittest discover -s tests -v

python tools/probe_shared_read.py --folder "D:\anh\CAM1" --seconds 60

# Thử backlog khi consumer chậm, không chạy model hay phát COM:
python tools/probe_shared_read.py --folder "D:\anh\CAM1" --seconds 60 --simulate-ai-ms 100

python main.py
```

Probe bỏ qua ảnh đã có khi khởi tạo. Cần ảnh MỚI xuất hiện khi probe chạy. Nó
không ghi/xóa/đổi tên nguồn, không gọi main.py, không dùng cấu hình COM hoặc model.
Tất cả kết quả probe được in ra console; có thể lưu console vào một vị trí riêng.

Nếu dùng EXE, phải build lại trên Windows với dependencies/config/models thực tế:

```powershell
pyinstaller --clean main.spec
```

Spec gốc yêu cầu các thư mục config, models, images và output tại máy build.
Gói này không có EXE đã build. Thay .py không cập nhật EXE cũ.

## 4. Tham số khởi đầu

```yaml
acquisition:
  poll_interval: 0.02
  queue_max_images: 32
  queue_max_mb: 128
  max_image_mb: 32
  retry_delays_ms: [5, 10, 20, 40]
  retry_timeout_ms: 250
  require_complete_markers: true
  max_decoded_pixels: 40000000
```

- Poll 20 ms là giá trị khởi đầu, KHÔNG bảo đảm lấy đủ ảnh. Đo thời gian ảnh tồn
  tại và tải ổ đĩa/CPU rồi điều chỉnh. Nếu scan/read lâu, không chạy bù thành burst.
- Giới hạn mỗi thư mục: 32 packet, 128 MiB bytes ảnh mã hóa, gồm packet đã lấy
  khỏi queue nhưng chưa ACK/NACK. Không phải giới hạn RAM toàn tiến trình.
  Còn buffer đọc, ảnh giải mã, ROI/GUI/history, job AI, model và số camera.
- max_decoded_pixels được kiểm tra sau imdecode, không chặn mọi cấp phát lớn
  trong decoder. Chỉ dùng nguồn ảnh tin cậy của hệ thống, kích thước đã biết.
- Một lần đầu + tối đa 4 retry cho phiên bản không đổi; dừng khi hết số lần
  hoặc quá 250 ms. Khoảng retry thực tế được làm tròn theo lịch quét và thời
  gian I/O, không phải deadline thời gian thực.
- Metadata đổi tạo phiên bản mới với ngân sách retry mới.
- File quá lớn hoặc access denied được báo lỗi và không retry vô hạn.
  Sharing/lock violation (Windows 32/33) có retry hữu hạn.
- JPEG/PNG marker là kiểm tra sơ bộ, không chứng minh ảnh ghi xong hay nội
  dung toàn vẹn. JPEG có SOI và EOI ở đâu đó trong dữ liệu vẫn cần imdecode.

## 5. Quá tải và mất ảnh

Nếu FIFO không còn chỗ cho phiên bản ảnh mới:
- Báo OVERLOAD và giữ lỗi, dừng worker thu nhận.
- Luồng chính fail-stop, không tạo kết quả OK giả; exit code 2.
- Không dùng drop-oldest hoặc ghi đè packet.
- Packet chưa xử lý lúc shutdown được ghi nhận DISCARD.
- Producer/phần mềm kia KHÔNG bị tự động dừng. Không có handshake dừng PLC,
  persistent queue hay cơ chế phục hồi toàn bộ ảnh sau crash trong bản này.

Log cần theo dõi:
- CAPTURE STATS / CAPTURE FINAL: scans, captured, accepted, read_errors,
  decode_errors, retries, missing, exhausted, queue_items, queue_bytes.
- read_ms_total / read_ms_max: tổng và lớn nhất của thời gian read attempt.
- MISSING_OR_REPLACED, READ_ERROR, DECODE_ERROR, RETRY_EXHAUSTED.
- OVERLOAD, WORKER_ERROR, STOP_TIMEOUT.
- BOUNDARY/DISCARD: ảnh bị loại theo chính sách reset/shutdown được báo rõ.

Không thể biết số ảnh xuất hiện rồi biến mất hoàn toàn giữa hai scan. Counter
missing chỉ đếm một số phiên bản từng quan sát nhưng không thu nhận lại được.
Không diễn giải captured/accepted như số sản phẩm vật lý.

## 6. Giới hạn cần kiểm tra khi vận hành

- Handle chia sẻ xóa giảm việc cản producer. Không vượt qua handle producer mở
  độc quyền, không làm ghi đồng thời trở thành atomic, không giữ ảnh tồn tại
  trước khi mở được. Nguồn ảnh lỗi/mất vẫn có thể xảy ra.
- Windows dùng GetFileInformationByHandle trước/sau, không so toàn bộ CRT fstat
  với os.stat. So scan/open chỉ dùng mtime+size chung; ảnh thay thế có cùng
  metadata vẫn có thể không được phân biệt. Không bảo đảm exactly-once.
- FIFO theo thứ tự thu nhận từng camera, không phải thứ tự sản phẩm toàn cục.
  File cùng scan được sắp first_seen/mtime/name. Không có frame/batch ID, mất
  ảnh ở một camera có thể khiến ghép batch sai giữa các camera.
- Trong redetect, ảnh xếp hàng của camera NG được dùng FIFO; ảnh có thể đã được
  chụp trong lúc inference trước còn chạy. Phải kiểm tra phù hợp với trigger
  dây chuyền. Nếu cần chỉ nhận ảnh chụp sau NG, phải có ID/trigger rõ ràng,
  không âm thầm bỏ packet theo timestamp.
- Ảnh camera đã OK vẫn có thể tích trong FIFO trong khi chờ camera NG; nếu
  producer liên tục phát, có thể quá tải. Không âm thầm bỏ chúng.
- Giữ chính sách gốc: accepted scan đổi mã/repeat log scan bỏ ảnh cũ và bỏ
  file đang có trong input. Reset có generation/epoch để job cũ không phát
  kết quả sang sản phẩm mới. Ở mode log sau OK chờ scan, ảnh mới vẫn được
  thu nhận nhưng có thể được reset loại ở scan tiếp theo; thao tác này có log.
- Không hỗ trợ hai camera trỏ cùng một folder nếu chưa có quy tắc định tuyến
  tên ảnh. Chia sẻ giữa phần mềm bạn và phần mềm tạo ảnh thì được hỗ trợ.
- Threads không có deadline cứng. I/O đồng bộ bị OS/network giữ có thể chưa
  dừng được sau join timeout; AI job đang chạy được chờ khi shutdown.
- Kết quả cũ bị suppress khi epoch mới đã được coordinator chấp nhận.
  Không có giao dịch atomic với sự kiện scan/COM bên ngoài.
- Muốn bảo đảm không mất ảnh: producer ghi xong, đóng file, bàn giao bằng
  tên/ID riêng và giữ đến khi bên nhận ACK. Bản consumer-only không thay thế
  được giao thức này.

## 7. Kết quả kiểm thử

Xem TEST_RESULTS.txt. Môi trường sandbox là Linux:
- 30 bài: 26 đạt, 4 bỏ qua vì cần Windows native.
- Kiểm tra source deleted-before-decode, ba ảnh FIFO, ghi đè cùng tên, ảnh
  chưa hoàn chỉnh, metadata đổi, retry sharing giả lập, biến mất, quá tải
  theo số ảnh/bytes, reset giữa read và lịch quét.
- main() được kiểm thử bằng AST thật với GUI/model/COM/reader giả lập:
  FIFO hai camera, GUI tick khi inference chậm, đổi product/repeat scan và
  giữ ROI OK khi redetect. Không phải thử GUI/YOLO/COM thật.
- Probe thử với 5 ảnh PNG nhỏ và consumer chậm: captured=accepted=5,
  không overload. Đây không phải benchmark throughput máy sản xuất.

Trên Windows cần chạy 4 test đang skip: xóa khi handle đọc mở, đổi tên khi
handle đọc mở, producer độc quyền và so token scan với native metadata.
Sau đó dùng Process Monitor theo dõi thư mục nguồn và tiến trình producer:
SHARING VIOLATION ở thao tác xóa/đổi tên phải được kiểm tra trước/sau sửa.

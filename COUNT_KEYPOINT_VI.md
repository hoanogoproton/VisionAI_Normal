# Count + keypoint angle

## Cách bật
Dùng `check_mode=count` và điền đủ 4 cột:
`keypoint_idx_1`, `keypoint_idx_2`, `expected_angle`, `angle_tolerance`.
Không cần mode mới và không cần thêm cột CSV.
Ví dụ đầy đủ nằm trong `examples/count_keypoint.csv`.
Thay camera, tên model, class_id, confidence và tọa độ ROI bằng cấu hình thực tế.

- Chỉ đếm detection có tâm bbox nằm trong compare_roi (bao gồm biên).
- Detection đã được lọc class/confidence và NMS bởi detector hiện có.
- Kiểm tra góc TỪNG vật thể được đếm, không chỉ confidence cao nhất.
- OK = số lượng đạt min_count/max_count VÀ mọi vật thể được đếm đều đạt góc.
- Vật thể sai góc vẫn được tính vào số lượng (không bỏ qua để đạt count).
- Vật thể ngoài compare_roi không được đếm/kiểm tra góc.
- Thiếu keypoint, index ngoài phạm vi, hai điểm trùng nhau hoặc tọa độ
  không hữu hạn => NG khi bật kiểm tra góc.
- Thiếu một phần cấu hình góc => INVALID_ANGLE_CONFIG, không bỏ qua âm thầm.
- Góc theo tọa độ ảnh: 0° sang phải, 90° xuống dưới, 180° sang trái,
  270° lên trên; hướng từ keypoint_idx_1 đến keypoint_idx_2.
- Chỉ số keypoint bắt đầu từ 0.
- angle_tolerance trong [0, 180]; expected_angle được chuẩn hóa modulo 360.
- Tất cả 4 cột góc trống: giữ nguyên count-only.
- min_count trống mặc định 1; max_count trống không giới hạn trên.
- min_count=0 cho phép vùng trống; nếu cần ít nhất một vật thể dùng min_count=1.
- check_mode=compare/forbidden/measure_width giữ nguyên luồng hiện có;
  compare_angle được bổ sung kiểm tra đầu vào không hợp lệ.
- Ngưỡng confidence của KEYPOINT không được thêm trong bản sửa này.
  Độ tin cậy vật thể vẫn dùng cột confidence hiện có.

## Ví dụ yêu cầu đúng 1 vật thể
check_mode=count, min_count=1, max_count=1,
keypoint_idx_1=0, keypoint_idx_2=3, expected_angle=120, angle_tolerance=20.

1 vật thể đúng góc: OK.
1 vật thể sai góc/thiếu keypoint: NG.
2 vật thể đều đúng góc: NG vì dư số lượng.
0 vật thể: NG.

Nếu cần đúng 2 vật thể, đổi min_count=max_count=2. Khi đó cả hai phải đạt góc.

## Hiển thị và dữ liệu
GUI và ảnh batch hiển thị góc/OK-NG riêng cho từng vật thể.
Kết quả count có `detected_count` và `angle_results`.
`angle_results` chứa object_index (đánh số theo thứ tự detection trong ROI),
passed, reason, angle_info và detection.

Lưu ý: label OK/NG trên từng vật thể là kết quả góc của riêng vật thể đó.
ROI vẫn có thể NG do thiếu/dư số lượng dù các label vật thể đều OK.
Nếu count sai, reason chính là COUNT_NG; chi tiết góc vẫn được tính và log.

## Cài bản sửa
Giải nén bản đầy đủ hoặc chép 5 file từ bản patch vào dự án hiện tại:
main.py, modules/comparator.py, modules/result_gui.py,
modules/result_visualizer.py, modules/csv_loader.py. Sao lưu trước khi thay.
Dùng mã nguồn Python; nếu đang chạy .exe phải build lại bằng cấu hình hiện có.
Không cần thay models hoặc thư viện.

## Test
Chạy từ thư mục dự án:
```bash
python -m unittest discover -s tests -p "test_count_keypoint.py" -v
python -m unittest discover -s tests -v
```

Bài test count/keypoint dùng detection giả lập để kiểm tra logic và luồng main;
không cần tải YOLO hoặc model thật. Cần các dependency cho test ảnh hiện có.
Chưa xác nhận inference model thật, camera/COM hoặc đóng gói .exe trong sandbox.


## Kết quả kiểm tra trong sandbox
- 30 test mới count/keypoint: PASS.
- Kiểm tra tọa độ overlay + cả GUI và ảnh batch: PASS (mock drawing calls).
- Toàn bộ 60 test: 50 PASS, 6 FAIL, 4 SKIP.
- Bản ZIP gốc: 30 test, 20 PASS, 6 FAIL, 4 SKIP.
- Cùng 6 test acquisition/state-machine thất bại ở cả hai bản.
  Không sửa các lỗi ngoài phạm vi count/keypoint trong bản này.
- 4 test Windows sharing bị bỏ qua vì sandbox không chạy Windows.
- Xem COUNT_KEYPOINT_TEST_RESULTS.txt để biết chi tiết.

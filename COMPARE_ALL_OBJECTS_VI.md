# Compare: tất cả object + min_count / max_count

## Logic mới
- Giữ nguyên detector: lọc class/confidence và NMS trước khi compare.
- Xét tất cả detection, không chọn confidence cao nhất.
- Chỉ đếm object có tâm bbox trong compare_roi (bao gồm biên).
- Object ngoài compare_roi không được đếm hoặc kiểm tra góc.
- min_count mặc định 1; max_count trống là không giới hạn.
- Kiểm tra giới hạn số lượng bao gồm cả hai đầu.
- Nếu cả 4 trường góc trống: chỉ kiểm tra số lượng/vị trí.
- Nếu có cấu hình góc: cần đủ 4 trường và tất cả object được đếm phải đạt góc.
- Object sai góc vẫn được tính vào detected_count.
- Nếu số lượng NG, vẫn tính góc từng object cho debug; giữ COUNT_NG làm reason chính.
- min_count=0 cho phép không có object nếu cấu hình hợp lệ.
- Cấu hình số lượng/góc sai hoặc bbox không hợp lệ trả NG.

## Thay đổi
- main.py: compare và count dùng chung compare_count_and_angle;
  log số lượng/góc từng object; đưa detected_count và angle_results vào kết quả/GUI.
- batch_detect_folder.py: compare dùng cùng logic và cung cấp dữ liệu overlay từng object.
- modules/comparator.py: compare_detection xét tất cả object và nhận min_count/max_count,
  giữ interface trả về 2 giá trị. compare_angle là helper cũ;
  compare mới gọi helper này riêng cho từng object, không truyền danh sách nhiều object.
- examples/compare_count_keypoint.csv: ví dụ CSV.
- tests/test_compare_all_objects.py: kiểm thử comparator, main và batch bằng mock.
- GUI và result_visualizer đã hỗ trợ angle_results; không cần sửa hai module này.

## Cài đặt
Giải nén bản ZIP mới, hoặc thay main.py, batch_detect_folder.py,
modules/comparator.py trong project đang dùng.
Không cần thay cấu trúc CSV: min_count/max_count đã có parser.
Thay đổi logic compare có chủ đích: không còn reason NOT_FOUND/OUT_OF_COMPARE_ROI
cho tình huống không có tâm object trong vùng; nay trả COUNT_NG.
Bản ZIP giữ nguyên các file khác, bao gồm tài liệu/patch lịch sử.

## Kiểm thử
Chạy từ thư mục AiVisionUPLOAD:

```bash
python -m unittest discover -s tests -v
```

87 tests: 83 passed, 4 skipped. Kiểm tra cú pháp tất cả file Python thành công.
Kiểm thử dùng dữ liệu giả và mock model/camera/GUI/COM; chưa chạy model thật
hoặc kiểm tra trực quan GUI trên máy đích.

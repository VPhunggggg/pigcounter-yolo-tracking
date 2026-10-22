# PigCounter — YOLO-Based Pig Counting & Tracking

Ứng dụng thị giác máy tính chạy local để phát hiện, theo dõi và đếm heo từ
video hoặc camera RTSP. Hệ thống kết hợp mô hình YOLO, tracker ByteTrack,
logic đếm hai chiều qua hai cổng, vùng ROI vẽ trực tiếp trên giao diện web
và báo cáo CSV theo lượt/sự kiện.

Lõi đếm được chuyển từ script `pig_counter_dual_gate_v5_1.py` gốc sang
Flask, có xử lý đảo chiều khi heo quay đầu và điều chỉnh vùng đếm khi đang chạy.

## Lấy mã nguồn

```bash
git clone https://github.com/VPhunggggg/pigcounter-yolo-tracking.git
cd pigcounter-yolo-tracking
```

Repo có sẵn mô hình `best.pt`; không cần tải trọng số riêng.

## Cách chạy (Windows)

1. Mở thư mục dự án vừa clone hoặc giải nén.
2. Chạy **`install.bat`** một lần để tạo môi trường ảo và cài thư viện. Bước
   này có thể mất 10-20 phút tuỳ mạng (tải PyTorch/Ultralytics).
3. Bấm đúp vào **`run.bat`** mỗi khi muốn khởi động ứng dụng.
4. Trình duyệt tự mở tại `http://127.0.0.1:5000`.

Yêu cầu máy đã cài sẵn **Python 3.9+** (tick "Add Python to PATH" lúc cài).

Trên Linux/macOS, chạy `bash run.sh` từ thư mục dự án; script sẽ tạo môi
trường ảo, cài thư viện và khởi động server.

## Cách dùng trong web

1. **Chọn nguồn**: tải video lên (kéo-thả hoặc bấm chọn file), hoặc dán link
   camera RTSP rồi bấm "Kết nối".
2. **Vẽ vùng đếm bằng tay** ngay trên khung hình thật của video/camera:
   - Vẽ vùng chuồng (ROI) — đa giác tự do, nhấp từng điểm, nhấp lại điểm đầu
     (hoặc bấm "Đóng vùng ROI") để khép kín.
   - Vẽ cổng vào (đường xanh lá) — 2 điểm.
   - Vẽ cổng ra (đường hồng) — 2 điểm.
   - Không còn toạ độ mặc định cứng — mọi video/camera đều phải vẽ tay theo
     đúng vị trí thực tế.
3. Bấm **"Bắt đầu đếm"**. Video (hoặc luồng RTSP) chạy kèm bảng thống kê theo
   đợt (lượt) ngay bên cạnh: heo trong chuồng, tổng heo đã đếm, số heo mỗi
   lượt.
4. **Nút bánh răng (⚙️)** ở góc video: cho phép tạm dừng và **vẽ lại vùng
   ROI/cổng** ngay trên khung hình mới nhất, **không cần tải lại video** —
   sau khi "Áp dụng" hệ thống chạy tiếp với vùng vẽ mới.
5. Khi video kết thúc (hoặc bấm Dừng với RTSP), có thể tải video đã annotate,
   CSV theo lượt (`_batches.csv`) và CSV từng sự kiện qua vạch (`_events.csv`).

## Thuật toán đếm

Giữ nguyên logic đã kiểm chứng từ bản gốc `v5.1`:

- Đếm dựa trên **crossing vật lý qua line**, không phụ thuộc track ID cố định
  → heo quay đầu (tracker đổi ID) vẫn bị trừ đúng.
- Ngưỡng xác nhận chiều thuận (+1) chặt hơn chiều ngược (-1) vì lúc quay đầu
  hay chen nhau dễ bị che khuất, confidence thấp hơn.
- Cooldown chỉ chặn sự kiện **cùng chiều** lặp do rung nhiễu, không chặn sự
  kiện ngược chiều.
- 1 lượt (batch) tự kết thúc khi chuồng trống liên tục 3 giây.

Khác với bản gốc: vùng chuồng và 2 cổng giờ là **đa giác/đường do người dùng
vẽ tay** thay vì toạ độ chuẩn hoá hard-code trong code.

## Kết nối camera RTSP

Dùng đúng kỹ thuật đã kiểm chứng bên RockApp:
- Set `OPENCV_FFMPEG_CAPTURE_OPTIONS` (rtsp_transport=tcp, buffer lớn,
  timeout hợp lý) trước khi import `cv2` để FFmpeg backend nhận đúng cấu hình.
- Luồng RTSP chạy liên tục trong thread riêng, tự động reconnect khi rớt kết
  nối.
- Xem trực tiếp qua MJPEG (`multipart/x-mixed-replace`) — nhẹ, không cần
  WebRTC, chạy được trên mọi trình duyệt.

## Chạy tốt trên máy KHÔNG có GPU (chỉ CPU)

Vài chỉnh sửa/tuỳ chọn để giảm tải CPU cho máy khách không có GPU:

1. **Tracker mặc định đã đổi sang `bytetrack.yaml`** (nhẹ hơn `botsort.yaml`
   vì không chạy camera-motion-compensation bằng optical flow mỗi frame —
   không cần thiết với camera cố định). Muốn quay lại botsort: set biến môi
   trường `PIG_TRACKER=botsort.yaml` trước khi chạy `run.bat`.

2. **Hạ kích thước ảnh suy luận** nếu vẫn còn nặng: set
   `PIG_IMGSZ=480` hoặc `PIG_IMGSZ=416` (mặc định 640). Đổi lấy 1 chút độ
   chính xác với vật thể nhỏ/xa để lấy tốc độ.

3. **Export model sang OpenVINO** — cách tăng tốc CPU lớn nhất (thường
   nhanh hơn 2-4 lần so với chạy .pt bằng PyTorch thuần trên CPU). Chạy
   LỆNH SAU 1 LẦN trên máy khách (cần cài thêm gói, xem bên dưới):
   ```
   pip install openvino
   python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='openvino', imgsz=640, half=False)"
   ```
   Lệnh trên tạo ra thư mục `best_openvino_model/` cạnh `best.pt`.
   App sẽ **tự động phát hiện và dùng thư mục này** ở lần chạy kế tiếp
   (xem `load_model()` trong `engine.py`) — không cần sửa gì thêm.
   Nếu muốn quay lại dùng `.pt` gốc, chỉ cần xoá/đổi tên thư mục
   `best_openvino_model/` đi.

4. **Phân bổ CPU cho ứng dụng**: set
   `PIG_TORCH_THREADS=<số nhân chia đều cho mỗi job>` (vd máy 8 nhân chạy
   2 phiên ứng dụng riêng thì set `PIG_TORCH_THREADS=4`), tránh các tiến
   trình giành hết nhân CPU của nhau. Trong mỗi phiên ứng dụng, chỉ có một
   job đếm chạy tại một thời điểm; bắt đầu job mới sẽ dừng job cũ.

## Cấu trúc thư mục

```
pigcounter-yolo-tracking/
  app.py            Flask server (routes, RTSP snapshot, job control)
  engine.py         Lõi thuật toán đếm heo (port từ script gốc, tổng quát hoá)
  best.pt           Model YOLO đã train sẵn (copy từ bản gốc)
  requirements.txt
  run.bat / run.sh  Chạy 1 cú nhấp
  install.bat       Cài môi trường lần đầu
  templates/index.html
  static/css/style.css
  static/js/app.js
  uploads/          Video người dùng tải lên
  outputs/          Video đã annotate + CSV kết quả
  state/            Phiên gần nhất và cấu hình vùng đếm (tạo khi chạy)
```

Các thư mục runtime (`uploads/`, `outputs/`, `state/`, `jobs/`) và môi
trường ảo được loại khỏi Git; ứng dụng tạo các thư mục cần thiết khi chạy.

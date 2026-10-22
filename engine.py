"""
engine.py — Lõi thuật toán đếm heo (port + tổng quát hoá từ
pig_counter_dual_gate_v5_1.py) để dùng trong web app.

Khác biệt so với bản gốc:
- Không còn toạ độ hard-code. Toàn bộ hình học (vùng chuồng ROI, cổng vào
  GREEN, cổng ra PINK) được truyền vào từ ngoài (người dùng vẽ tay trên canvas)
  dưới dạng toạ độ chuẩn hoá (0..1).
- Hỗ trợ cập nhật lại hình học "nóng" (hot-reload) trong lúc đang chạy, để
  phục vụ nút "Cài đặt / vẽ lại vùng" mà không cần upload lại video.
- Chạy được cả với file video lẫn luồng RTSP (is_live) liên tục.
- Expose trạng thái + frame đã annotate ra ngoài qua callback, để Flask có
  thể poll số liệu và stream MJPEG.
"""

from __future__ import annotations

import csv
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ultralytics import YOLO


def load_model(model_path):
    """Nạp model YOLO 1 LẦN DUY NHẤT (dùng chung cho mọi job) để tránh phải
    tải lại model (tốn vài giây tới cả phút) mỗi khi bấm "Bắt đầu đếm".

    FIX CPU-PERF: nếu cạnh model_path (vd best.pt) đã có sẵn bản export
    OpenVINO (thư mục "<ten>_openvino_model/") hoặc ONNX ("<ten>.onnx"),
    TỰ ĐỘNG dùng bản đó thay vì .pt gốc. Trên CPU, chạy qua OpenVINO/ONNX
    Runtime thường nhanh hơn PyTorch thuần từ 2-4 lần vì các engine này tối
    ưu riêng cho suy luận trên CPU (PyTorch vốn tối ưu chính cho GPU).
    API model.track(...) dùng y hệt, không cần đổi gì thêm trong engine.
    Không tự convert ở đây vì việc export cần cài thêm gói 'openvino'/'onnx'
    và có thể mất vài phút — xem hướng dẫn export 1 lần trong README.
    """
    model_path = Path(model_path)
    ov_dir = model_path.with_name(model_path.stem + "_openvino_model")
    onnx_path = model_path.with_suffix(".onnx")

    if ov_dir.is_dir():
        chosen = ov_dir
        print(f"[MODEL] Tim thay ban OpenVINO da export tai {ov_dir} — "
              f"dung ban nay de chay nhanh hon tren CPU (thay vi {model_path.name}).")
    elif onnx_path.is_file():
        chosen = onnx_path
        print(f"[MODEL] Tim thay ban ONNX da export tai {onnx_path} — "
              f"dung ban nay de chay nhanh hon tren CPU (thay vi {model_path.name}).")
    else:
        chosen = model_path

    print(f"[MODEL] Dang nap model AI tu: {chosen}")
    t0 = time.time()
    model = YOLO(str(chosen))
    pig_class_id = find_pig_class(model)
    print(f"[MODEL] Da nap xong model ({time.time() - t0:.1f}s). Class heo id={pig_class_id}.")
    _warmup_model(model, pig_class_id)
    return model, pig_class_id


def _warmup_model(model, pig_class_id):
    """"Làm nóng" model bằng vài lần suy luận trên khung hình giả (đen) ngay
    lúc khởi động server. Lý do: chỉ NẠP model (YOLO(path)) là chưa đủ — lần
    suy luận (inference) ĐẦU TIÊN trên GPU/CPU luôn chậm hơn hẳn các lần sau
    (do CUDA/cuDNN phải biên dịch & tối ưu kernel lần đầu). Nếu không làm ở
    đây, chi phí "khởi động nguội" đó sẽ dồn vào đúng video ĐẦU TIÊN người
    dùng tải lên -> khiến video đầu luôn có độ trễ lâu hơn các lần sau."""
    try:
        device = 0 if (torch is not None and torch.cuda.is_available()) else "cpu"
        dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
        t0 = time.time()
        for _ in range(2):
            model.predict(dummy, imgsz=IMGSZ, conf=TRACK_CONF, iou=IOU, device=device, verbose=False)
        print(f"[MODEL] Da 'lam nong' model xong ({time.time() - t0:.1f}s). "
              f"Video dau tien gio se khong con bi cham nua.")
    except Exception as exc:  # noqa: BLE001
        print(f"[MODEL] Khong the 'lam nong' model truoc (khong nghiem trong, "
              f"video dau co the van hoi cham mot chut): {exc}")

# ============================================================
# Ngưỡng mặc định (giữ nguyên từ bản v5.1 đã kiểm chứng)
# ============================================================

COUNT_CONF_PLUS = 0.45
COUNT_CONF_MINUS = 0.25
TRACK_CONF = 0.15
IOU = 0.70

# FIX CPU-PERF: BoT-SORT (tracker mặc định trước đây) bật sẵn "camera motion
# compensation" (GMC) chạy sparse optical-flow trên TOÀN BỘ khung hình mỗi
# frame — chi phí này khá nặng trên CPU và HOÀN TOÀN không cần thiết cho
# camera đứng yên cố định (không pan/tilt) như trong use-case đếm heo.
# ByteTrack (chỉ dùng Kalman filter + IoU matching, không GMC, không ReID)
# nhẹ hơn đáng kể trên CPU mà vẫn giữ đúng logic gán/track_id cho phần đếm
# cắt vạch bên dưới (không đổi API, không đổi field nào khác trong engine).
# Có thể ép quay lại botsort bằng biến môi trường PIG_TRACKER=botsort.yaml
# nếu sau này cần track qua occlusion tốt hơn và có dư CPU/GPU.
TRACKER = os.environ.get("PIG_TRACKER", "bytetrack.yaml")

# FIX CPU-PERF: cho phép hạ kích thước ảnh đưa vào model qua biến môi trường
# khi chạy máy chỉ có CPU (vd PIG_IMGSZ=480 hoặc 416) để đổi lấy tốc độ,
# đánh đổi 1 phần độ chính xác với vật thể nhỏ/xa. Mặc định giữ nguyên 640
# để không đổi hành vi cho máy đã dùng ổn định trước đó.
IMGSZ = int(os.environ.get("PIG_IMGSZ", "640"))

# FIX CPU-PERF: mặc định để PyTorch tự dùng hết số nhân CPU sẵn có (giữ
# nguyên hành vi cũ). Nếu định chạy NHIỀU camera đồng thời trên cùng 1 máy
# CPU, nên set PIG_TORCH_THREADS=<số nhân chia đều cho từng job> để các job
# không giành hết cache/nhân CPU của nhau, tránh làm CHẬM ĐI cả hệ thống dù
# tưởng chạy song song sẽ nhanh hơn.
_TORCH_THREADS = os.environ.get("PIG_TORCH_THREADS")
if torch is not None and _TORCH_THREADS:
    try:
        torch.set_num_threads(max(1, int(_TORCH_THREADS)))
        print(f"[PERF] torch.set_num_threads({_TORCH_THREADS}) theo PIG_TORCH_THREADS")
    except Exception as exc:  # noqa: BLE001
        print(f"[PERF] Khong the set torch threads: {exc}")

EMPTY_SECONDS = 3.0
STARTUP_EMPTY_CONFIRM_SECONDS = 3.0

MIN_TRACK_HITS = 2
PLUS_SOURCE_FRAMES = 3
PLUS_TARGET_FRAMES = 2
MINUS_SOURCE_FRAMES = 2
MINUS_TARGET_FRAMES = 2
SAME_DIRECTION_COOLDOWN_FRAMES = 8

LINE_BAND_NORM = 0.012
LINE_END_MARGIN = 0.05

PIG_CLASS_NAME = "pig"


# ============================================================
# STATE
# ============================================================

@dataclass
class LineTrackState:
    total_hits: int = 0
    stable_side: int = 0
    stable_side_frames: int = 0
    stable_side_peak_conf: float = 0.0
    last_clear_point: Optional[tuple] = None

    pending_delta: int = 0
    pending_to_side: int = 0
    pending_target_frames: int = 0
    pending_required_target_frames: int = 0
    pending_conf_threshold: float = 0.0
    pending_peak_conf: float = 0.0
    pending_point: Optional[tuple] = None

    last_event_frame: int = -100_000
    last_event_delta: int = 0


@dataclass
class BatchState:
    batch_no: int = 1
    mode: str = "AUTO_START"
    active: bool = False
    start_t: Optional[float] = None
    empty_since: Optional[float] = None
    occupied_since: Optional[float] = None
    net_count: int = 0
    event_track_ids: set = field(default_factory=set)
    green_tracks: dict = field(default_factory=dict)
    pink_tracks: dict = field(default_factory=dict)


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def norm_point(point, w, h):
    return float(point[0]) * w, float(point[1]) * h


def norm_polygon(points, w, h):
    arr = np.asarray([[x * w, y * h] for x, y in points], dtype=np.float32)
    return np.round(arr).astype(np.int32).reshape((-1, 1, 2))


def midpoint(a, b):
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def signed_distance_to_line(point, a, b):
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    return (dx * (py - ay) - dy * (px - ax)) / length


def side_with_band(point, a, b, band_px):
    d = signed_distance_to_line(point, a, b)
    if abs(d) <= band_px:
        return 0
    return 1 if d > 0 else -1


def segment_line_intersection(p1, p2, a, b, end_margin=0.0):
    p = np.asarray(p1, dtype=np.float64)
    r = np.asarray(p2, dtype=np.float64) - p
    q = np.asarray(a, dtype=np.float64)
    s = np.asarray(b, dtype=np.float64) - q

    def cross2(u, v):
        return float(u[0] * v[1] - u[1] * v[0])

    denom = cross2(r, s)
    if abs(denom) < 1e-9:
        return False, None

    qp = q - p
    t = cross2(qp, s) / denom
    u = cross2(qp, r) / denom
    if 0.0 <= t <= 1.0 and -end_margin <= u <= 1.0 + end_margin:
        inter = p + t * r
        return True, (float(inter[0]), float(inter[1]))
    return False, None


def direction_delta(from_side, to_side, pos_src, pos_tgt):
    if from_side == pos_src and to_side == pos_tgt:
        return +1
    if from_side == pos_tgt and to_side == pos_src:
        return -1
    return 0


def reset_pending(state: LineTrackState):
    state.pending_delta = 0
    state.pending_to_side = 0
    state.pending_target_frames = 0
    state.pending_required_target_frames = 0
    state.pending_conf_threshold = 0.0
    state.pending_peak_conf = 0.0
    state.pending_point = None


def maybe_confirm_pending(state: LineTrackState, current_side, frame_index):
    if state.pending_delta == 0 or state.pending_to_side == 0 or current_side != state.pending_to_side:
        return 0, None
    if state.pending_target_frames < state.pending_required_target_frames:
        return 0, None
    if state.pending_peak_conf < state.pending_conf_threshold:
        return 0, None
    if state.pending_delta == state.last_event_delta and (
        frame_index - state.last_event_frame < SAME_DIRECTION_COOLDOWN_FRAMES
    ):
        reset_pending(state)
        return 0, None
    delta = state.pending_delta
    point = state.pending_point
    state.last_event_frame = frame_index
    state.last_event_delta = delta
    reset_pending(state)
    return delta, point


def update_bidirectional_line(*, point, confidence, state: LineTrackState, line_a, line_b,
                               positive_source_sign, positive_target_sign, band_px, frame_index):
    state.total_hits += 1
    current_side = side_with_band(point, line_a, line_b, band_px)

    if current_side == 0:
        return 0, None

    if state.stable_side == 0:
        state.stable_side = current_side
        state.stable_side_frames = 1
        state.stable_side_peak_conf = float(confidence)
        state.last_clear_point = (float(point[0]), float(point[1]))
        return 0, None

    if current_side == state.stable_side:
        state.stable_side_frames += 1
        state.stable_side_peak_conf = max(state.stable_side_peak_conf, float(confidence))
        if state.pending_delta != 0 and state.pending_to_side == current_side:
            state.pending_target_frames += 1
            state.pending_peak_conf = max(state.pending_peak_conf, float(confidence))
            delta, cp = maybe_confirm_pending(state, current_side, frame_index)
            if delta != 0:
                state.last_clear_point = (float(point[0]), float(point[1]))
                return delta, cp
        state.last_clear_point = (float(point[0]), float(point[1]))
        return 0, None

    old_side = state.stable_side
    old_side_frames = state.stable_side_frames
    old_side_peak_conf = state.stable_side_peak_conf

    intersects, crossing_point = False, None
    if state.last_clear_point is not None:
        intersects, crossing_point = segment_line_intersection(
            state.last_clear_point, point, line_a, line_b, end_margin=LINE_END_MARGIN
        )

    candidate_delta = direction_delta(old_side, current_side, positive_source_sign, positive_target_sign)

    state.stable_side = current_side
    state.stable_side_frames = 1
    state.stable_side_peak_conf = float(confidence)
    state.last_clear_point = (float(point[0]), float(point[1]))
    reset_pending(state)

    if not intersects or candidate_delta == 0:
        return 0, None

    if candidate_delta > 0:
        required_source_frames = PLUS_SOURCE_FRAMES
        required_target_frames = PLUS_TARGET_FRAMES
        conf_threshold = COUNT_CONF_PLUS
    else:
        required_source_frames = MINUS_SOURCE_FRAMES
        required_target_frames = MINUS_TARGET_FRAMES
        conf_threshold = COUNT_CONF_MINUS

    if old_side_frames < required_source_frames or state.total_hits < MIN_TRACK_HITS:
        return 0, None

    state.pending_delta = candidate_delta
    state.pending_to_side = current_side
    state.pending_target_frames = 1
    state.pending_required_target_frames = required_target_frames
    state.pending_conf_threshold = conf_threshold
    state.pending_peak_conf = max(old_side_peak_conf, float(confidence))
    state.pending_point = crossing_point

    delta, confirmed_point = maybe_confirm_pending(state, current_side, frame_index)
    if delta != 0:
        return delta, confirmed_point
    return 0, None


def polygon_center(poly):
    if poly is None or len(poly) < 3:
        return None
    polygon = np.asarray(poly, dtype=np.float32)
    m = cv2.moments(polygon)
    if abs(m["m00"]) > 1e-6:
        return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    mean_point = polygon.mean(axis=0)
    return int(mean_point[0]), int(mean_point[1])


def detection_center(result, index):
    if result.masks is not None and result.masks.xy is not None and index < len(result.masks.xy):
        c = polygon_center(result.masks.xy[index])
        if c is not None:
            return c
    x1, y1, x2, y2 = result.boxes.xyxy[index].detach().cpu().numpy()
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def find_pig_class(model):
    names = model.names
    pairs = names.items() if isinstance(names, dict) else enumerate(names)
    for cid, cname in pairs:
        if str(cname).strip().lower() == PIG_CLASS_NAME.lower():
            return int(cid)
    # fallback: dùng class đầu tiên nếu model chỉ có 1 lớp
    if len(names) > 0:
        first_key = next(iter(names.items() if isinstance(names, dict) else enumerate(names)))
        return int(first_key[0])
    raise RuntimeError(f"Khong tim thay class '{PIG_CLASS_NAME}'. Classes={model.names}")


def put_text(image, text, position, scale=0.62, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(image, (x - 5, y - th - 6), (x + tw + 5, y + baseline + 5), (0, 0, 0), -1)
    cv2.putText(image, text, position, font, scale, color, thickness, cv2.LINE_AA)


def open_source(source):
    if isinstance(source, int):
        return cv2.VideoCapture(source), True
    s = str(source)
    if s.strip().isdigit():
        return cv2.VideoCapture(int(s.strip())), True
    is_live = s.lower().startswith(("rtsp://", "rtsps://", "rtmp://", "http://", "https://"))
    return cv2.VideoCapture(s, cv2.CAP_FFMPEG), is_live


# ============================================================
# ENGINE
# ============================================================

class Geometry:
    """Hình học do người dùng vẽ tay, toạ độ chuẩn hoá (0..1)."""

    def __init__(self, chamber_polygon_norm, green_line_norm, pink_line_norm):
        if len(chamber_polygon_norm) < 3:
            raise ValueError("Vùng chuồng (ROI) cần tối thiểu 3 điểm.")
        if len(green_line_norm) != 2 or len(pink_line_norm) != 2:
            raise ValueError("Mỗi cổng (line) cần đúng 2 điểm.")
        self.chamber_polygon_norm = [tuple(p) for p in chamber_polygon_norm]
        self.green_line_norm = [tuple(p) for p in green_line_norm]
        self.pink_line_norm = [tuple(p) for p in pink_line_norm]

    def to_dict(self):
        return {
            "chamber_polygon": self.chamber_polygon_norm,
            "green_line": self.green_line_norm,
            "pink_line": self.pink_line_norm,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["chamber_polygon"], d["green_line"], d["pink_line"])


class PigCounterEngine:
    """
    Chạy vòng lặp đọc frame -> YOLO track -> áp dụng logic 2 cổng (dual gate)
    -> cập nhật state dùng chung (thread-safe) để Flask đọc & stream ra ngoài.
    """

    def __init__(self, job_id, model_path, source, is_live, geometry: Geometry,
                 output_dir: Path, device=None, model=None, pig_class_id=None, model_lock=None):
        self.job_id = job_id
        self.model_path = str(model_path)
        self.source = source
        self.is_live = is_live
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # model + pig_class_id: nếu được truyền sẵn từ ngoài (nạp 1 lần lúc
        # khởi động server) thì DÙNG LẠI, không nạp lại mỗi lần bấm Start.
        # model_lock: khoá dùng chung để tránh 2 job chạy song song cùng lúc
        # gọi model.track() đè state tracker lên nhau (ultralytics tracker
        # không thread-safe khi persist=True).
        self._preloaded_model = model
        self._preloaded_pig_class_id = pig_class_id
        self._model_lock = model_lock or threading.Lock()

        self.device = device
        if self.device is None:
            self.device = 0 if (torch is not None and torch.cuda.is_available()) else "cpu"

        self._lock = threading.RLock()
        self.geometry = geometry

        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        self.latest_jpeg = None
        self.latest_raw_jpeg = None  # khung hình SẠCH (chưa có ROI/vạch/khung nhận diện) để chỉnh lại vùng vẽ
        self.latest_frame_lock = threading.Lock()

        self.finished = []       # list[dict] các lượt đã chốt
        self.event_log = []      # list[dict] log từng sự kiện qua vạch
        self.status = "starting"  # starting|running|paused|finished|error|stopped
        self.error_message = None

        self.batch_state = BatchState()
        self.pigs_in_chamber = 0
        self.total_finished_pigs = 0
        self.frame_index = 0
        self.fps = 25.0
        self.width = 0
        self.height = 0

        self.output_video_path = self.output_dir / f"{job_id}_counted.mp4"
        self.output_batches_csv = self.output_dir / f"{job_id}_batches.csv"
        self.output_events_csv = self.output_dir / f"{job_id}_events.csv"
        # FIX PERF: theo dõi đã ghi bao nhiêu dòng ra CSV, để _save_csvs()
        # chỉ append phần MỚI thay vì ghi lại toàn bộ lịch sử mỗi lần gọi
        # (xem chi tiết ở _save_csvs).
        self._batches_csv_written = 0
        self._events_csv_written = 0

        self._thread = None

    # ---------------- public API ----------------

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_event.set()

    def pause(self):
        self.pause_event.set()
        self.status = "paused"

    def resume(self):
        self.pause_event.clear()
        self.status = "running"

    def update_geometry(self, geometry: Geometry):
        with self._lock:
            self.geometry = geometry

    def get_status(self):
        with self._lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "error": self.error_message,
                "batch_no": self.batch_state.batch_no,
                "mode": self.batch_state.mode,
                "net_count": self.batch_state.net_count,
                "pigs_in_chamber": self.pigs_in_chamber,
                # FIX PERF: trước đây trả về TOÀN BỘ self.finished mỗi lần
                # poll (mỗi giây/lần). Với job chạy nhiều ngày tích luỹ hàng
                # trăm/nghìn lượt, payload JSON gửi đi mỗi giây sẽ CÀNG LÚC
                # CÀNG LỚN, và phía trình duyệt build lại TOÀN BỘ bảng HTML
                # từ đầu mỗi lần nhận (xem renderStatus() trong app.js) ->
                # cùng kiểu lỗi "càng chạy lâu càng chậm" như _save_csvs
                # trước đây, chỉ khác là ở đường status polling. Chỉ cần gửi
                # N lượt GẦN NHẤT để hiển thị "đang chạy tốt" — lịch sử đầy
                # đủ đã có sẵn trong file CSV tải về sau khi job kết thúc.
                "finished_batches": list(self.finished[-50:]),
                "total_batches": len(self.finished),
                "total_pigs": self.total_finished_pigs,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "is_live": self.is_live,
            }

    def get_latest_jpeg(self):
        with self.latest_frame_lock:
            return self.latest_jpeg

    def get_latest_raw_jpeg(self):
        """Khung hình sạch gần nhất (không có ROI/vạch/khung AI vẽ đè lên),
        dùng để hiển thị khi người dùng chỉnh lại vùng vẽ cho khỏi rối mắt."""
        with self.latest_frame_lock:
            return self.latest_raw_jpeg or self.latest_jpeg

    # ---------------- internals ----------------

    def _derived_geometry(self, geometry: Geometry):
        w, h = self.width, self.height
        chamber_polygon = norm_polygon(geometry.chamber_polygon_norm, w, h)
        green_a = norm_point(geometry.green_line_norm[0], w, h)
        green_b = norm_point(geometry.green_line_norm[1], w, h)
        pink_a = norm_point(geometry.pink_line_norm[0], w, h)
        pink_b = norm_point(geometry.pink_line_norm[1], w, h)

        chamber_center = polygon_center(chamber_polygon.reshape(-1, 2)) or (w / 2, h / 2)

        green_inside_distance = signed_distance_to_line(chamber_center, green_a, green_b)
        pink_inside_distance = signed_distance_to_line(chamber_center, pink_a, pink_b)

        green_inside_sign = 1 if green_inside_distance >= 0 else -1
        pink_inside_sign = 1 if pink_inside_distance >= 0 else -1

        # GREEN: ngoài -> trong = +1 ; trong -> ngoài = -1
        green_pos_src = -green_inside_sign
        green_pos_tgt = green_inside_sign
        # PINK: trong -> ngoài = +1 ; ngoài -> trong = -1
        pink_pos_src = pink_inside_sign
        pink_pos_tgt = -pink_inside_sign

        band_px = max(4.0, LINE_BAND_NORM * h)

        return {
            "chamber_polygon": chamber_polygon,
            "green_a": green_a, "green_b": green_b,
            "pink_a": pink_a, "pink_b": pink_b,
            "green_pos_src": green_pos_src, "green_pos_tgt": green_pos_tgt,
            "pink_pos_src": pink_pos_src, "pink_pos_tgt": pink_pos_tgt,
            "band_px": band_px,
        }

    def _finalize_batch(self, now_s, reason):
        state = self.batch_state
        start_s = state.start_t if state.start_t is not None else now_s
        counted_gate = "GREEN_ENTRY" if state.mode == "ENTRY_GREEN" else "PINK_EXIT"
        row = {
            "batch_no": state.batch_no,
            "mode": state.mode,
            "counted_gate": counted_gate,
            "pig_count": max(0, int(state.net_count)),
            "start_s": round(float(start_s), 3),
            "end_s": round(float(now_s), 3),
            "duration_s": round(float(now_s - start_s), 3),
            "event_track_ids": sorted(state.event_track_ids),
            "reason": reason,
        }
        print(f"[JOB {self.job_id}] [LUOT {state.batch_no}] CHOT: {row['pig_count']} heo | "
              f"cong={counted_gate} | ly do={reason} | thoi gian={row['duration_s']:.1f}s")
        return row

    def _reset_for_next_batch(self):
        state = self.batch_state
        state.batch_no += 1
        state.mode = "WAIT_ENTRY"
        state.active = False
        state.start_t = None
        state.empty_since = None
        state.occupied_since = None
        state.net_count = 0
        state.event_track_ids.clear()
        state.green_tracks.clear()
        state.pink_tracks.clear()

    def _save_csvs(self):
        # FIX PERF: trước đây hàm này mở file ở mode "w" và ghi lại TOÀN BỘ
        # self.finished + self.event_log mỗi lần được gọi (mỗi khi 1 lượt
        # heo được chốt xong). event_log/finished KHÔNG BAO GIỜ được dọn
        # trong suốt vòng đời job (chạy camera live liên tục nhiều ngày),
        # nên càng về sau, mỗi lần chốt lượt phải ghi lại càng nhiều dòng
        # cũ + mới -> chi phí tăng dần vô hạn theo thời gian chạy. Hàm này
        # lại chạy ngay trên thread xử lý video chính (_run) nên mỗi lần
        # ghi sẽ CHẶN LUÔN việc đọc/xử lý frame kế tiếp -> đúng là nguyên
        # nhân "càng chạy lâu càng chậm".
        #
        # Sửa: chuyển sang ghi APPEND — chỉ ghi thêm các dòng MỚI kể từ lần
        # lưu trước (ghi header đúng 1 lần khi tạo file), nhờ vậy chi phí
        # mỗi lần gọi chỉ phụ thuộc số dòng MỚI phát sinh, không phụ thuộc
        # tổng lịch sử đã tích lũy — dù job chạy bao lâu, mỗi lần chốt lượt
        # vẫn nhanh như nhau.
        self._append_batches_csv()
        self._append_events_csv()

    def _append_batches_csv(self):
        new_rows = self.finished[self._batches_csv_written:]
        if not new_rows and self._batches_csv_written > 0:
            return
        is_new_file = self._batches_csv_written == 0
        with open(self.output_batches_csv, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if is_new_file:
                w.writerow(["batch_no", "mode", "counted_gate", "pig_count", "start_seconds",
                            "end_seconds", "duration_seconds", "event_track_ids", "reason"])
            for row in new_rows:
                w.writerow([row["batch_no"], row["mode"], row["counted_gate"], row["pig_count"],
                            row["start_s"], row["end_s"], row["duration_s"],
                            ",".join(map(str, row["event_track_ids"])), row["reason"]])
        self._batches_csv_written = len(self.finished)

    def _append_events_csv(self):
        new_rows = self.event_log[self._events_csv_written:]
        if not new_rows and self._events_csv_written > 0:
            return
        is_new_file = self._events_csv_written == 0
        with open(self.output_events_csv, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if is_new_file:
                w.writerow(["batch_no", "time_seconds", "gate", "track_id", "raw_delta",
                            "applied_delta", "count_after", "note"])
            for ev in new_rows:
                w.writerow([ev["batch_no"], round(ev["time_s"], 3), ev["gate"], ev["track_id"],
                            ev["raw_delta"], ev["applied_delta"], ev["count_after"], ev["note"]])
        self._events_csv_written = len(self.event_log)

    def _run(self):
        # FIX BUG-CRASH-CLEANUP: khai báo trước ở ngoài try để cap/writer vẫn
        # truy cập được trong finally dù exception xảy ra trước khi kịp gán
        # (vd lỗi ngay lúc mở nguồn video/RTSP).
        cap = None
        writer = None
        try:
            if self._preloaded_model is not None:
                model = self._preloaded_model
                pig_class_id = self._preloaded_pig_class_id
                print(f"[JOB {self.job_id}] Dung model AI da nap san (khong phai cho nap lai).")
            else:
                # fallback: khong co model nap san thi tu nap rieng (cham hon)
                model, pig_class_id = load_model(self.model_path)

            # FIX BUG-TRACKER-CARRYOVER: model được DÙNG CHUNG cho MỌI job
            # (nạp 1 lần lúc khởi động server). model.track(..., persist=True)
            # cố ý GIỮ NGUYÊN trạng thái tracker (track ID, danh sách track đã
            # "mất dấu" dùng để nhận-diện-lại khi bị che khuất) giữa các LẦN
            # GỌI — đúng khi cần trong 1 phiên video liên tục, nhưng
            # ultralytics KHÔNG tự biết đây là 1 JOB HOÀN TOÀN MỚI (video/
            # camera khác) mỗi khi người dùng F5 hay bấm "Bắt đầu đếm" lại.
            # Hậu quả: tracker mang theo "rác" tích luỹ từ TẤT CẢ các job đã
            # chạy trước đó kể từ lúc mở server (không chỉ trong 1 job, mà
            # CỘNG DỒN QUA TỪNG LẦN F5) — mỗi frame phải so khớp với danh
            # sách track cũ càng lúc càng dài -> càng F5 nhiều lần, xử lý
            # càng chậm dần, dù mỗi job tính riêng vẫn ổn định (không phải
            # lỗi "càng chạy lâu càng chậm" trong 1 job đã sửa trước đó, mà là
            # "càng F5 nhiều lần càng chậm" giữa CÁC job với nhau).
            # Sửa: xoá predictor đã cache trên model NGAY LÚC BẮT ĐẦU job mới,
            # buộc ultralytics tạo predictor + tracker HOÀN TOÀN MỚI ở lần
            # gọi model.track() đầu tiên của job này — sạch, không mang theo
            # gì từ job trước.
            with self._model_lock:
                try:
                    if getattr(model, "predictor", None) is not None:
                        model.predictor = None
                        print(f"[JOB {self.job_id}] Da reset tracker AI ve trang thai sach "
                              f"(khong con mang rac tu cac job truoc).")
                except Exception as exc:  # noqa: BLE001
                    print(f"[JOB {self.job_id}] Khong the reset tracker (khong nghiem trong, "
                          f"chi la co the mang theo mot it track ID cu tu job truoc): {exc}")

            cap, detected_live = open_source(self.source)
            self.is_live = self.is_live or detected_live

            if not cap.isOpened():
                raise RuntimeError(f"Không mở được nguồn video: {self.source}")

            # FIX DIAGNOSTIC: log rõ đang chạy CPU hay GPU — nếu CPU cao mà
            # đây in ra "cpu" (không phải GPU), thì CPU cao là chuyện BÌNH
            # THƯỜNG của việc chạy YOLO trên CPU real-time, KHÔNG liên quan
            # tới lỗi "cang chay lau cang cham" đã sửa ở _save_csvs.
            if self.device in (0, "0") and torch is not None and torch.cuda.is_available():
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                except Exception:
                    gpu_name = "?"
                print(f"[JOB {self.job_id}] Dang chay tren GPU: {gpu_name} (device={self.device})")
            else:
                print(f"[JOB {self.job_id}] Dang chay tren CPU (device={self.device}) — "
                      f"YOLO real-time tren CPU von di ton nhieu CPU, day la binh thuong, "
                      f"khong phai loi 'cang chay lau cang cham'.")

            fps = float(cap.get(cv2.CAP_PROP_FPS))
            self.fps = fps if math.isfinite(fps) and fps > 1 else 25.0
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self.width <= 0 or self.height <= 0:
                raise RuntimeError("Không đọc được kích thước video.")

            print(f"[JOB {self.job_id}] Bat dau xu ly nguon: {self.source} "
                  f"({self.width}x{self.height} @ {self.fps:.1f}fps, live={self.is_live})")

            writer = cv2.VideoWriter(
                str(self.output_video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps, (self.width, self.height)
            )

            state = self.batch_state
            wall_start = time.monotonic()
            crossing_flashes = []
            fail_count = 0
            startup_start_t = None

            # FIX DIAGNOSTIC: đo & log thời gian xử lý trung bình mỗi frame
            # theo từng cửa sổ 300 frame. Đây là cách để BẠN TỰ KIỂM CHỨNG
            # xem lỗi "cang chay lau cang cham" đã hết chưa: nếu con số
            # ms/frame này giữ ỔN ĐỊNH xuyên suốt nhiều giờ chạy -> đã fix.
            # Nếu số này CỨ TĂNG DẦN theo thời gian -> vẫn còn nghẽn ở đâu đó
            # khác, cần báo lại kèm log này để tìm tiếp.
            _perf_window_start = time.monotonic()
            _perf_window_frames = 0
            PERF_LOG_EVERY_N_FRAMES = int(os.environ.get("PIG_PERF_LOG_EVERY_N", "300"))

            self.status = "running"

            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    time.sleep(0.05)
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.is_live:
                        fail_count += 1
                        if fail_count > 5:
                            cap.release()
                            time.sleep(1.0)
                            cap, _ = open_source(self.source)
                            fail_count = 0
                        time.sleep(0.05)
                        continue
                    else:
                        break
                fail_count = 0

                if self.is_live:
                    now_s = time.monotonic() - wall_start
                else:
                    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    now_s = pos_ms / 1000.0 if pos_ms > 0 else self.frame_index / max(self.fps, 1.0)

                self.frame_index += 1

                # FIX CPU-PERF: trước đây encode JPEG khung SẠCH này ở MỌI
                # frame, dù chỉ dùng để hiển thị khi người dùng bấm "Chỉnh
                # lại vùng vẽ" — 1 thao tác thủ công, không cần ảnh mới từng
                # frame. Encode JPEG tốn CPU đáng kể khi làm liên tục 25-30
                # lần/giây cho không mục đích gì (không ai đang xem nó real-
                # time). Giảm xuống cập nhật mỗi 10 frame (vẫn đủ mới trong
                # chưa tới 1 giây ở fps thông thường) để tiết kiệm CPU mà
                # không ảnh hưởng trải nghiệm người dùng.
                if self.frame_index % 10 == 1:
                    ok_raw, raw_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok_raw:
                        with self.latest_frame_lock:
                            self.latest_raw_jpeg = raw_buf.tobytes()

                with self._lock:
                    geometry = self.geometry
                geo = self._derived_geometry(geometry)

                with self._model_lock:
                    result = model.track(
                        frame, persist=True, tracker=TRACKER, classes=[pig_class_id],
                        conf=TRACK_CONF, iou=IOU, imgsz=IMGSZ, device=self.device, verbose=False,
                    )[0]

                annotated = result.plot()
                pigs_in_chamber = 0

                # Chỉ THU THẬP các lượt cắt vạch trong frame này; việc có tính
                # (+1/-1) vào net_count hay không phụ thuộc cổng nào đang "mở"
                # (xem state machine bên dưới) — giống hệt bản gốc v5.1.
                green_events = []
                pink_events = []

                # Cổng nào đang KHÔNG active thì TẮT HẲN việc tính toán cắt vạch
                # của cổng đó (không chỉ bỏ qua kết quả) — để tránh trường hợp
                # heo đi/lùi qua cổng không active vẫn vô tình ảnh hưởng, và vì
                # nhiều khi người ta không lùa heo đi ngược lại qua cổng kia.
                # Trong lúc còn đang XÁC ĐỊNH cổng nào sẽ dùng (AUTO_START /
                # WAIT_ENTRY) thì vẫn phải tính CẢ HAI để biết heo đi cổng nào.
                compute_green = state.mode in ("AUTO_START", "WAIT_ENTRY", "ENTRY_GREEN")
                # Cổng HỒNG chỉ còn được tính trong lúc AUTO_START (đang xác
                # định lượt 1) hoặc khi đã thật sự đang active ở EXIT_PINK.
                # WAIT_ENTRY (chờ heo qua cổng, kể cả lượt 2 trở đi) không
                # còn tính cổng hồng nữa.
                compute_pink = state.mode in ("AUTO_START", "EXIT_PINK")

                boxes = result.boxes
                if boxes is not None and len(boxes) > 0 and boxes.id is not None:
                    confidences = boxes.conf.detach().cpu().numpy()
                    track_ids = boxes.id.int().detach().cpu().numpy()

                    for i in range(len(boxes)):
                        track_id = int(track_ids[i])
                        conf = float(confidences[i])
                        center = detection_center(result, i)

                        if point_in_polygon(center, geo["chamber_polygon"]):
                            pigs_in_chamber += 1

                        if compute_green:
                            green_state = state.green_tracks.setdefault(track_id, LineTrackState())
                            g_delta, g_point = update_bidirectional_line(
                                point=center, confidence=conf, state=green_state,
                                line_a=geo["green_a"], line_b=geo["green_b"],
                                positive_source_sign=geo["green_pos_src"],
                                positive_target_sign=geo["green_pos_tgt"],
                                band_px=geo["band_px"], frame_index=self.frame_index,
                            )
                            if g_delta != 0:
                                green_events.append((track_id, g_delta, g_point))

                        if compute_pink:
                            pink_state = state.pink_tracks.setdefault(track_id, LineTrackState())
                            p_delta, p_point = update_bidirectional_line(
                                point=center, confidence=conf, state=pink_state,
                                line_a=geo["pink_a"], line_b=geo["pink_b"],
                                positive_source_sign=geo["pink_pos_src"],
                                positive_target_sign=geo["pink_pos_tgt"],
                                band_px=geo["band_px"], frame_index=self.frame_index,
                            )
                            if p_delta != 0:
                                pink_events.append((track_id, p_delta, p_point))

                self.pigs_in_chamber = pigs_in_chamber
                chamber_occupied = pigs_in_chamber > 0

                if startup_start_t is None:
                    startup_start_t = now_s

                # ---- batch state machine (đã sửa theo yêu cầu mới) ----
                # QUY TẮC MỚI:
                #   - CHỈ LƯỢT ĐẦU TIÊN (batch_no == 1, mode AUTO_START) mới
                #     được phép dùng cổng HỒNG (PINK EXIT), và chỉ khi phát
                #     hiện có SẴN heo trong chuồng trong vòng 3 giây đầu tiên
                #     kể từ lúc bắt đầu chạy.
                #   - Nếu trong 3 giây đó chuồng vẫn TRỐNG (không có heo sẵn)
                #     thì coi như bỏ hẳn cổng HỒNG -> chuyển sang chỉ dùng
                #     cổng XANH (GREEN ENTRY) cho lượt 1 và MỌI lượt sau.
                #   - Từ lượt 2 trở đi: LUÔN LUÔN chỉ dùng cổng XANH, không
                #     bao giờ tự động rơi về cổng HỒNG nữa (kể cả khi phát
                #     hiện heo có sẵn trong chuồng mà bỏ lỡ lượt cắt vạch
                #     xanh) — người dùng phải lùa heo qua đúng cổng vào.
                if state.mode == "AUTO_START":
                    if chamber_occupied:
                        self._start_batch("EXIT_PINK", now_s)
                    elif any(delta > 0 for _, delta, _ in green_events):
                        self._start_batch("ENTRY_GREEN", now_s)
                    elif now_s - startup_start_t >= STARTUP_EMPTY_CONFIRM_SECONDS:
                        # Khong co heo san trong 3s dau -> bo han cong HONG,
                        # tu day chi con dung cong XANH (cho ca lan nay lan
                        # cac luot sau).
                        state.mode = "WAIT_ENTRY"
                        state.active = False
                    state.empty_since = None

                elif state.mode == "WAIT_ENTRY":
                    # Chi con cho heo cat vach XANH; KHONG con fallback sang
                    # cong HONG nua (cong HONG chi danh rieng cho luc khoi
                    # dong AUTO_START o tren).
                    if any(delta > 0 for _, delta, _ in green_events):
                        self._start_batch("ENTRY_GREEN", now_s)

                # ---- chỉ cổng đang active mới được tính vào net_count ----
                if state.mode == "ENTRY_GREEN":
                    active_events = green_events
                    active_color = (0, 255, 0)
                elif state.mode == "EXIT_PINK":
                    active_events = pink_events
                    active_color = (255, 0, 255)
                else:
                    active_events = []
                    active_color = (0, 255, 0)

                for track_id, delta, point in active_events:
                    self._apply_gate_event(
                        "GREEN ENTRY" if state.mode == "ENTRY_GREEN" else "PINK EXIT",
                        track_id, delta, now_s, point,
                    )
                    if point is not None:
                        crossing_flashes.append((self.frame_index + 15, int(point[0]), int(point[1]),
                                                  track_id, active_color, "+" if delta > 0 else "-"))

                if state.active:
                    if pigs_in_chamber > 0:
                        state.empty_since = None
                    else:
                        if state.empty_since is None:
                            state.empty_since = now_s
                        if now_s - state.empty_since >= EMPTY_SECONDS:
                            row = self._finalize_batch(now_s, "chamber_empty_3s")
                            self.finished.append(row)
                            self.total_finished_pigs = sum(r["pig_count"] for r in self.finished)
                            self._save_csvs()
                            self._reset_for_next_batch()

                self._draw_overlay(annotated, geo, crossing_flashes, now_s)

                writer.write(annotated)
                ok_jpeg, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok_jpeg:
                    with self.latest_frame_lock:
                        self.latest_jpeg = buf.tobytes()

                # FIX DIAGNOSTIC: log ms/frame trung bình mỗi 300 frame để
                # theo dõi xu hướng theo thời gian (xem giải thích ở trên).
                _perf_window_frames += 1
                if _perf_window_frames >= PERF_LOG_EVERY_N_FRAMES:
                    _elapsed = time.monotonic() - _perf_window_start
                    _avg_ms = (_elapsed / _perf_window_frames) * 1000.0
                    _proc_fps = _perf_window_frames / _elapsed if _elapsed > 0 else 0.0
                    print(f"[JOB {self.job_id}] [PERF] frame_index={self.frame_index} "
                          f"avg={_avg_ms:.1f}ms/frame ({_proc_fps:.1f} fps xu ly) "
                          f"batches_da_chot={len(self.finished)} events_da_ghi={len(self.event_log)}")
                    _perf_window_start = time.monotonic()
                    _perf_window_frames = 0

            # kết thúc vòng lặp
            if state.active and not self.is_live:
                now_s = self.frame_index / max(self.fps, 1.0)
                self.finished.append(self._finalize_batch(now_s, "video_end"))
                self.total_finished_pigs = sum(r["pig_count"] for r in self.finished)

            self.status = "stopped" if self.stop_event.is_set() else "finished"

        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"[JOB {self.job_id}] [LOI] {exc}")
            traceback.print_exc()
            self.error_message = str(exc)
            self.status = "error"

        finally:
            # FIX BUG-CRASH-CLEANUP: trước đây _save_csvs()/cap.release()/
            # writer.release() chỉ nằm ở nhánh chạy XONG BÌNH THƯỜNG — nếu có
            # exception xảy ra GIỮA CHỪNG vòng lặp (RTSP rớt kết nối gây lỗi
            # decode, hết bộ nhớ, v.v.), code nhảy thẳng vào except phía trên
            # và 3 dòng dọn dẹp này BỊ BỎ QUA hoàn toàn. Hậu quả: (1) VideoWriter
            # không được release() đúng cách -> file .mp4 output dễ bị HỎNG
            # (thiếu trailer/index MP4, dù file vẫn có dung lượng) và có thể bị
            # khoá (không xoá/đổi tên được) cho tới khi tiến trình Python
            # thoát hẳn; (2) các dòng CSV đã có trong RAM nhưng chưa kịp ghi ra
            # đĩa bị mất luôn khi crash. Chuyển hết vào finally để LUÔN chạy dù
            # kết thúc bình thường, bị dừng, hay bị lỗi giữa chừng.
            try:
                self._save_csvs()
            except Exception as exc2:  # noqa: BLE001
                print(f"[JOB {self.job_id}] Loi khi luu CSV luc don dep: {exc2}")
            if writer is not None:
                try:
                    writer.release()
                except Exception as exc2:  # noqa: BLE001
                    print(f"[JOB {self.job_id}] Loi khi release video writer: {exc2}")
            if cap is not None:
                try:
                    cap.release()
                except Exception as exc2:  # noqa: BLE001
                    print(f"[JOB {self.job_id}] Loi khi release video capture: {exc2}")

    def _apply_gate_event(self, gate_name, track_id, delta, now_s, point):
        state = self.batch_state
        state.net_count = max(0, state.net_count + delta)
        state.event_track_ids.add(track_id)
        note = "forward_valid" if delta > 0 else "reverse_valid_id_switch_tolerant"
        self.event_log.append({
            "batch_no": state.batch_no, "time_s": now_s, "gate": gate_name,
            "track_id": track_id, "raw_delta": delta, "applied_delta": delta,
            "count_after": state.net_count, "note": note,
        })
        sign = "+1" if delta > 0 else "-1"
        print(f"[JOB {self.job_id}] [LUOT {state.batch_no}] {sign} | {gate_name} | "
              f"ID={track_id} | COUNT={state.net_count} | t={now_s:.2f}s")

    def _start_batch(self, mode, now_s):
        state = self.batch_state
        state.mode = mode
        state.active = True
        state.start_t = now_s
        state.empty_since = None
        state.occupied_since = None
        gate_vn = "CONG XANH (VAO)" if mode == "ENTRY_GREEN" else "CONG HONG (RA)"
        print(f"[JOB {self.job_id}] [LUOT {state.batch_no}] Bat dau dung {gate_vn} | t={now_s:.2f}s")

    def _draw_overlay(self, annotated, geo, crossing_flashes, now_s):
        state = self.batch_state
        # Cổng nào KHÔNG active thì vẽ MỜ (xám) để biết ngay là đang tắt,
        # cổng đang active vẽ đúng màu như bình thường.
        green_active = state.mode in ("AUTO_START", "WAIT_ENTRY", "ENTRY_GREEN")
        pink_active = state.mode in ("AUTO_START", "WAIT_ENTRY", "EXIT_PINK")
        green_color = (0, 255, 0) if green_active else (110, 110, 110)
        pink_color = (255, 0, 255) if pink_active else (110, 110, 110)

        cv2.polylines(annotated, [geo["chamber_polygon"]], True, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(annotated, (int(geo["green_a"][0]), int(geo["green_a"][1])),
                 (int(geo["green_b"][0]), int(geo["green_b"][1])), green_color, 5, cv2.LINE_AA)
        cv2.line(annotated, (int(geo["pink_a"][0]), int(geo["pink_a"][1])),
                 (int(geo["pink_b"][0]), int(geo["pink_b"][1])), pink_color, 5, cv2.LINE_AA)

        active = []
        for expire, x, y, tid, color, sign in crossing_flashes:
            if self.frame_index <= expire:
                active.append((expire, x, y, tid, color, sign))
                cv2.circle(annotated, (x, y), 18, color, 4)
                cv2.putText(annotated, f"{sign} ID {tid}", (x + 20, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        crossing_flashes[:] = active

        put_text(annotated, f"LUOT: {state.batch_no}", (20, 35))
        put_text(annotated, f"NET COUNT: {state.net_count}", (20, 70))
        put_text(annotated, f"HEO TRONG CHUONG: {self.pigs_in_chamber}", (20, 105))
        put_text(annotated, f"DA CHOT: {len(self.finished)} LUOT / {self.total_finished_pigs} HEO", (20, 140))


# ============================================================
# JOB REGISTRY
# ============================================================

class JobRegistry:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def add(self, job_id, engine: PigCounterEngine):
        with self._lock:
            self._jobs[job_id] = engine

    def get(self, job_id) -> Optional[PigCounterEngine]:
        with self._lock:
            return self._jobs.get(job_id)

    def remove(self, job_id):
        with self._lock:
            self._jobs.pop(job_id, None)

    def all_ids(self):
        with self._lock:
            return list(self._jobs.keys())

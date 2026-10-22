import os
import sys
import time
import uuid
import threading
from pathlib import Path

# Options FFmpeg cho RTSP (kỹ thuật lấy từ RockApp) — PHẢI set trước khi import cv2
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", (
    "rtsp_transport;tcp"
    "|fflags;nobuffer+discardcorrupt+genpts"
    "|buffer_size;8388608"
    "|max_delay;500000"
    "|stimeout;5000000"
    "|timeout;5000000"
    "|analyzeduration;2000000"
    "|probesize;2000000"
))

import json  # noqa: E402
import logging  # noqa: E402

import cv2  # noqa: E402
from flask import Flask, request, jsonify, render_template, send_from_directory, Response  # noqa: E402
from werkzeug.utils import secure_filename  # noqa: E402

from engine import PigCounterEngine, Geometry, JobRegistry, load_model  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATE_DIR = BASE_DIR / "state"
MODEL_PATH = BASE_DIR / "best.pt"
LAST_SESSION_PATH = STATE_DIR / "last_session.json"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB


class _SuppressNoisyPollingLogs(logging.Filter):
    """Ẩn các dòng log truy cập lặp lại liên tục (polling status/stream) khỏi
    log của werkzeug, để log không bị trôi và các log quan trọng khác vẫn
    nhìn thấy được. Không ảnh hưởng tới việc xử lý request, chỉ ẩn log."""

    NOISY_SUBSTRINGS = ("/status", "/stream")

    def filter(self, record):
        try:
            message = record.getMessage()
        except Exception:
            return True
        return not any(s in message for s in self.NOISY_SUBSTRINGS)


logging.getLogger("werkzeug").addFilter(_SuppressNoisyPollingLogs())

registry = JobRegistry()
# lưu nguồn (video path / rtsp url) + hình học theo job_id để hỗ trợ "vẽ lại không cần upload lại"
job_sources = {}
job_geometry = {}


def _save_last_session(job_id):
    """Lưu lại nguồn video/RTSP + hình học (ROI/2 cổng) vừa vẽ, để lần sau mở
    app lên có thể tự động tải lại đúng video + đúng vạch đã vẽ, không cần
    vẽ lại từ đầu."""
    src = job_sources.get(job_id)
    geo = job_geometry.get(job_id)
    if not src or not geo:
        return
    try:
        LAST_SESSION_PATH.write_text(
            json.dumps({
                "source_type": src["type"],
                "source": src["source"],
                "geometry": geo.to_dict(),
            }, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[STATE] Khong luu duoc phien gan nhat: {exc}")


def _load_last_session():
    if not LAST_SESSION_PATH.exists():
        return None
    try:
        data = json.loads(LAST_SESSION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("source_type") == "video" and not Path(data.get("source", "")).exists():
        # video cu da bi xoa/di chuyen -> khong con gi de tu dong tai lai
        return None
    return data


def _startup_cleanup():
    """FIX DISK: dọn rác từ phiên chạy TRƯỚC mỗi khi server khởi động lại.

    job_sources/job_geometry/registry đều là dict trong RAM -> mất sạch mỗi
    khi tắt/bật lại server. Vì vậy BẤT KỲ file nào còn sót trong uploads/
    hoặc outputs/ lúc khởi động chắc chắn là rác từ phiên trước, do 2
    nguyên nhân đã phát hiện:
      1. Video upload cũ không bị xoá khi người dùng F5 rồi upload video
         khác (savedSessions phía JS bị mất sau F5 nên không kịp gọi
         /cancel cho job cũ).
      2. Output (.mp4/.csv) của các job từng chạy dở/bị thay thế trước khi
         có bản vá tự-xoá-khi-start-job-moi.
    Giữ lại DUY NHẤT 1 video đang được last_session.json tham chiếu (nếu
    có) để tính năng "tự tải lại phiên gần nhất" vẫn hoạt động bình
    thường; xoá mọi thứ khác.
    """
    keep_path = None
    try:
        if LAST_SESSION_PATH.exists():
            data = json.loads(LAST_SESSION_PATH.read_text(encoding="utf-8"))
            if data.get("source_type") == "video" and data.get("source"):
                keep_path = str(Path(data["source"]).resolve())
    except Exception as exc:  # noqa: BLE001
        print(f"[STARTUP-CLEANUP] Khong doc duoc last_session.json: {exc}")

    removed, kept, failed = 0, 0, 0
    for folder in (UPLOAD_DIR, OUTPUT_DIR):
        if not folder.is_dir():
            continue
        for f in folder.iterdir():
            if not f.is_file():
                continue
            if keep_path and str(f.resolve()) == keep_path:
                kept += 1
                continue
            try:
                f.unlink()
                removed += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[STARTUP-CLEANUP] Khong xoa duoc {f}: {exc}")

    if removed or failed:
        print(f"[STARTUP-CLEANUP] Da xoa {removed} file rac tu phien truoc "
              f"(giu lai {kept} file dang dung, {failed} file loi khong xoa duoc).")


# ============================================================
# NẠP MODEL AI 1 LẦN DUY NHẤT lúc khởi động server (không nạp lại mỗi lần
# bấm "Bắt đầu đếm" nữa) -> hết cảnh phải đợi vài giây/chờ mỗi lần chạy job.
# model_lock: dùng để tránh 2 job chạy song song gọi model.track() đè state
# tracker lên nhau (tracker của ultralytics không thread-safe).
# ============================================================
print("=" * 70)
print("[STARTUP] Dang khoi dong Pig Counter server...")
_startup_cleanup()
MODEL, PIG_CLASS_ID = load_model(MODEL_PATH)
MODEL_LOCK = threading.Lock()
print("[STARTUP] San sang. Moi lan bam 'Bat dau dem' se KHONG phai doi nap lai model nua.")
print("=" * 70)

ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def _new_job_id():
    return uuid.uuid4().hex[:12]


def _open_snapshot_capture(source):
    with _suppress_stderr():
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    return cap


import contextlib  # noqa: E402


@contextlib.contextmanager
def _suppress_stderr():
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)


def _encode_jpeg_base64(frame):
    import base64
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ============================================================
# PAGES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder + "/img", "pig-icon.png")


# ============================================================
# API: upload video -> tạo job + trả về snapshot khung hình đầu để vẽ ROI
# ============================================================

@app.route("/api/upload", methods=["POST"])
def upload_video():
    file = request.files.get("video")
    if not file or file.filename == "":
        return jsonify(success=False, error="Chưa chọn video."), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return jsonify(success=False, error=f"Định dạng {ext} không được hỗ trợ."), 400

    job_id = _new_job_id()
    filename = secure_filename(f"{job_id}{ext}")
    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    cap = cv2.VideoCapture(str(save_path))
    if not cap.isOpened():
        return jsonify(success=False, error="Không đọc được video vừa tải lên."), 400

    ok, frame = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not ok:
        return jsonify(success=False, error="Không đọc được khung hình đầu tiên."), 400

    job_sources[job_id] = {"type": "video", "source": str(save_path)}

    return jsonify(
        success=True,
        job_id=job_id,
        source_type="video",
        width=width,
        height=height,
        snapshot=_encode_jpeg_base64(frame),
    )


# ============================================================
# API: kết nối RTSP -> chụp 1 khung hình để vẽ ROI
# ============================================================

@app.route("/api/rtsp/snapshot", methods=["POST"])
def rtsp_snapshot():
    data = request.get_json(force=True, silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url.startswith(("rtsp://", "rtsps://")):
        return jsonify(success=False, error="URL phải bắt đầu bằng rtsp:// hoặc rtsps://"), 400

    cap = _open_snapshot_capture(url)
    if not cap.isOpened():
        cap.release()
        return jsonify(success=False, error="Không kết nối được tới camera RTSP."), 400

    ok, frame = None, None
    for _ in range(30):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not ok or frame is None:
        return jsonify(success=False, error="Kết nối được nhưng không nhận được khung hình."), 400

    job_id = _new_job_id()
    job_sources[job_id] = {"type": "rtsp", "source": url}

    return jsonify(
        success=True,
        job_id=job_id,
        source_type="rtsp",
        width=width,
        height=height,
        snapshot=_encode_jpeg_base64(frame),
    )


# ============================================================
# API: lưu / cập nhật hình học (ROI + 2 line) do người dùng vẽ tay
# ============================================================

@app.route("/api/job/<job_id>/geometry", methods=["POST"])
def set_geometry(job_id):
    if job_id not in job_sources:
        return jsonify(success=False, error="Job không tồn tại."), 404

    data = request.get_json(force=True, silent=True) or {}
    try:
        geometry = Geometry(
            chamber_polygon_norm=data["chamber_polygon"],
            green_line_norm=data["green_line"],
            pink_line_norm=data["pink_line"],
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify(success=False, error=f"Hình học không hợp lệ: {exc}"), 400

    job_geometry[job_id] = geometry
    _save_last_session(job_id)

    engine = registry.get(job_id)
    if engine is not None:
        engine.update_geometry(geometry)

    return jsonify(success=True)


# ============================================================
# API: bắt đầu / dừng / tạm dừng / tiếp tục job đếm
# ============================================================

@app.route("/api/job/<job_id>/start", methods=["POST"])
def start_job(job_id):
    src_info = job_sources.get(job_id)
    geometry = job_geometry.get(job_id)
    if src_info is None:
        return jsonify(success=False, error="Job không tồn tại. Hãy upload video hoặc kết nối RTSP trước."), 404
    if geometry is None:
        return jsonify(success=False, error="Chưa có vùng vẽ (ROI/cổng). Hãy vẽ trước khi bắt đầu."), 400

    existing = registry.get(job_id)
    if existing is not None and existing.status in ("running", "starting", "paused"):
        return jsonify(success=False, error="Job này đang chạy rồi."), 400

    # FIX BUG: app này KHÔNG được thiết kế để chạy nhiều job cùng lúc — chỉ
    # có 1 model dùng chung + 1 khoá (MODEL_LOCK) để tránh 2 job gọi
    # model.track() đè state tracker lên nhau. Trước đây route này chỉ kiểm
    # tra CHÍNH job_id sắp start, không hề biết/dừng các job KHÁC còn đang
    # chạy ngầm trong registry — nên qua nhiều lần bấm "Bắt đầu đếm" ở các
    # tab/phiên trình duyệt khác nhau (hoặc JS không kịp huỷ job cũ), các
    # job cũ cứ tồn tại mãi trong nền, tranh nhau MODEL_LOCK, khiến job nào
    # cũng chạy chậm dần dù trên UI chỉ thấy đúng 1 job. Giờ CHỦ ĐỘNG dừng
    # mọi job khác trước khi start job mới, đảm bảo tại một thời điểm chỉ
    # có tối đa 1 job thực sự đang chạy.
    for other_id in registry.all_ids():
        if other_id == job_id:
            continue
        other = registry.get(other_id)
        if other is not None and other.status in ("running", "starting", "paused"):
            print(f"[JOB {job_id}] Phat hien job cu con chay ngam: {other_id} (status={other.status}) "
                  f"-> tu dong dung no lai truoc khi start job moi.")
            try:
                other.stop()
                # FIX RACE: stop() chỉ báo hiệu dừng (stop_event.set()), luồng
                # xử lý nền cần thêm chút thời gian mới thoát vòng lặp và
                # writer.release() file video. Phải đợi thread đó thoát hẳn
                # trước khi xoá file, nếu không trên Windows sẽ lỗi "file
                # đang được sử dụng" và xoá thất bại (job cũ vẫn kẹt trên đĩa).
                if other._thread is not None:
                    other._thread.join(timeout=5)
            except Exception as exc:  # noqa: BLE001
                print(f"[JOB {job_id}] Loi khi dung job cu {other_id}: {exc}")

            # FIX DISK: job {other_id} bị job mới thay thế coi như đã "mất
            # dấu" — UI hiện tại (và mọi phiên "resume" sau này) chỉ còn
            # nhớ job MỚI, nên file output (.mp4/.csv) của job cũ sẽ không
            # bao giờ được ai xem lại nữa. Trước đây các file này bị bỏ
            # quên vĩnh viễn trên đĩa. Nguyên nhân phổ biến nhất khiến việc
            # này xảy ra liên tục: mỗi lần F5 trang, app.js tự gọi lại
            # /api/last-session + /start, tạo 1 job MỚI xử lý lại từ đầu
            # -> mỗi lần F5 sinh thêm 1 bộ file output -> ổ đĩa tăng dần.
            # Xoá file của job vừa bị thay thế để tại một thời điểm chỉ có
            # tối đa 1 bộ output tồn tại trên đĩa.
            for f in (other.output_video_path, other.output_batches_csv, other.output_events_csv):
                try:
                    if f.exists():
                        f.unlink()
                except Exception as exc:  # noqa: BLE001
                    print(f"[JOB {job_id}] Khong xoa duoc file cu {f}: {exc}")

            registry.remove(other_id)
            job_sources.pop(other_id, None)
            job_geometry.pop(other_id, None)

    is_live = src_info["type"] == "rtsp"
    engine = PigCounterEngine(
        job_id=job_id,
        model_path=MODEL_PATH,
        source=src_info["source"],
        is_live=is_live,
        geometry=geometry,
        output_dir=OUTPUT_DIR,
        model=MODEL,
        pig_class_id=PIG_CLASS_ID,
        model_lock=MODEL_LOCK,
    )
    registry.add(job_id, engine)
    engine.start()

    return jsonify(success=True, job_id=job_id)


@app.route("/api/jobs")
def list_jobs():
    """FIX DIAGNOSTIC: liệt kê TẤT CẢ job hiện có trong registry (kể cả job
    không hiển thị trên UI hiện tại) — dùng để kiểm tra có job nào chạy
    ngầm ngoài ý muốn hay không, đúng tình huống vừa gặp ở trên."""
    result = []
    for jid in registry.all_ids():
        engine = registry.get(jid)
        if engine is None:
            continue
        result.append({
            "job_id": jid,
            "status": engine.status,
            "frame_index": engine.frame_index,
            "source": job_sources.get(jid, {}).get("source"),
        })
    return jsonify(success=True, jobs=result)


@app.route("/api/job/<job_id>/pause", methods=["POST"])
def pause_job(job_id):
    engine = registry.get(job_id)
    if engine is None:
        return jsonify(success=False, error="Job chưa chạy."), 404
    engine.pause()

    # chụp lại khung hình mới nhất (SẠCH, không có ROI/vạch/khung AI) để vẽ lại ROI cho chính xác
    jpeg = engine.get_latest_raw_jpeg()
    snapshot = None
    if jpeg:
        import base64
        snapshot = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")

    return jsonify(success=True, snapshot=snapshot, width=engine.width, height=engine.height)


@app.route("/api/job/<job_id>/resume", methods=["POST"])
def resume_job(job_id):
    engine = registry.get(job_id)
    if engine is None:
        return jsonify(success=False, error="Job chưa chạy."), 404
    engine.resume()
    return jsonify(success=True)


@app.route("/api/job/<job_id>/stop", methods=["POST"])
def stop_job(job_id):
    engine = registry.get(job_id)
    if engine is None:
        return jsonify(success=False, error="Job chưa chạy."), 404
    engine.stop()
    return jsonify(success=True)


@app.route("/api/job/<job_id>/status")
def job_status(job_id):
    engine = registry.get(job_id)
    if engine is None:
        return jsonify(success=False, error="Job chưa chạy."), 404
    return jsonify(success=True, **engine.get_status())


@app.route("/api/job/<job_id>/stream")
def job_stream(job_id):
    engine = registry.get(job_id)
    if engine is None:
        return jsonify(success=False, error="Job chưa chạy."), 404

    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            jpeg = engine.get_latest_jpeg()
            if jpeg is not None:
                yield boundary + jpeg + b"\r\n"
            time.sleep(1 / 20)
            if engine.status in ("finished", "stopped", "error") and engine.get_latest_jpeg() is jpeg:
                # gửi thêm vài khung cuối rồi dừng generator để trình duyệt đóng kết nối gọn gàng
                break

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame",
                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/job/<job_id>/download/<kind>")
def download_file(job_id, kind):
    engine = registry.get(job_id)
    mapping = {}
    if engine is not None:
        mapping = {
            "video": engine.output_video_path,
            "batches": engine.output_batches_csv,
            "events": engine.output_events_csv,
        }
    path = mapping.get(kind)
    if path is None or not Path(path).exists():
        return jsonify(success=False, error="Không tìm thấy file."), 404
    return send_from_directory(path.parent, path.name, as_attachment=True)


@app.route("/api/job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    """Dừng job (nếu đang chạy) và xoá video đã upload cho job này (nếu có),
    dùng khi người dùng đổi sang nguồn khác mà không muốn giữ video cũ."""
    engine = registry.get(job_id)
    if engine is not None:
        try:
            engine.stop()
        except Exception:  # noqa: BLE001
            pass
        registry.remove(job_id)

    src_info = job_sources.pop(job_id, None)
    job_geometry.pop(job_id, None)
    if src_info and src_info.get("type") == "video":
        try:
            Path(src_info["source"]).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    return jsonify(success=True)


@app.route("/api/last-session")
def last_session():
    """Trả về video/RTSP + hình học (ROI/2 cổng) đã vẽ lần chạy gần nhất
    (nếu còn), để frontend tự động tải lại mà không cần vẽ lại từ đầu."""
    data = _load_last_session()
    if not data:
        return jsonify(success=True, found=False)

    src_type = data["source_type"]
    source = data["source"]

    if src_type == "video":
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return jsonify(success=True, found=False)
        ok, frame = cap.read()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if not ok:
            return jsonify(success=True, found=False)
    else:  # rtsp
        cap = _open_snapshot_capture(source)
        if not cap.isOpened():
            cap.release()
            return jsonify(success=True, found=False)
        ok, frame = None, None
        for _ in range(20):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if not ok or frame is None:
            return jsonify(success=True, found=False)

    try:
        geometry = Geometry.from_dict(data["geometry"])
    except Exception:
        return jsonify(success=True, found=False)

    job_id = _new_job_id()
    job_sources[job_id] = {"type": src_type, "source": source}
    job_geometry[job_id] = geometry

    print(f"[STARTUP] Tu dong tai lai phien gan nhat: {source} (job_id moi={job_id})")

    return jsonify(
        success=True,
        found=True,
        job_id=job_id,
        source_type=src_type,
        width=width,
        height=height,
        snapshot=_encode_jpeg_base64(frame),
        geometry=geometry.to_dict(),
    )


@app.route("/api/health")
def health():
    return jsonify(success=True, status="ok")


# ============================================================
# API: dừng hẳn ứng dụng (dùng cho nút "Dừng ứng dụng" trên giao diện,
# để người dùng không rành máy tính không cần đóng cửa sổ đen / Task Manager)
# ============================================================

@app.route("/api/shutdown", methods=["POST"])
def shutdown():
    # FIX BUG-CRASH-CLEANUP: trước đây chỉ báo hiệu dừng (engine.stop()) rồi
    # NGỦ CỐ ĐỊNH 0.6 GIÂY bất kể job có xử lý xong frame hiện tại hay chưa,
    # sau đó os._exit(0) giết tiến trình NGAY LẬP TỨC — os._exit không chạy
    # bất kỳ dọn dẹp Python nào (kể cả khối finally trong PigCounterEngine._run
    # vừa thêm để release() video writer / flush CSV). Nếu 1 frame đang xử lý
    # dở mất hơn 0.6s (rất dễ xảy ra khi chạy CPU), bấm "Dừng ứng dụng" vẫn có
    # thể làm HỎNG file video của job đang chạy, y hệt như tắt máy đột ngột.
    # Sửa: đợi THẬT SỰ tới khi các luồng xử lý thoát hẳn (join), có giới hạn
    # thời gian an toàn để tránh treo vô hạn nếu 1 job nào đó kẹt bất thường.
    threads_to_wait = []
    for job_id in list(registry.all_ids()):
        engine = registry.get(job_id)
        if engine is not None:
            try:
                engine.stop()
                if engine._thread is not None:
                    threads_to_wait.append(engine._thread)
            except Exception:  # noqa: BLE001
                pass

    def _kill():
        for t in threads_to_wait:
            t.join(timeout=10)
        os._exit(0)

    threading.Thread(target=_kill, daemon=True).start()
    return jsonify(success=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)

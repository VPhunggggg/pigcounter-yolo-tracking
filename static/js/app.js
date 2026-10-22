(function () {
  "use strict";

  // ---------------------------------------------------------
  // STATE
  // ---------------------------------------------------------
  let jobId = null;
  let sourceType = null; // 'video' | 'rtsp'
  let srcW = 0, srcH = 0;
  let snapshotImage = null; // Image object for initial draw canvas
  let pollTimer = null;

  const draw = {
    mode: "chamber", // chamber | green | pink
    chamber: [],     // array of {x,y} in canvas pixel coords
    green: [],
    pink: [],
  };
  const redraw = {
    mode: "chamber",
    chamber: [],
    green: [],
    pink: [],
  };
  // hình học (toạ độ chuẩn hoá 0..1) đã lưu gần nhất cho job hiện tại,
  // dùng để tự điền lại khi mở bảng "Chỉnh lại vùng vẽ"
  let lastGeometryNorm = null;

  // Lưu lại phiên GẦN NHẤT cho từng LOẠI nguồn (video / rtsp), để khi người
  // dùng chuyển qua lại giữa 2 tab "Tải video lên" <-> "Camera RTSP" thì
  // phiên cũ (video/URL + vùng đã vẽ) vẫn còn nguyên, không bị mất / không
  // phải tải lại hay vẽ lại từ đầu.
  const savedSessions = { video: null, rtsp: null };

  // ---------------------------------------------------------
  // HELPERS
  // ---------------------------------------------------------
  const $ = (sel) => document.querySelector(sel);
  const $all = (sel) => Array.from(document.querySelectorAll(sel));

  function showError(msg) {
    const box = $("#sourceError");
    box.textContent = msg;
    box.classList.remove("hidden");
  }
  function clearError() {
    $("#sourceError").classList.add("hidden");
  }

  function setTopbarStatus(text, live) {
    const el = $("#topbarStatus");
    el.innerHTML = `<span class="dot ${live ? "dot-live" : "dot-idle"}"></span> ${text}`;
  }

  // ---------------------------------------------------------
  // STEP NAV (1 Nguồn / 2 Vẽ vùng / 3 Đang đếm)
  // ---------------------------------------------------------
  function setStep(step) {
    // step: "source" | "draw" | "run"
    const order = ["source", "draw", "run"];
    const idx = order.indexOf(step);
    const map = {
      source: { pill: $("#pillSource"), section: $("#sourceSection") },
      draw: { pill: $("#pillDraw"), section: $("#drawSection") },
      run: { pill: $("#pillRun"), section: $("#runSection") },
    };
    order.forEach((key, i) => {
      map[key].section.classList.toggle("hidden", key !== step);
      if (key === step) map[key].section.classList.add("panel-anim");
      map[key].pill.classList.toggle("active", key === step);
      map[key].pill.classList.toggle("done", i < idx);
      map[key].pill.disabled = i > idx;
    });
    $("#lineA").style.background = idx >= 1 ? "var(--accent-2)" : "var(--border)";
    $("#lineB").style.background = idx >= 2 ? "var(--accent-2)" : "var(--border)";
  }

  // cho phép bấm vào bước "1. Chọn nguồn" (khi đã hoàn thành) để quay lại đổi nguồn
  $("#pillSource").addEventListener("click", () => {
    if ($("#pillSource").disabled) return;
    resetToSource();
  });

  async function resetToSource() {
    // CHỈ dừng job đang chạy (không xoá video / không huỷ job hẳn), để nếu
    // người dùng quay lại nguồn này sau (qua tab hoặc tự động khôi phục) thì
    // video + vùng đã vẽ vẫn còn nguyên, không phải tải/vẽ lại từ đầu.
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    $("#streamImg").src = ""; // ngắt kết nối MJPEG stream ngay lập tức trên trình duyệt
    if (jobId) {
      try { await fetch(`/api/job/${jobId}/stop`, { method: "POST" }); } catch (e) { /* ignore */ }
    }
    jobId = null; sourceType = null; lastGeometryNorm = null;
    $("#downloadRow").classList.add("hidden");
    $("#uploadProgress").classList.add("hidden");
    $("#uploadFill").style.width = "0%";
    $("#uploadPct").textContent = "0%";
    draw.chamber = []; draw.green = []; draw.pink = [];
    clearError();
    setTopbarStatus("Chưa có phiên đếm", false);
    setStep("source");
  }

  // ---------------------------------------------------------
  // DỪNG ỨNG DỤNG (tắt hẳn server, dành cho người dùng không rành máy tính)
  // ---------------------------------------------------------
  $("#shutdownBtn").addEventListener("click", () => {
    $("#shutdownConfirmOverlay").classList.remove("hidden");
  });
  $("#cancelShutdownBtn").addEventListener("click", () => {
    $("#shutdownConfirmOverlay").classList.add("hidden");
  });
  $("#confirmShutdownBtn").addEventListener("click", async () => {
    $("#confirmShutdownBtn").disabled = true;
    $("#confirmShutdownBtn").textContent = "Đang dừng...";
    try { await fetch("/api/shutdown", { method: "POST" }); } catch (e) { /* server đóng ngay, có thể lỗi mạng là bình thường */ }
    $("#shutdownConfirmOverlay").classList.add("hidden");
    $("#shutdownDoneOverlay").classList.remove("hidden");
  });

  // ---------------------------------------------------------
  // SIDEBAR TOGGLE (thu gọn chỉ còn icon / mở rộng đầy đủ chữ)
  // ---------------------------------------------------------
  const SIDEBAR_COLLAPSE_KEY = "pigcounter_sidebar_collapsed";
  const sidebarEl = $("#sidebar");
  function applySidebarState(collapsed) {
    sidebarEl.classList.toggle("collapsed", collapsed);
    $("#sidebarToggle").textContent = collapsed ? "▶" : "◀";
  }
  applySidebarState(localStorage.getItem(SIDEBAR_COLLAPSE_KEY) === "1");
  $("#sidebarToggle").addEventListener("click", () => {
    const collapsed = !sidebarEl.classList.contains("collapsed");
    applySidebarState(collapsed);
    localStorage.setItem(SIDEBAR_COLLAPSE_KEY, collapsed ? "1" : "0");
  });

  // ---------------------------------------------------------
  // CHỌN NGUỒN (Tải video lên / Camera RTSP) — nút bên sidebar
  // Bấm là CHUYỂN THẲNG qua nguồn đó ngay, tự dừng & xoá phiên/video cũ đang xem dở.
  // ---------------------------------------------------------
  $all(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const tab = btn.dataset.tab; // 'upload' | 'rtsp'
      const targetType = tab === "upload" ? "video" : "rtsp";
      const alreadyActiveOnSourceStep =
        btn.classList.contains("active") && !$("#sourceSection").classList.contains("hidden");

      if (!alreadyActiveOnSourceStep) {
        // Dừng job hiện tại (KHÔNG xoá video/huỷ job) trước khi chuyển tab
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        $("#streamImg").src = "";
        if (jobId) {
          try { await fetch(`/api/job/${jobId}/stop`, { method: "POST" }); } catch (e) { /* ignore */ }
        }
        clearError();

        const saved = savedSessions[targetType];
        if (saved) {
          // Đã từng có video/URL cho tab này rồi -> khôi phục lại y như cũ,
          // không cần tải/kết nối lại hay vẽ lại vùng.
          onSourceReady(saved, saved.geometry);
        } else {
          // Chưa có phiên nào cho loại nguồn này -> về màn hình chọn nguồn trống
          jobId = null; sourceType = null; lastGeometryNorm = null;
          $("#downloadRow").classList.add("hidden");
          draw.chamber = []; draw.green = []; draw.pink = [];
          setTopbarStatus("Chưa có phiên đếm", false);
          setStep("source");
        }
      }
      $all(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      $("#tabUpload").classList.toggle("hidden", tab !== "upload");
      $("#tabRtsp").classList.toggle("hidden", tab !== "rtsp");
    });
  });

  // ---------------------------------------------------------
  // UPLOAD VIDEO
  // ---------------------------------------------------------
  const dropzone = $("#dropzone");
  const videoInput = $("#videoInput");

  dropzone.addEventListener("click", () => videoInput.click());
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) uploadVideo(e.dataTransfer.files[0]);
  });
  videoInput.addEventListener("change", () => {
    if (videoInput.files.length) uploadVideo(videoInput.files[0]);
  });

  function uploadVideo(file) {
    clearError();
    const form = new FormData();
    form.append("video", file);

    const progressRow = $("#uploadProgress");
    const fill = $("#uploadFill");
    const pct = $("#uploadPct");
    progressRow.classList.remove("hidden");

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        const p = Math.round((e.loaded / e.total) * 100);
        fill.style.width = p + "%";
        pct.textContent = p + "%";
      }
    });
    xhr.onload = () => {
      progressRow.classList.add("hidden");
      let data;
      try { data = JSON.parse(xhr.responseText); } catch (e) { data = null; }
      if (!data || !data.success) {
        showError((data && data.error) || "Tải video thất bại.");
        return;
      }
      // nếu đang thay THẬT SỰ bằng 1 video mới -> xoá job/video CŨ trên server
      // (chỉ xảy ra khi người dùng chủ động chọn file khác, không phải khi
      // chỉ chuyển qua lại giữa các tab)
      const oldVideoJobId = savedSessions.video ? savedSessions.video.job_id : null;
      onSourceReady(data);
      if (oldVideoJobId && oldVideoJobId !== data.job_id) {
        fetch(`/api/job/${oldVideoJobId}/cancel`, { method: "POST" }).catch(() => {});
      }
    };
    xhr.onerror = () => {
      progressRow.classList.add("hidden");
      showError("Lỗi mạng khi tải video lên.");
    };
    xhr.send(form);
  }

  // ---------------------------------------------------------
  // RTSP CONNECT
  // ---------------------------------------------------------
  $("#rtspConnectBtn").addEventListener("click", async () => {
    clearError();
    const url = $("#rtspUrl").value.trim();
    if (!url) { showError("Nhập địa chỉ RTSP."); return; }
    $("#rtspConnectBtn").disabled = true;
    $("#rtspConnectBtn").textContent = "Đang kết nối...";
    try {
      const res = await fetch("/api/rtsp/snapshot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await res.json();
      if (!data.success) { showError(data.error || "Không kết nối được camera."); return; }
      const oldRtspJobId = savedSessions.rtsp ? savedSessions.rtsp.job_id : null;
      onSourceReady(data);
      if (oldRtspJobId && oldRtspJobId !== data.job_id) {
        fetch(`/api/job/${oldRtspJobId}/stop`, { method: "POST" }).catch(() => {});
      }
    } catch (e) {
      showError("Lỗi mạng khi kết nối RTSP.");
    } finally {
      $("#rtspConnectBtn").disabled = false;
      $("#rtspConnectBtn").textContent = "Kết nối";
    }
  });

  function onSourceReady(data, prefillGeometry) {
    jobId = data.job_id;
    sourceType = data.source_type;
    srcW = data.width;
    srcH = data.height;

    setStep("draw");
    setTopbarStatus("Đang vẽ vùng đếm cho phiên " + jobId, false);

    loadSnapshotIntoCanvas(
      data.snapshot, $("#drawCanvas"), draw,
      () => renderDraw($("#drawCanvas"), draw),
      prefillGeometry || null,
    );
    if (prefillGeometry) lastGeometryNorm = prefillGeometry;

    // nhớ lại phiên này theo loại nguồn (video/rtsp) để phục hồi khi chuyển tab qua lại
    savedSessions[sourceType] = {
      job_id: jobId, source_type: sourceType, width: srcW, height: srcH,
      snapshot: data.snapshot, geometry: prefillGeometry || null,
    };
  }

  // ---------------------------------------------------------
  // TỰ ĐỘNG TẢI LẠI VIDEO + VÙNG VẼ (ROI/2 cổng) CỦA PHIÊN GẦN NHẤT
  // khi mở lại ứng dụng, và CHẠY LUÔN — không cần xem/chỉnh lại ROI hay vạch
  // nữa. Muốn dùng video khác thì bấm "Tải video lên" / "Camera RTSP" ở
  // sidebar để chuyển nguồn như bình thường.
  // ---------------------------------------------------------
  async function tryRestoreLastSession() {
    try {
      const res = await fetch("/api/last-session");
      const data = await res.json();
      if (!data.success || !data.found) return;

      jobId = data.job_id;
      sourceType = data.source_type;
      srcW = data.width;
      srcH = data.height;
      lastGeometryNorm = data.geometry;
      savedSessions[sourceType] = {
        job_id: jobId, source_type: sourceType, width: srcW, height: srcH,
        snapshot: data.snapshot, geometry: data.geometry,
      };

      // hình học đã được server lưu sẵn cho job_id này rồi (từ lần trước),
      // nên chỉ cần gọi start, không cần vẽ / gửi lại geometry.
      const startRes = await fetch(`/api/job/${jobId}/start`, { method: "POST" });
      const startData = await startRes.json();
      if (!startData.success) {
        showError(startData.error || "Không tự chạy lại được video trước đó.");
        return;
      }

      setStep("run");
      $("#stopBtn").classList.remove("hidden");
      $("#changeSourceBtn").classList.add("hidden");
      $("#liveBadge").classList.toggle("hidden", sourceType !== "rtsp");
      $("#streamImg").src = `/api/job/${jobId}/stream?t=${Date.now()}`;
      setTopbarStatus("Tự động chạy lại video/luồng trước đó — phiên " + jobId, true);
      startPolling();
    } catch (e) {
      // không có phiên trước hoặc lỗi mạng -> bỏ qua, người dùng tự chọn nguồn mới
    }
  }
  document.addEventListener("DOMContentLoaded", tryRestoreLastSession);

  // ---------------------------------------------------------
  // CANVAS DRAWING (hand-drawn ROI polygon + 2 gate lines)
  // ---------------------------------------------------------
  function loadSnapshotIntoCanvas(dataUrl, canvas, state, onReady, prefillNorm) {
    const img = new Image();
    img.onload = () => {
      snapshotImage = img;
      const maxW = 1100;
      const scale = Math.min(1, maxW / img.width);
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      canvas._srcImg = img;
      if (prefillNorm) {
        state.chamber = normPointsToCanvas(prefillNorm.chamber_polygon, canvas);
        if (state.chamber.length >= 3) state.chamber.closed = true;
        state.green = normPointsToCanvas(prefillNorm.green_line, canvas);
        state.pink = normPointsToCanvas(prefillNorm.pink_line, canvas);
      } else {
        state.chamber = []; state.green = []; state.pink = [];
      }
      state.mode = "chamber";
      onReady();
    };
    img.src = dataUrl;
  }

  function renderDraw(canvas, state) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(canvas._srcImg, 0, 0, canvas.width, canvas.height);

    drawPolyline(ctx, state.chamber, "#ff5c5c", true);
    drawPolyline(ctx, state.green, "#3ddc73", false);
    drawPolyline(ctx, state.pink, "#ff5fb0", false);

    updateStatusLabels(state);
  }

  function drawPolyline(ctx, points, color, closable) {
    if (points.length === 0) return;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
    if (closable && points.length > 2 && points.closed) ctx.closePath();
    ctx.stroke();
    points.forEach((p, i) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  function updateStatusLabels(state) {
    const chamberOk = state.chamber.closed && state.chamber.length >= 3;
    const greenOk = state.green.length === 2;
    const pinkOk = state.pink.length === 2;
    const c1 = $("#statChamber"), c2 = $("#statGreen"), c3 = $("#statPink");
    if (c1) {
      c1.textContent = "ROI: " + (chamberOk ? "đã vẽ ✓" : `${state.chamber.length} điểm`);
      c1.classList.toggle("ok", chamberOk);
      c2.textContent = "Cổng vào: " + (greenOk ? "đã vẽ ✓" : `${state.green.length}/2 điểm`);
      c2.classList.toggle("ok", greenOk);
      c3.textContent = "Cổng ra: " + (pinkOk ? "đã vẽ ✓" : `${state.pink.length}/2 điểm`);
      c3.classList.toggle("ok", pinkOk);
      $("#startCountBtn").disabled = !(chamberOk && greenOk && pinkOk);
    }
  }

  function attachCanvasEvents(canvas, state, renderFn, instructionsEl) {
    canvas.addEventListener("click", (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (canvas.width / rect.width);
      const y = (e.clientY - rect.top) * (canvas.height / rect.height);

      if (state.mode === "chamber") {
        if (state.chamber.closed) return;
        if (state.chamber.length >= 3) {
          const first = state.chamber[0];
          const dist = Math.hypot(first.x - x, first.y - y);
          if (dist < 12) { state.chamber.closed = true; renderFn(); return; }
        }
        state.chamber.push({ x, y });
      } else if (state.mode === "green") {
        if (state.green.length < 2) state.green.push({ x, y });
        else state.green = [{ x, y }];
      } else if (state.mode === "pink") {
        if (state.pink.length < 2) state.pink.push({ x, y });
        else state.pink = [{ x, y }];
      }
      renderFn();
      if (instructionsEl) updateInstructions(instructionsEl, state);
    });
  }

  function updateInstructions(el, state) {
    if (state.mode === "chamber") {
      el.innerHTML = "Nhấp chuột để thêm điểm cho <b>vùng chuồng (ROI)</b>. Cần tối thiểu 3 điểm, nhấp lại vào điểm đầu tiên (hoặc bấm \"Đóng vùng ROI\") để khép kín.";
    } else if (state.mode === "green") {
      el.innerHTML = "Nhấp 2 điểm để vẽ <b>đường cổng vào (xanh lá)</b> — heo đi từ ngoài vào chuồng qua đây sẽ được +1.";
    } else {
      el.innerHTML = "Nhấp 2 điểm để vẽ <b>đường cổng ra (hồng)</b> — heo đi từ trong chuồng ra ngoài qua đây sẽ được +1.";
    }
  }

  $all(".tool-btn[data-mode]").forEach((btn) => {
    btn.addEventListener("click", () => {
      $all(".tool-btn[data-mode]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      draw.mode = btn.dataset.mode;
      updateInstructions($("#drawInstructions"), draw);
    });
  });

  $("#undoPointBtn").addEventListener("click", () => {
    if (draw.mode === "chamber") { draw.chamber.closed = false; draw.chamber.pop(); }
    else draw[draw.mode].pop();
    renderDraw($("#drawCanvas"), draw);
  });
  $("#clearAllBtn").addEventListener("click", () => {
    draw.chamber = []; draw.green = []; draw.pink = [];
    renderDraw($("#drawCanvas"), draw);
  });
  $("#closePolygonBtn").addEventListener("click", () => {
    if (draw.chamber.length >= 3) { draw.chamber.closed = true; renderDraw($("#drawCanvas"), draw); }
  });

  attachCanvasEvents($("#drawCanvas"), draw, () => renderDraw($("#drawCanvas"), draw), $("#drawInstructions"));

  function toNormPoints(points, canvas) {
    return points.map((p) => [p.x / canvas.width, p.y / canvas.height]);
  }

  function normPointsToCanvas(normPoints, canvas) {
    return (normPoints || []).map(([nx, ny]) => ({ x: nx * canvas.width, y: ny * canvas.height }));
  }

  $("#startCountBtn").addEventListener("click", async () => {
    const canvas = $("#drawCanvas");
    const payload = {
      chamber_polygon: toNormPoints(draw.chamber, canvas),
      green_line: toNormPoints(draw.green, canvas),
      pink_line: toNormPoints(draw.pink, canvas),
    };
    const geoRes = await fetch(`/api/job/${jobId}/geometry`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const geoData = await geoRes.json();
    if (!geoData.success) { showError(geoData.error || "Lưu vùng vẽ thất bại."); return; }
    lastGeometryNorm = payload;
    if (savedSessions[sourceType]) savedSessions[sourceType].geometry = payload;

    const startRes = await fetch(`/api/job/${jobId}/start`, { method: "POST" });
    const startData = await startRes.json();
    if (!startData.success) { showError(startData.error || "Không bắt đầu được."); return; }

    setStep("run");
    $("#stopBtn").classList.remove("hidden");
    $("#changeSourceBtn").classList.add("hidden");
    $("#liveBadge").classList.toggle("hidden", sourceType !== "rtsp");
    $("#streamImg").src = `/api/job/${jobId}/stream?t=${Date.now()}`;
    setTopbarStatus("Đang đếm — phiên " + jobId, true);
    startPolling();
  });

  // ---------------------------------------------------------
  // POLLING STATUS + BATCH TABLE
  // ---------------------------------------------------------
  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollStatus, 1000);
    pollStatus();
  }

  async function pollStatus() {
    if (!jobId) return;
    try {
      const res = await fetch(`/api/job/${jobId}/status`);
      const data = await res.json();
      if (!data.success) return;
      renderStatus(data);
      if (["finished", "stopped", "error"].includes(data.status)) {
        clearInterval(pollTimer);
        onJobEnded(data);
      }
    } catch (e) { /* silent */ }
  }

  function renderStatus(data) {
    $("#statBatchNo").textContent = data.batch_no;
    $("#statInChamber").textContent = data.pigs_in_chamber;
    $("#statTotalPigs").textContent = data.total_pigs;

    const modeLabels = {
      AUTO_START: "Đang chờ heo xuất hiện trong chuồng…",
      WAIT_ENTRY: "Chờ heo qua cổng vào",
      ENTRY_GREEN: "Đang đếm theo cổng VÀO (xanh)",
      EXIT_PINK: "Đang đếm theo cổng RA (hồng)",
    };
    $("#modeLine").textContent = "Chế độ: " + (modeLabels[data.mode] || data.mode);

    const tbody = $("#batchTableBody");
    tbody.innerHTML = "";
    data.finished_batches.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${row.batch_no}</td><td>${row.counted_gate === "GREEN_ENTRY" ? "Vào" : "Ra"}</td>
        <td>${row.pig_count}</td><td>${row.duration_s.toFixed(1)}s</td>`;
      tbody.appendChild(tr);
    });
  }

  function onJobEnded(data) {
    setTopbarStatus("Đã kết thúc phiên " + jobId, false);
    $("#liveBadge").classList.add("hidden");
    $("#downloadRow").classList.remove("hidden");
    $("#dlVideo").href = `/api/job/${jobId}/download/video`;
    $("#dlBatches").href = `/api/job/${jobId}/download/batches`;
    $("#dlEvents").href = `/api/job/${jobId}/download/events`;
    // Phiên đã kết thúc (dừng/xong/lỗi) -> ẩn nút Dừng, hiện rõ nút để
    // chọn video/nguồn khác, tránh người dùng bị "kẹt" không biết bấm đâu.
    $("#stopBtn").classList.add("hidden");
    $("#changeSourceBtn").classList.remove("hidden");
    if (data.status === "error") showError(data.error || "Đã xảy ra lỗi trong quá trình đếm.");
  }

  $("#stopBtn").addEventListener("click", async () => {
    if (!jobId) return;
    await fetch(`/api/job/${jobId}/stop`, { method: "POST" });
  });

  // Nút "Đổi video / nguồn khác" — quay lại bước 1 để tải video mới hoặc
  // kết nối RTSP khác, không cần thao tác gì thêm.
  $("#changeSourceBtn").addEventListener("click", async () => {
    await resetToSource();
  });

  // ---------------------------------------------------------
  // SETTINGS (redraw ROI) OVERLAY — không cần upload lại video
  // ---------------------------------------------------------
  $("#settingsBtn").addEventListener("click", async () => {
    if (!jobId) return;
    const res = await fetch(`/api/job/${jobId}/pause`, { method: "POST" });
    const data = await res.json();
    if (!data.success) { showError(data.error || "Không tạm dừng được job."); return; }

    redraw.mode = "chamber";
    $all(".tool-btn[data-mode2]").forEach((b) => b.classList.remove("active"));
    $(`.tool-btn[data-mode2="chamber"]`).classList.add("active");

    const canvas = $("#redrawCanvas");
    if (data.snapshot) {
      // ảnh trả về là khung hình SẠCH, và tự điền lại vùng đã vẽ trước đó (lastGeometryNorm)
      // để người dùng chỉ cần sửa đúng phần sai, không phải vẽ lại từ đầu.
      loadSnapshotIntoCanvas(data.snapshot, canvas, redraw, () => renderDraw(canvas, redraw), lastGeometryNorm);
    }
    $("#redrawOverlay").classList.remove("hidden");
  });

  $all(".tool-btn[data-mode2]").forEach((btn) => {
    btn.addEventListener("click", () => {
      $all(".tool-btn[data-mode2]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      redraw.mode = btn.dataset.mode2;
    });
  });
  $("#undoPointBtn2").addEventListener("click", () => {
    if (redraw.mode === "chamber") { redraw.chamber.closed = false; redraw.chamber.pop(); }
    else redraw[redraw.mode].pop();
    renderDraw($("#redrawCanvas"), redraw);
  });
  // xoá RIÊNG phần đang chọn (ROI hoặc 1 trong 2 cổng) — giữ nguyên 2 phần còn lại
  $("#clearOneBtn2").addEventListener("click", () => {
    if (redraw.mode === "chamber") { redraw.chamber = []; redraw.chamber.closed = false; }
    else redraw[redraw.mode] = [];
    renderDraw($("#redrawCanvas"), redraw);
  });
  // xoá TẤT CẢ, vẽ lại từ đầu cả 3 phần
  $("#clearAllBtn2").addEventListener("click", () => {
    redraw.chamber = []; redraw.green = []; redraw.pink = [];
    renderDraw($("#redrawCanvas"), redraw);
  });
  $("#closePolygonBtn2").addEventListener("click", () => {
    if (redraw.chamber.length >= 3) { redraw.chamber.closed = true; renderDraw($("#redrawCanvas"), redraw); }
  });
  attachCanvasEvents($("#redrawCanvas"), redraw, () => renderDraw($("#redrawCanvas"), redraw), null);

  $("#cancelRedrawBtn").addEventListener("click", async () => {
    $("#redrawOverlay").classList.add("hidden");
    if (jobId) await fetch(`/api/job/${jobId}/resume`, { method: "POST" });
  });

  $("#applyRedrawBtn").addEventListener("click", async () => {
    const chamberOk = redraw.chamber.closed && redraw.chamber.length >= 3;
    const greenOk = redraw.green.length === 2;
    const pinkOk = redraw.pink.length === 2;
    if (!(chamberOk && greenOk && pinkOk)) {
      alert("Vui lòng vẽ đủ vùng chuồng (ROI) + 2 cổng trước khi áp dụng.");
      return;
    }
    const canvas = $("#redrawCanvas");
    const payload = {
      chamber_polygon: toNormPoints(redraw.chamber, canvas),
      green_line: toNormPoints(redraw.green, canvas),
      pink_line: toNormPoints(redraw.pink, canvas),
    };
    await fetch(`/api/job/${jobId}/geometry`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    lastGeometryNorm = payload;
    await fetch(`/api/job/${jobId}/resume`, { method: "POST" });
    $("#redrawOverlay").classList.add("hidden");
  });

})();

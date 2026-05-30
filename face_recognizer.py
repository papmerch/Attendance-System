"""
Employee Attendance System using Facial Recognition

Pipeline:
  1. Detect faces in each frame (OpenCV DNN + Caffe SSD)
  2. For each detected face, extract a 512-d embedding (MobileFaceNet ONNX)
  3. Compare embedding against enrolled known faces (cosine similarity)
  4. Display the person's name if a match is found above the threshold
  5. Automatically log attendance to attendance/Attendance.csv
  6. View live attendance dashboard at http://localhost:8080/attendance

Model files needed (in models/):
  - deploy.prototxt          (Caffe detection model architecture)
  - res10_300x300_ssd_iter_140000.caffemodel  (Caffe detection weights)
  - w600k_mbf.onnx           (MobileFaceNet recognition model, 13.6 MB)

Known faces (in known_faces/):
  - Place images in a subdirectory: known_faces/your_name/photo.jpg
  - Or directly: known_faces/your_name.jpg
  - Use MULTIPLE photos of yourself with different expressions for best results!

Attendance:
  - Recognized employees are logged once per cooldown window (default 4h)
  - View today's attendance: http://localhost:8080/attendance
  - View monthly report:    http://localhost:8080/report
  - CSV stored at:          attendance/Attendance.csv
"""

import cv2
import sys
import os
import time
import datetime
import threading
import socketserver
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from attendance_manager import AttendanceManager


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Detection model (Caffe SSD)
PROTOTXT_PATH = os.path.join(BASE_DIR, "models", "deploy.prototxt")
DETECTION_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel"
)

# Recognition model (MobileFaceNet ONNX)
RECOGNITION_MODEL_PATH = os.path.join(BASE_DIR, "models", "w600k_mbf.onnx")

# Known faces directory
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")

# Skip these subdirectories during enrollment
# LFW is skipped by default so its 5749 faces don't cause false name changes
EXCLUDED_DIRS = {"lfw"}

# Cache file for pre-computed embeddings
EMBEDDINGS_CACHE_PATH = os.path.join(KNOWN_FACES_DIR, "_embeddings_cache.npz")
CACHE_VERSION = 3

# Thresholds
DETECTION_CONFIDENCE = 0.5
RECOGNITION_THRESHOLD = 0.35

RECOG_INPUT_SIZE = 112

# Attendance tracking
attendance_mgr = AttendanceManager()
recent_attendance = {}
_attendance_lock = threading.Lock()

# Attendance cooldown (seconds) for on-screen flash messages
ATTENDANCE_FLASH_SECONDS = 5


# -----------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------
def load_detection_model(prototxt_path, model_path):
    if not os.path.exists(prototxt_path):
        raise FileNotFoundError(f"Detection prototxt not found: {prototxt_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Detection model not found: {model_path}")
    print("[INFO] Loading detection model (Caffe SSD)...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print("[INFO] Detection model loaded.")
    return net


def load_recognition_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Recognition model not found: {model_path}\n"
            "Download it:\n"
            "  curl -L -o /tmp/buffalo_sc.zip \\\n"
            "    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip\n"
            "  unzip -j /tmp/buffalo_sc.zip w600k_mbf.onnx -d models/"
        )
    print("[INFO] Loading recognition model (MobileFaceNet ONNX)...")
    net = cv2.dnn.readNetFromONNX(model_path)
    print("[INFO] Recognition model loaded.")
    return net


# -----------------------------------------------------------------------
# Face Embedding Extraction
# -----------------------------------------------------------------------
def get_face_embedding(frame, box, recognition_net):
    x1, y1, x2, y2 = box
    margin_x = int((x2 - x1) * 0.1)
    margin_y = int((y2 - y1) * 0.1)
    h, w = frame.shape[:2]
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w - 1, x2 + margin_x)
    y2 = min(h - 1, y2 + margin_y)

    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        return np.zeros(512, dtype=np.float32)

    blob = cv2.dnn.blobFromImage(face_roi, 1.0 / 255.0,
                                 (RECOG_INPUT_SIZE, RECOG_INPUT_SIZE), swapRB=True)
    recognition_net.setInput(blob)
    embedding = recognition_net.forward().flatten()

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


# -----------------------------------------------------------------------
# Face Enrollment — stores MULTIPLE embeddings per person
# -----------------------------------------------------------------------
def enroll_known_faces(known_faces_dir, detection_net, recognition_net):
    # Check cache
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        try:
            data = np.load(EMBEDDINGS_CACHE_PATH, allow_pickle=True)
            cached_version = data.get("version", 0)
        except Exception:
            cached_version = 0

        if cached_version != CACHE_VERSION:
            print(f"[INFO] Cache version mismatch, re-scanning...")
        else:
            cache_mtime = os.path.getmtime(EMBEDDINGS_CACHE_PATH)
            newest_image = 0
            for root, _, files in os.walk(known_faces_dir):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        newest_image = max(newest_image,
                                           os.path.getmtime(os.path.join(root, f)))
            if newest_image < cache_mtime:
                print("[INFO] Loading cached embeddings...")
                names = data["names"].tolist()
                embeddings = data["embeddings"]
                known = {}
                for n, emb in zip(names, embeddings):
                    known.setdefault(n, []).append(emb)
                print(f"[INFO] Loaded {len(known)} people ({len(names)} images) from cache.")
                return known

    # Scan for images, skipping excluded dirs
    print("[INFO] Scanning known_faces/ for images...")
    image_paths = []
    skipped_lfw = False
    for root, dirs, files in os.walk(known_faces_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        if any(excl in root.split(os.sep) for excl in EXCLUDED_DIRS):
            skipped_lfw = True
            continue
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("_"):
                image_paths.append(os.path.join(root, f))

    if skipped_lfw:
        print("  [INFO] Skipped 'lfw' directory to avoid false name changes.")
        print("  [INFO] Only your personal photos will be used for recognition.")

    if not image_paths:
        print("[WARNING] No images found in known_faces/.")
        print("  Create a subdirectory: mkdir known_faces/you && cp photo.jpg known_faces/you/")
        print("  Or place file: cp photo.jpg known_faces/your_name.jpg")
        return {}

    # Enroll every image — keep all embeddings per person
    known_faces = {}
    for img_path in image_paths:
        rel = os.path.relpath(img_path, known_faces_dir)
        parts = rel.replace("\\", "/").split("/")

        if len(parts) >= 2 and os.path.isdir(os.path.join(known_faces_dir, parts[0])):
            name = parts[0]
        else:
            name = os.path.splitext(parts[-1])[0]

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] Could not read {rel}, skipping.")
            continue

        h, w = img.shape[:2]
        if h < 20 or w < 20:
            continue

        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
        detection_net.setInput(blob)
        detections = detection_net.forward()

        best_face = None
        best_conf = 0
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > best_conf and conf >= DETECTION_CONFIDENCE:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                best_face = box.astype("int")
                best_conf = conf

        if best_face is None:
            print(f"  [WARN] No face detected in {rel}, skipping.")
            continue

        embedding = get_face_embedding(img, best_face, recognition_net)
        known_faces.setdefault(name, []).append(embedding)
        count = len(known_faces[name])
        print(f"  Enrolled: {name} (photo {count}, detection: {best_conf:.0%})")

    # Cache flattened format
    if known_faces:
        flat_names, flat_embs = [], []
        for name, embs in known_faces.items():
            for emb in embs:
                flat_names.append(name)
                flat_embs.append(emb)
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        np.savez_compressed(EMBEDDINGS_CACHE_PATH, version=CACHE_VERSION,
                            names=flat_names, embeddings=np.array(flat_embs))
        print(f"[INFO] Cached {len(flat_names)} embeddings ({len(known_faces)} people).")

    print(f"[INFO] Enrolled {len(known_faces)} person(s):")
    for name in sorted(known_faces):
        print(f"       - {name} ({len(known_faces[name])} photo{'s' if len(known_faces[name]) > 1 else ''})")
    return known_faces


# -----------------------------------------------------------------------
# Face Matching — compares against ALL embeddings per person
# -----------------------------------------------------------------------
def find_best_match(embedding, known_faces, threshold=RECOGNITION_THRESHOLD):
    if not known_faces:
        return "Unknown", 0.0

    best_name = "Unknown"
    best_score = 0.0

    for name, emb_list in known_faces.items():
        for known_emb in emb_list:
            sim = float(np.dot(embedding, known_emb))
            if sim > best_score:
                best_score = sim
                best_name = name

    return best_name, best_score


# -----------------------------------------------------------------------
# Attendance helpers
# -----------------------------------------------------------------------
def set_attendance_flash(name, message):
    with _attendance_lock:
        recent_attendance[name] = {"msg": message, "time": time.time()}

def cleanup_attendance_flash():
    now = time.time()
    with _attendance_lock:
        expired = [n for n, v in recent_attendance.items()
                   if now - v["time"] > ATTENDANCE_FLASH_SECONDS]
        for n in expired:
            del recent_attendance[n]

def get_attendance_flash(name):
    with _attendance_lock:
        entry = recent_attendance.get(name)
        if entry and time.time() - entry["time"] <= ATTENDANCE_FLASH_SECONDS:
            return entry["msg"]
    return None


# -----------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------
def annotate_frame(frame, detections, detection_net, recognition_net,
                   known_faces, det_conf_threshold, rec_threshold):
    height, width = frame.shape[:2]
    CONFIDENT = rec_threshold
    LOW = 0.10

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    detection_net.setInput(blob)
    detections = detection_net.forward()

    face_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < det_conf_threshold:
            continue

        face_count += 1
        box = detections[0, 0, i, 3:7] * [width, height, width, height]
        x1, y1, x2, y2 = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)

        embedding = get_face_embedding(frame, (x1, y1, x2, y2), recognition_net)
        name, score = find_best_match(embedding, known_faces, rec_threshold)

        if face_count <= 2 and known_faces:
            print(f"  match: '{name}' score={score:.3f}  ", end="")

        if score >= CONFIDENT:
            color = (0, 255, 0)
            is_new, att_msg = attendance_mgr.mark_attendance(name)
            if is_new:
                set_attendance_flash(name, att_msg)
            label = f"NAME: {name} ({score * 100:.1f}%)"
            att_display = get_attendance_flash(name) or "Already recorded"
        elif score >= LOW:
            color = (0, 255, 255)
            label = f"GUESS: {name}? ({score * 100:.1f}%)"
            att_display = None
        else:
            color = (0, 0, 255)
            label = f"NO MATCH ({score * 100:.1f}%)"
            att_display = None

        if att_display:
            att_color = (0, 200, 0) if "recorded" in att_display else (0, 255, 255)
            att_label = f"ATTEND: {att_display}"
            y_att = y2 + 20
            if y_att + 20 > height:
                y_att = y1 - 40
            (aw, ah), _ = cv2.getTextSize(att_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y_att - ah - 2),
                          (x1 + aw + 4, y_att + 2), att_color, -1)
            cv2.putText(frame, att_label, (x1 + 2, y_att),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = y1 - 10 if y1 - 10 > lh else y1 + lh + 10
        cv2.rectangle(frame, (x1, label_y - lh - 4),
                      (x1 + lw + 4, label_y + 4), color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return face_count


# -----------------------------------------------------------------------
# Debug: print top matches
# -----------------------------------------------------------------------
def run_match_debug(det_net, rec_net, known_faces):
    print("\n[DEBUG] Capturing one frame for match test...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Could not capture frame.")
        return

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    det_net.setInput(blob)
    detections = det_net.forward()

    found = False
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < DETECTION_CONFIDENCE:
            continue
        found = True
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        x1, y1, x2, y2 = box.astype("int")
        emb = get_face_embedding(frame, (x1, y1, x2, y2), rec_net)
        scores = []
        for name, emb_list in known_faces.items():
            for known_emb in emb_list:
                sim = float(np.dot(emb, known_emb))
                scores.append((sim, name))
        scores.sort(reverse=True)

        print(f"\n  Detection confidence: {conf:.2f}")
        print(f"  {'Rank':<6} {'Name':<30} {'Score':<8}")
        print(f"  {'-'*44}")
        for rank, (sim, name) in enumerate(scores[:5], 1):
            mark = " <<<" if rank == 1 else ""
            print(f"  {rank:<6} {name:<30} {sim:.4f}{mark}")
        print(f"\n  Threshold: {RECOGNITION_THRESHOLD}")
        if scores[0][0] >= RECOGNITION_THRESHOLD:
            print(f"  ✓ {scores[0][1]} is above threshold")
        else:
            print(f"  ✗ Best score ({scores[0][0]:.3f}) below threshold — press - to lower it")
        break

    if not found:
        print("  No face detected.")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print()
    print("=" * 60)
    print("  >>> EMPLOYEE ATTENDANCE SYSTEM <<<")
    print("  Recognized faces are logged to attendance/Attendance.csv")
    print("  Attendance dashboard at http://localhost:8080/attendance")
    print("=" * 60)
    print()

    try:
        det_net = load_detection_model(PROTOTXT_PATH, DETECTION_MODEL_PATH)
        rec_net = load_recognition_model(RECOGNITION_MODEL_PATH)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    known_faces = enroll_known_faces(KNOWN_FACES_DIR, det_net, rec_net)

    if "--match" in sys.argv or "--debug" in sys.argv:
        run_match_debug(det_net, rec_net, known_faces)
        return

    # MJPEG over HTTP — live video in browser
    latest_jpeg = [None]
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for webcam...", (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    _, ph_jpeg = cv2.imencode(".jpg", placeholder)
    latest_jpeg[0] = ph_jpeg.tobytes()

    class MJPEGHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                while True:
                    jpeg = latest_jpeg[0]
                    if jpeg is not None:
                        try:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                        except (BrokenPipeError, ConnectionResetError):
                            return
                    time.sleep(0.03)
            elif self.path == "/attendance":
                records = attendance_mgr.get_today_attendance()
                summary = attendance_mgr.get_summary(records)
                rows_html = ""
                for name in sorted(summary):
                    times = ", ".join(summary[name])
                    rows_html += f"<tr><td>{name}</td><td>{times}</td></tr>"
                html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Attendance - Today</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',sans-serif; background:#f5f5f5; padding:30px; }}
  h1 {{ color:#333; margin-bottom:20px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; box-shadow:0 2px 8px rgba(0,0,0,0.1); }}
  th,td {{ padding:12px 16px; text-align:left; border-bottom:1px solid #eee; }}
  th {{ background:#4a90d9; color:#fff; }}
  tr:hover {{ background:#f0f7ff; }}
  .nav {{ margin-bottom:20px; }}
  .nav a {{ color:#4a90d9; text-decoration:none; margin-right:15px; font-weight:600; }}
  .count {{ margin-top:15px; color:#666; }}
</style></head><body>
<div class="nav">
  <a href="/">&#8592; Live Feed</a>
  <a href="/attendance">Attendance (Today)</a>
  <a href="/report">Monthly Report</a>
</div>
<h1>Today's Attendance ({datetime.date.today()})</h1>
<table><thead><tr><th>Employee</th><th>Check-in Time(s)</th></tr></thead>
<tbody>{rows_html}</tbody></table>
<div class="count">Total present: {len(summary)}</div>
</body></html>"""
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(html.encode())
            elif self.path == "/report":
                now = datetime.datetime.now()
                year_month = now.strftime("%Y-%m")
                records = attendance_mgr.get_monthly_report(year_month)
                summary = attendance_mgr.get_summary(records)
                rows_html = ""
                for name in sorted(summary):
                    times = ", ".join(summary[name])
                    rows_html += f"<tr><td>{name}</td><td>{times}</td></tr>"
                html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Attendance - {year_month}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',sans-serif; background:#f5f5f5; padding:30px; }}
  h1 {{ color:#333; margin-bottom:20px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; box-shadow:0 2px 8px rgba(0,0,0,0.1); }}
  th,td {{ padding:12px 16px; text-align:left; border-bottom:1px solid #eee; }}
  th {{ background:#4a90d9; color:#fff; }}
  tr:hover {{ background:#f0f7ff; }}
  .nav {{ margin-bottom:20px; }}
  .nav a {{ color:#4a90d9; text-decoration:none; margin-right:15px; font-weight:600; }}
  .count {{ margin-top:15px; color:#666; }}
</style></head><body>
<div class="nav">
  <a href="/">&#8592; Live Feed</a>
  <a href="/attendance">Attendance (Today)</a>
  <a href="/report">Monthly Report</a>
</div>
<h1>Monthly Report - {year_month}</h1>
<table><thead><tr><th>Employee</th><th>Dates</th></tr></thead>
<tbody>{rows_html}</tbody></table>
<div class="count">Total employees: {len(summary)}</div>
</body></html>"""
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(html.encode())
            else:
                today_str = datetime.date.today().isoformat()
                html = f"""<!DOCTYPE html>
<html><head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ margin:0; background:#000; font-family:'Segoe UI',sans-serif; }}
  .nav {{ position:fixed; top:0; left:0; right:0; z-index:10;
          background:rgba(0,0,0,0.7); padding:10px 20px; text-align:center; }}
  .nav a {{ color:#4a90d9; text-decoration:none; margin:0 12px; font-weight:600; font-size:14px; }}
  .nav a:hover {{ color:#6ab0ff; }}
  img {{ width:100%; height:100vh; object-fit:contain; }}
</style></head><body>
<div class="nav">
  <a href="/">&#9679; Live Feed</a>
  <a href="/attendance">Attendance (Today)</a>
  <a href="/report">Monthly Report</a>
</div>
<img src="/stream">
</body></html>"""
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(html.encode())

        def log_message(self, format, *args):
            pass

    print()
    print(f"[INFO] Live video at:  http://localhost:8080")
    print(f"[INFO] Attendance:     http://localhost:8080/attendance")
    print(f"[INFO] Monthly Report: http://localhost:8080/report")
    print("[KEYS]  Press Ctrl+C to quit")
    print("        Use --threshold N to set threshold (e.g. --threshold 0.25)")
    print("        Use --match or --debug to print top-5 match scores for one frame")
    print()
    sys.stdout.flush()

    print("[INFO] Starting webcam...")
    capture_thread_alive = [True]

    def capture_loop():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam. Check connection.")
            capture_thread_alive[0] = False
            return

        threshold = RECOGNITION_THRESHOLD
        if "--threshold" in sys.argv:
            idx = sys.argv.index("--threshold")
            if idx + 1 < len(sys.argv):
                threshold = max(0.0, min(1.0, float(sys.argv[idx + 1])))

        while capture_thread_alive[0]:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            cleanup_attendance_flash()

            face_count = annotate_frame(
                frame, None, det_net, rec_net, known_faces,
                DETECTION_CONFIDENCE, threshold
            )

            att_today = len(attendance_mgr.get_today_attendance())
            cv2.putText(frame, f"Faces: {face_count}  Today: {att_today}  Thresh: {threshold:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            latest_jpeg[0] = jpeg.tobytes()

        cap.release()

    threading.Thread(target=capture_loop, daemon=True).start()

    class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    srv = ThreadedHTTPServer(("127.0.0.1", 8080), MJPEGHandler)

    import subprocess
    try:
        subprocess.Popen(["xdg-open", "http://localhost:8080"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        pass

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Quitting...")
    finally:
        capture_thread_alive[0] = False
        srv.shutdown()


if __name__ == "__main__":
    main()

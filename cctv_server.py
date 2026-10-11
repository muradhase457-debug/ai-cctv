"""
Traffic CCTV AI server  (YOLOv8 + ByteTrack + line counting + recording)

Install:   pip install ultralytics flask opencv-python
Run:       python cctv_server.py http://PHONE_IP:8080/video     (phone as IP camera)
           python cctv_server.py 0                              (PC webcam)
           python cctv_server.py traffic.mp4                    (video file)
Watch:     phone browser -> http://PC_IP:5000   (same Wi-Fi)
"""
import cv2, sys, time, threading, os
from flask import Flask, Response, jsonify
from ultralytics import YOLO

SRC = sys.argv[1] if len(sys.argv) > 1 else "0"
SRC = int(SRC) if SRC.isdigit() else SRC
CLS = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COL = {"person": (250, 139, 167), "bicycle": (248, 132, 192), "car": (113, 113, 248),
       "motorcycle": (60, 146, 251), "bus": (21, 204, 250), "truck": (250, 165, 96)}
LINE = 0.55  # counting line position (fraction of frame height)

model = YOLO("yolov8s.pt")  # first run downloads weights (needs internet once)
app = Flask(__name__)
lock = threading.Lock()
S = {"jpg": None, "rec": False, "fps": 0.0,
     "counts": {n: {"in": 0, "out": 0, "seen": 0} for n in CLS.values()}}


def worker():
    cap = cv2.VideoCapture(SRC)
    side, seen, writer = {}, set(), None
    os.makedirs("recordings", exist_ok=True)
    t0 = time.time()
    while True:
        ok, f = cap.read()
        if not ok:
            cap.release(); time.sleep(1); cap = cv2.VideoCapture(SRC); continue
        h, w = f.shape[:2]
        ly = int(h * LINE)
        r = model.track(f, persist=True, tracker="bytetrack.yaml",
                        classes=list(CLS), conf=0.3, verbose=False)[0]
        if r.boxes.id is not None:
            for b, i, c in zip(r.boxes.xyxy.int().tolist(), r.boxes.id.int().tolist(),
                               r.boxes.cls.int().tolist()):
                n = CLS[c]; cy = (b[1] + b[3]) // 2; s = cy > ly
                with lock:
                    if i not in seen:
                        seen.add(i); S["counts"][n]["seen"] += 1
                    if i in side and side[i] != s:
                        S["counts"][n]["in" if s else "out"] += 1
                side[i] = s
                col = COL[n]
                cv2.rectangle(f, (b[0], b[1]), (b[2], b[3]), col, 2)
                cv2.putText(f, f"{n} #{i}", (b[0], max(15, b[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.line(f, (0, ly), (w, ly), (11, 158, 245), 3)
        cv2.putText(f, time.strftime("%d.%m.%Y %H:%M:%S"), (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # recording
        if S["rec"] and writer is None:
            writer = cv2.VideoWriter(f"recordings/cctv-{int(time.time())}.mp4",
                                     cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        if not S["rec"] and writer is not None:
            writer.release(); writer = None
        if writer is not None:
            writer.write(f)
            cv2.circle(f, (w - 18, 14), 7, (0, 0, 255), -1)
        S["fps"] = 1 / max(time.time() - t0, 1e-3); t0 = time.time()
        S["jpg"] = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>CCTV</title><body style="margin:0;background:#0b0b0d;color:#eee;font:14px system-ui">
<img src="/video" style="width:100%;display:block">
<div style="padding:10px"><button id=r style="background:#f59e0b;border:0;border-radius:8px;padding:12px 16px;font-weight:700">⏺ REC</button>
<span id=f style="margin-left:10px;color:#aaa"></span>
<div style="background:#1a140a;border:1px solid #5a3d12;border-radius:10px;padding:12px;margin-top:10px">
Total unique: <b id=t style="font-size:26px">0</b><table id=tb style="width:100%;margin-top:6px"></table></div></div>
<script>
const r=document.getElementById('r');
r.onclick=()=>fetch('/rec',{method:'POST'});
async function u(){const d=await (await fetch('/stats')).json();let t=0,h='<tr style="color:#f59e0b"><td>Type<td>In<td>Out<td>Seen</tr>';
for(const k in d.counts){const o=d.counts[k];t+=o.seen;h+=`<tr><td>${k}<td>${o.in}<td>${o.out}<td>${o.seen}</tr>`}
document.getElementById('tb').innerHTML=h;document.getElementById('t').textContent=t;
document.getElementById('f').textContent=d.fps.toFixed(1)+' fps';
r.textContent=d.rec?'⏹ STOP':'⏺ REC';r.style.background=d.rec?'#ef4444':'#f59e0b'}
setInterval(u,1000);u();</script>"""


@app.route("/")
def index(): return PAGE


@app.route("/stats")
def stats():
    with lock: return jsonify(counts=S["counts"], rec=S["rec"], fps=S["fps"])


@app.route("/rec", methods=["POST"])
def rec():
    S["rec"] = not S["rec"]; return jsonify(rec=S["rec"])


@app.route("/video")
def video():
    def gen():
        while True:
            j = S["jpg"]
            if j: yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)

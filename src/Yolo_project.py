from ultralytics import YOLO
import cv2
import torch
import time
from collections import defaultdict

model = YOLO("yolov8s.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Using device:", device)

ALLOWED_CLASSES = ["cup", "chair", "cell phone", "book", "person",
                   "scissors", "keyboard", "toothbrush", "bottle", "fork"]

CONF = 0.55        
IOU  = 0.40        
IMGSZ = 960
TTL = 5            

name_map = model.names  # id -> label
allowed_ids = [i for i, n in name_map.items() if n in ALLOWED_CLASSES]
allowed_set = set(allowed_ids)

active_tracks = {}

prev_time = time.time()

results_stream = model.track(
    source=0,                      
    stream=True,
    persist=True,                  
    tracker="bytetrack.yaml",      
    imgsz=IMGSZ,
    conf=CONF,
    iou=IOU,
    classes=allowed_ids,           
    verbose=False,
    half=True if device == "cuda" else False
)

for r in results_stream:
    frame = r.orig_img.copy()
    boxes = r.boxes

    present_keys = set()

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy        
        clss = boxes.cls         
        confs = boxes.conf       
        ids   = boxes.id        

        n = len(xyxy)
        for i in range(n):
            cls_id = int(clss[i].item())
            if cls_id not in allowed_set:
                continue

            if ids is None or ids[i] is None:
                x1, y1, x2, y2 = map(int, xyxy[i].tolist())
                label = name_map[cls_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                cv2.putText(frame, f"{label} ...", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                continue

            tid = int(ids[i].item())
            x1, y1, x2, y2 = map(int, xyxy[i].tolist())
            conf = float(confs[i].item())

            key = (cls_id, tid)
            present_keys.add(key)
            active_tracks[key] = {"bbox": (x1, y1, x2, y2), "conf": conf, "miss": 0}

    to_drop = []
    for key, info in active_tracks.items():
        if key not in present_keys:
            info["miss"] += 1
            if info["miss"] > TTL:
                to_drop.append(key)
    for key in to_drop:
        del active_tracks[key]

    per_class_counts = defaultdict(int)
    for (cls_id, tid), info in active_tracks.items():
        x1, y1, x2, y2 = info["bbox"]
        label = name_map[cls_id]
        per_class_counts[label] += 1

        color = (0, 255, 0) if info["miss"] == 0 else (0, 200, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ID {tid}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    y = 30
    for obj in ALLOWED_CLASSES:
        c = per_class_counts.get(obj, 0)
        if c > 0:
            cv2.putText(frame, f"{obj}: {c}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 28

    fps = 1.0 / max(1e-6, (time.time() - prev_time))
    prev_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.imshow("YOLOv8s + ByteTrack — Stable IDs & Counts", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

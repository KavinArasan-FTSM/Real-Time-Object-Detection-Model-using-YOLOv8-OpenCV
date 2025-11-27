# YOLOv8 + ByteTrack Object Tracker

## Description
This project implements a **real-time object tracking system** using **YOLOv8** for detection and **ByteTrack** for stable ID tracking.  
It identifies objects from a webcam stream, assigns persistent IDs, and counts instances of specified object classes.

Built in **Python** using **PyTorch** and **OpenCV**.

---

## Features
- Real-time object detection and tracking  
- Stable IDs for objects across frames  
- Count of objects per class displayed on the frame  
- Filter by allowed object classes  
- FPS display for performance monitoring  

---

## Allowed Classes
The tracker currently tracks the following objects:
cup, chair, cell phone, book, person, scissors, keyboard, toothbrush, bottle, fork

---

## Tech Stack
- Python 3.10+  
- [YOLOv8](https://ultralytics.com/yolov8)  
- ByteTrack tracker  
- PyTorch  
- OpenCV  

---
## Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/<yourusername>/YOLOv8-Object-Tracker.git
cd YOLOv8-Object-Tracker

2. (Optional) Create and activate a virtual environment
# Linux/macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Download YOLOv8 weights
ultralytics YOLOv8s.pt

5. Run the tracker
python src/yolo_bytetrack.py
NOTE : Uses your default webcam (source=0), Press q to exit


from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x.pt")
model = YOLO('runs/detect/train7/weights/last.pt')

# Customize validation settings
validation_results = model.val(data="word_det.yaml", imgsz=1024, batch=2, conf=0.25, iou=0.4, device="4", plots=True)
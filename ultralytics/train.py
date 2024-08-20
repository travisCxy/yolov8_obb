import sys
sys.path.append("/mnt/data5/cxy/project/ultralytics")
from ultralytics import YOLO, RTDETR

# Load a model
# model = RTDETR("./ultralytics/cfg/models/rt-detr/rtdetr-resnet101.yaml")  # build a new model from YAML
# model = RTDETR('rtdetr-resnet101.pt')#.load("yolov8x.pt")  # build a new model from YAML
# model = YOLO("yolov8-rtdetrx.pt")  # build a new model from YAML


#model = YOLO("./ultralytics/yolov8x.pt")  # load a pretrained model (recommended for training)


#model = YOLO("./ultralytics/yolov8x-obb.pt")  # load a pretrained model (recommended for training)
#model = YOLO("runs/obb/best.pt")  # load a pretrained model (recommended for training)
#model = YOLO("runs/detect/train3/weights/best.pt")  # load a pretrained model (recommended for training)
#model = YOLO("./ultralytics/cfg/models/v8/yolov8.yaml", task="detect")  # build a new model from YAML
#model = YOLO("yolov10x.pt")  # build a new model from YAML
#model = YOLO("./ultralytics/cfg/models/v10/yolov10x_obb.yaml", task="obb").load("yolov10x.pt")  # build a new model from YAML
#model = YOLO("./runs/obb/obb_pretrain/best.pt")
model = YOLO("./runs/obb/train17/weights/last.pt")
# Train the model
#results = model.train(data="sjb_det.yaml", epochs=100, imgsz=1024)
results = model.train(data="sjb_det.yaml", epochs=50, imgsz=1280, resume=False, lr0=0.0002, momentum=0.9, warmup_epochs=0)
from ultralytics import YOLO, RTDETR

# Load a pretrained YOLOv8n model
#model = YOLO("yolov8x.pt")
#model = YOLO('runs/detect/train2/weights/last.pt')




#model = YOLO('runs/detect/train3/weights/best.pt')  #yolov8 word det best
#model = YOLO('runs/detect/train4/weights/best.pt')  #yolov8 sjb det best
#model = YOLO('runs/detect/train10/weights/last.pt')  #yolov8 sjb det best
#model = YOLO('runs/obb/train6/weights/best.pt')
#model = YOLO('runs/obb/train14/weights/best.pt')
#model = YOLO('runs/obb/best/best.pt')
model = YOLO('runs/obb/train17/weights/last.pt')
#model = YOLO('runs/obb/train31/weights/epoch41.pt')
#model = YOLO('runs/obb/train37/weights/epoch41.pt')
#model = YOLO('runs/obb/train46/weights/last.pt')
#model = YOLO('runs/obb/train47/weights/last.pt')
# Run inference on 'bus.jpg' with arguments
model.predict("/mnt/server_data2/dataset/word_det/SJB_DET_20240701/test_data/test.txt", save=True, imgsz=1024, conf=0.25, iou=0.25, device="4", save_txt=True, save_conf=True)
#model.predict("/mnt/server_data2/dataset/word_det/SJB_DET_20240701/2024-02-27.txt", save=True, imgsz=1024, conf=0.25, iou=0.25, device="4", save_txt=True, save_conf=True)



# Đây là code train trên Kaggle nhưng chắc vẫn chạy được bình thường thôi
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='VisDrone.yaml', 
    epochs=30,            
    imgsz=640,            
    batch=32,             
    device=[0, 1],        # sửa chỗ này nếu máy chỉ có 1 GPU, đang để 2 vì Kaggle có 2 con T4
    project='drone_project', 
    name='yolov8n_visdrone'  
)
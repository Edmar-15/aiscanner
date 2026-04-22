from ultralytics import YOLO

print("Starting training...")

model = YOLO("yolov8n-cls.pt")

results = model.train(
    data="dataset",
    epochs=20,
    imgsz=224
)
print(" Training finished!")
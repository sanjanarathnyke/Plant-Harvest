from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Predict using custom image
results = model.predict(
    source="2007m_JPEG.rf.bf913b95d80f7b274ca2eaa3c3fbe472.jpg",
    save=True,
    imgsz=640
)

print("Prediction completed!")

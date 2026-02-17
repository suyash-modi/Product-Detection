from ultralytics import YOLO

# 1. Download and Load the Extra Large model
print("⏳ Downloading model (this may take a moment)...")
model = YOLO('yolov8x.pt')

# 2. Export to OpenVINO with High Resolution
print("🚀 Exporting to OpenVINO format...")
model.export(format='openvino', imgsz=1280)

print("✅ Done! check for the folder 'yolov8x_openvino_model'")
import cv2
import numpy as np
from openvino import Core

class ProductDetector:
    def __init__(self, model_path):
        self.core = Core()
        print(f"[AI] Loading High-Precision Model: {model_path}...")
        self.model = self.core.read_model(model=model_path)
        self.compiled = self.core.compile_model(self.model, "CPU")
        self.output_layer = self.compiled.output(0)

    def infer(self, frame):
        # 1. High-Res Resize (1280x1280)
        target_size = 1280 
        
        # [FIX] Convert BGR (OpenCV default) to RGB (YOLO requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        resized = cv2.resize(rgb_frame, (target_size, target_size))
        
        # 2. HWC -> CHW and Normalize (0-1)
        blob = np.expand_dims(resized.transpose(2, 0, 1), 0).astype(np.float32)
        blob /= 255.0
        
        # 3. Inference
        return self.compiled([blob])[self.output_layer]
import numpy as np
import cv2

def create_zones(raw_output, original_shape, labels, conf_thresh=0.50):
    """
    Parses output, removes people, and EXPANDS boxes for interaction zones.
    """
    # 1. Configuration: How much bigger should the box be?
    # PADDING = 0.15 means "Add 15% size to all sides"
    # This is better than fixed pixels because it scales with product size.
    PADDING_PERCENT = 0.15 

    # Standard Parsing ...
    detections = raw_output[0].transpose()
    zones = []
    boxes = []
    scores = []
    class_ids = []

    h_orig, w_orig = original_shape[:2]
    x_factor = w_orig / 1280
    y_factor = h_orig / 1280

    for row in detections:
        classes_scores = row[4:]
        max_score = np.max(classes_scores)

        if max_score > conf_thresh:
            class_id = np.argmax(classes_scores)
            
            # IGNORE PEOPLE (Keep this rule!)
            if class_id == 0: continue

            cx, cy, w, h = row[0], row[1], row[2], row[3]
            
            # --- 🔥 THE FIX: EXPAND THE BOX ---
            # Calculate padding based on size
            pad_w = w * PADDING_PERCENT
            pad_h = h * PADDING_PERCENT

            # New Expanded Dimensions
            new_w = w + (pad_w * 2)
            new_h = h + (pad_h * 2)
            
            # Calculate new top-left corner (shifting it up and left)
            # Note: We must ensure we don't go off-screen (< 0)
            left = int((cx - new_w/2) * x_factor)
            top = int((cy - new_h/2) * y_factor)
            width = int(new_w * x_factor)
            height = int(new_h * y_factor)

            # Safety Check: Don't draw boxes outside the image
            left = max(0, left)
            top = max(0, top)
            width = min(w_orig - left, width)
            height = min(h_orig - top, height)

            boxes.append([left, top, width, height])
            scores.append(float(max_score))
            class_ids.append(class_id)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, 0.30)

    if len(indices) > 0:
        for i in indices.flatten():
            lbl = labels.get(str(class_ids[i]), "product")
            zones.append({
                "product": lbl,
                "confidence": round(scores[i], 2),
                "bbox": boxes[i] # This is now the Expanded Box
            })

    return zones
"""Extract (crop) product images from frame using zone bounding boxes."""
import os
import cv2


def extract_product_crops(frame, zones, output_dir, prefix="product"):
    """
    Crop each zone from the frame and save as image files.
    
    Args:
        frame: BGR image (OpenCV format)
        zones: List of zone dicts with 'bbox' [x, y, w, h]
        output_dir: Directory to save cropped images
        prefix: Filename prefix for saved images
    
    Returns:
        List of zones with added 'crop_path' key for each zone
    """
    os.makedirs(output_dir, exist_ok=True)
    h_frame, w_frame = frame.shape[:2]
    
    for i, zone in enumerate(zones):
        x, y, w, h = zone['bbox']
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        
        if w <= 0 or h <= 0:
            zone['crop_path'] = None
            continue

        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            zone['crop_path'] = None
            continue

        product_slug = zone.get('product', 'item').replace(' ', '_').lower()
        safe_name = "".join(c for c in product_slug if c.isalnum() or c in '_-')
        filename = f"{prefix}_{safe_name}_{i}_{y}_{x}.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, crop)
        zone['crop_path'] = path

    return zones

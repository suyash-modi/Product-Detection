import cv2

def open_video(source=0):
    # If source is a number (0, 1), convert to int for webcam
    # If it's a string ("rtsp://..."), keep it as string
    src = int(source) if str(source).isdigit() else source
    
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    
    return cap
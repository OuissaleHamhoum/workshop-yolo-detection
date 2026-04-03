import cv2
from IPython.display import display, Image
import ipywidgets as widgets
from threading import Thread
import time

# Widget to display frames
img_widget = widgets.Image(format='jpeg')
display(img_widget)

cap = cv2.VideoCapture(0)  # 0 = default webcam

running = True

def stream():
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        img_widget.value = buffer.tobytes()
        time.sleep(0.03)  # ~30 FPS

# Start stream in background thread
t = Thread(target=stream)
t.start()
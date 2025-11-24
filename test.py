import time
import numpy as np
from ultralytics import YOLO
import cv2

model = YOLO('gestures.pt')

# 预热模型
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
_ = model(dummy)
print("Model pre-warmed")

cap =cv2.VideoCapture(0)
start_time = 0.0
end_time = 0.0

for i in range(100):
    if i==0:
        start_time = time.time()
    ret, frame = cap.read()
    if ret:
        results = model(frame)
    if i==99:
        end_time = time.time()

print(f'处理109帧时间开销{(end_time-start_time)*1000}ms')


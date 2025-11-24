import cv2
import socket
import struct
import threading
import queue
from ultralytics import YOLO
import numpy as np
import time
import pickle

JETSON_IP = "10.249.84.43"
JETSON_PORT = 5000

cap = cv2.VideoCapture(0)
model = YOLO("gestures.pt")  # 本地 YOLO

# 预热模型
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
_ = model(dummy)
print("Model pre-warmed")

frame_queue = queue.Queue()
local_queue = queue.Queue()
jetson_queue = queue.Queue()
result_queue = queue.Queue()

frame_id = 0
next_fid = 0
buffer = {}
lock = threading.Lock()
start_time = 0.0
flag = True

# 捕获摄像头帧
def capture_thread():
    global frame_id, flag, start_time
    while True:
        if flag:
            start_time = time.time()
            flag = False
        ret, frame = cap.read()
        if not ret:
            continue
        frame_queue.put((frame_id, frame))
        frame_id += 1

# 分发帧到本地或 Jetson
def dispatcher_thread():
    while True:
        fid, frame = frame_queue.get()
        if fid % 5 == 4:
            jetson_queue.put((fid, frame))
        else:
            local_queue.put((fid, frame))

# 本地推理
def local_worker():
    while True:
        fid, frame = local_queue.get()
        results = model(frame)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detections.append((cls_name, conf, (x1, y1, x2, y2)))
        result_queue.put((fid, detections))

# Socket 接收函数
def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# Jetson 推理
def jetson_worker():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((JETSON_IP, JETSON_PORT))
    print("[JETSON] connected")

    while True:
        fid, frame = jetson_queue.get()

        # 转 JPEG bytes
        _, jpg = cv2.imencode('.jpg', frame)
        frame_bytes = jpg.tobytes()

        # 发送 fid + frame_bytes
        data = pickle.dumps((fid, frame_bytes), protocol=pickle.HIGHEST_PROTOCOL)
        msg = struct.pack(">I", len(data)) + data
        s.sendall(msg)

        # 接收 Jetson 推理结果
        raw_len = recvall(s, 4)
        if not raw_len:
            break
        size = struct.unpack(">I", raw_len)[0]
        data = recvall(s, size)
        fid_ret, det = pickle.loads(data)
        result_queue.put((fid_ret, det))

# 顺序输出
def reorder_output():
    global next_fid, buffer
    while True:
        fid, det = result_queue.get()
        with lock:
            buffer[fid] = det

            if next_fid not in buffer and next_fid == 0:
                next_fid = min(buffer.keys())

            while next_fid in buffer:
                good_result = buffer.pop(next_fid)
                # 输出 fid + 精简结果
                print(f"[OUTPUT] fid={next_fid}, det={good_result}")
                if next_fid == 99:
                    end_time = time.time()
                    print(f'处理100帧时间开销{(end_time-start_time)*1000}ms')
                next_fid += 1

threading.Thread(target=capture_thread, daemon=True).start()
threading.Thread(target=dispatcher_thread, daemon=True).start()
threading.Thread(target=local_worker, daemon=True).start()
threading.Thread(target=jetson_worker, daemon=True).start()
threading.Thread(target=reorder_output, daemon=True).start()

print("Running...")
threading.Event().wait()

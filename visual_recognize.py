import cv2
from ultralytics import YOLO

import socket
# 连接 Webots 控制器
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 8000))  # Webots 端监听 8000 端口

model = YOLO('gestures.pt')
cap = cv2.VideoCapture(0)

last_cmd = "NONE"

def send_if_changed(cmd):
    global last_cmd
    if cmd != last_cmd:
        client.send(cmd.encode())
        last_cmd = cmd

while True:
    ret, frame = cap.read()
    if ret:
        results = model(frame)
        print(len(results[0].boxes))
        cv2.imshow('frame', frame)
        for box in results[0].boxes:
            cls_id = int(box.cls[0])  # 类别 ID（int）
            conf = float(box.conf[0])  # 置信度（0~1 float）
            cls_name = results[0].names[cls_id]  # 类别名称

            if conf >0.9:
                if cls_name == "like":
                    send_if_changed("UP")
                elif cls_name == "dislike":
                    send_if_changed("DOWN")
                elif cls_name == "grabbing":
                    send_if_changed("GRAB")
                elif cls_name == "palm":
                    send_if_changed("RELEASE")
                elif cls_name == "fist":
                    send_if_changed("STOP")

            print(f"类别: {cls_name}, 置信度: {conf:.2f}")
        annotated_frame = results[0].plot()  # BGR numpy array

        cv2.imshow('frame', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
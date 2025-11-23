import cv2
from ultralytics import YOLO
from sliding_window import SlidingWindow
import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 8888))   # Webots controller 监听 8888
print("Connected to Webots!")

def send_cmd(cmd):
    client.send(cmd.encode())

last_sent = None

model = YOLO('gestures.pt')
cap = cv2.VideoCapture(0)
sw = SlidingWindow(window_size=10, stable_threshold=9)

last_hand = None
counter = 0

def iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0, x2-x1)* max(0, y2-y1)
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = area1 + area2 - inter

    return inter / union if union != 0 else 0.0

while True:
    ret, frame = cap.read()
    if ret:
        results = model(frame)

        boxes = results[0].boxes

        if len(boxes) > 0:
            boxes_list = []
            for b in boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                boxes_list.append((area, b, xyxy))

            boxes_list.sort(key=lambda x: x[0], reverse=True)
            _, this_hand, this_xyxy = boxes_list[0]

            if last_hand is None:
                last_hand = this_hand
            else:
                last_xyxy = last_hand.xyxy[0].cpu().numpy()
                iou_value = iou(this_xyxy, last_xyxy)

                if iou_value > 0.4:  # 稳定阈值
                    last_hand = this_hand

            cls_id = int(last_hand.cls[0])
            conf = float(last_hand.conf[0])
            cls_name = results[0].names[cls_id]

            if conf > 0.9:
                sw.add(cls_name)

            # annotated_frame = results[0].plot()
            annotated_frame = frame.copy()
            lx1, ly1, lx2, ly2 = map(int, last_hand.xyxy[0].cpu().numpy())
            cv2.rectangle(annotated_frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 3)
            cv2.putText(annotated_frame, "CONTROL HAND",
                        (lx1, ly1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
        else:
            annotated_frame = frame

        stable_gesture = sw.get_stable()

        if stable_gesture != last_sent:
            if stable_gesture == "like":
                send_cmd("UP")
            elif stable_gesture == "dislike":
                send_cmd("DOWN")
            elif stable_gesture == "palm":
                send_cmd("OPEN")
            elif stable_gesture == "grabbing":
                send_cmd("CLOSE")

            last_sent = stable_gesture

        cv2.putText(annotated_frame, f'Stable: {stable_gesture}',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.imshow('frame', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

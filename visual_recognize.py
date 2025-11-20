import cv2
from ultralytics import YOLO
from sliding_window import SlidingWindow

model = YOLO('gestures.pt')
cap = cv2.VideoCapture(0)
sw = SlidingWindow(window_size=10, stable_threshold=9)

while True:
    ret, frame = cap.read()
    if ret:
        results = model(frame)

        boxes = results[0].boxes

        if len(boxes) > 0:
            best_box = max(boxes, key=lambda b: float(b.conf[0]))

            cls_id = int(best_box.cls[0])
            conf = float(best_box.conf[0])
            cls_name = results[0].names[cls_id]

            if conf > 0.9:
                sw.add(cls_name)

            annotated_frame = results[0].plot(
                boxes=[best_box]
            )
        else:
            annotated_frame = frame

        stable_gesture = sw.get_stable()


        cv2.putText(annotated_frame, f'Stable: {stable_gesture}',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.imshow('frame', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

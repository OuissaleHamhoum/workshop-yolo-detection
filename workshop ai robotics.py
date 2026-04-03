import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()  # take a picture
    if not ret:
        break

    h, w, _ = frame.shape

    results = model(frame)

    for r in results:
        if len(r.boxes) == 0:
            continue

        x1, y1, x2, y2 = map(int, r.boxes.xyxy[0])
        cls_name = model.names[int(r.boxes.cls[0])]

        if cls_name == "cell phone":
            box_center = (x1 + x2) // 2

            if box_center < w // 3:
                position = "LEFT"
            elif box_center > w * 2 // 3:
                position = "RIGHT"
            else:
                position = "CENTER"

            print("Cell phone position:", position)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, cls_name + " " + position, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Divide frame into 3 zones
    line1 = int(w / 3)
    line2 = int(2 * w / 3)
    cv2.line(frame, (line1, 0), (line1, h), (255, 0, 0), 2)
    cv2.line(frame, (line2, 0), (line2, h), (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Workshop - YOLO Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
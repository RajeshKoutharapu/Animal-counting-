import cv2
from ultralytics import YOLO

model = YOLO('resources/weights/yolov8m-sheep.pt')
file_path = "resources/videos/aerial-sheep.mp4"
cap = cv2.VideoCapture(file_path)
unique_id = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, tracker="bytetrack.yaml", persist=True)
    img = results[0].plot()

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        unique_id.update(ids)

    cv2.putText(img, f'Sheep Count: {len(unique_id)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Tracking', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

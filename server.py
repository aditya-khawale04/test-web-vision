from flask import Flask, Response
import cv2
import torch
from shapely.geometry import Point, Polygon

app = Flask(__name__)

# Load YOLOv5s model
print("🔁 Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()
print("✅ YOLOv5 model loaded.")

# Define polygon area (example: a quadrilateral zone)
# You can adjust the points (x, y) as per your frame size
polygon_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
polygon = Polygon(polygon_points)

# Open webcam (or video file)
video_source = 0
cap = cv2.VideoCapture(video_source)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)

        # Draw the polygon danger zone
        pts = cv2.UMat(np.array(polygon_points, np.int32)).get()
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        alert_triggered = False

        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 0:  # person class
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                person_center = Point(cx, cy)

                if not polygon.contains(person_center):
                    alert_triggered = True
                    cv2.putText(frame, "❗ OUTSIDE ZONE", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Draw person box and center point
                label = f"Person {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if alert_triggered:
            cv2.putText(frame, "⚠ ALERT: Person outside danger zone!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def home():
    return "Go to /video_feed to see the stream."

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    import numpy as np
    app.run(host='0.0.0.0', port=5001, debug=True)

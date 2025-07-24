from flask import Flask, Response
import cv2
import torch

app = Flask(__name__)

# Load YOLOv5s model from torch hub
print("🔁 Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()
print("✅ YOLOv5 model loaded.")

# Open webcam or video file
video_source = 0  # Can be changed to "video.mp4" or other filename
cap = cv2.VideoCapture(video_source)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLOv5 detection
        results = model(img_rgb)

        # Draw bounding boxes for persons
        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 0:  # Class 0 = person
                x1, y1, x2, y2 = map(int, box)
                label = f"Person {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def home():
    return "Video stream running. Go to /video_feed to view."

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

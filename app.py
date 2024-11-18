from flask import Flask, request, render_template, redirect, url_for, send_file, jsonify
import os
import cv2
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = YOLO('resources/weights/yolov8m-sheep.pt')
unique_id = set()
processing_status = {'status': 'waiting', 'count': 0}

def process_video(file_path):
    global processing_status, unique_id
    cap = cv2.VideoCapture(file_path)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.mp4')
    out = None

    unique_id.clear()  # Clear previous IDs to start fresh

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, tracker="bytetrack.yaml", persist=True)
        img = results[0].plot()

        # Update unique ID set if boxes are detected
        if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            unique_id.update(ids)

        # Display the count on the frame
        cv2.putText(img, f'Sheep Count: {len(unique_id)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Initialize VideoWriter if not already done
        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width, _ = img.shape
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
        out.write(img)

    cap.release()
    if out:
        out.release()

    processing_status = {'status': 'completed', 'count': len(unique_id)}
    return output_path


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        video = request.files['video']
        if not video:
            return 'No video uploaded!', 400

        filename = secure_filename(video.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(file_path)

        # Start video processing
        process_video(file_path)
        return jsonify({'status': 'processing'}), 200

    return render_template('index.html')

@app.route('/status', methods=['GET'])
def status():
    return jsonify(processing_status)

@app.route('/output')
def serve_output_video():
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.mp4')
    if os.path.exists(output_path):
        return send_file(output_path, mimetype='video/mp4', as_attachment=False, conditional=True)
    else:
        return "Output video not found", 404


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)

from flask import Flask, render_template, Response, jsonify
import cv2
import os
from datetime import datetime
import pandas as pd
from deepface import DeepFace
import threading

app = Flask(__name__)
camera = cv2.VideoCapture(0)

known_faces_dir = "known_faces"
attendance_file = "attendance.csv"

known_faces = {}
if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)

for file in os.listdir(known_faces_dir):
    if file.endswith((".jpg", ".png", ".jpeg")):
        known_faces[file.split('.')[0]] = os.path.join(known_faces_dir, file)

if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(attendance_file, index=False)

last_frame = None

def generate_frames():
    global last_frame
    while True:
        success, frame = camera.read()
        if not success:
            break
        last_frame = frame.copy()
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    global last_frame
    if last_frame is None:
        return jsonify({"status": "error", "message": "No video frame captured"})

    try:
        result = DeepFace.find(img_path=last_frame, db_path=known_faces_dir, enforce_detection=False)

        if len(result) > 0 and not result[0].empty:
            name = os.path.basename(result[0].iloc[0]["identity"]).split(".")[0]
        else:
            name = "Unknown"

        df = pd.read_csv(attendance_file)
        if name != "Unknown" and name not in df["Name"].values:
            new_entry = pd.DataFrame([[name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]], columns=["Name", "Time"])
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(attendance_file, index=False)
            return jsonify({"status": "success", "name": name})
        else:
            return jsonify({"status": "error", "message": "Already marked or unknown"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/export_excel', methods=['GET'])
def export_excel():
    try:
        df = pd.read_csv(attendance_file)
        excel_path = "attendance.xlsx"
        df.to_excel(excel_path, index=False)
        return jsonify({"status": "success", "file": excel_path})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

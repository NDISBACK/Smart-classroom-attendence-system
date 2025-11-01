from flask import Flask, render_template, Response, request, jsonify, send_file
from flask_cors import CORS
import cv2
from deepface import DeepFace
import os
import base64
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Ensure folders exist
os.makedirs("known_faces", exist_ok=True)
os.makedirs("attendance", exist_ok=True)

attendance_file = "attendance/attendance.csv"
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(attendance_file, index=False)

# Utility: save attendance without duplicates
def mark_attendance(name):
    df = pd.read_csv(attendance_file)
    if name not in df["Name"].values:
        df = pd.concat([df, pd.DataFrame([[name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]],
                                         columns=["Name", "Time"])])
        df.to_csv(attendance_file, index=False)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/take_attendance", methods=["POST"])
def take_attendance():
    data = request.get_json()
    img_data = base64.b64decode(data['image'].split(',')[1])
    with open("temp.jpg", "wb") as f:
        f.write(img_data)

    known_faces = [f"known_faces/{f}" for f in os.listdir("known_faces") if f.endswith(".jpg")]
    for face_path in known_faces:
        try:
            result = DeepFace.verify("temp.jpg", face_path, model_name="Facenet", enforce_detection=False)
            if result["verified"]:
                name = os.path.splitext(os.path.basename(face_path))[0]
                mark_attendance(name)
                return jsonify({"message": f"✅ {name} marked present!"})
        except Exception as e:
            print("Error verifying:", e)
    return jsonify({"message": "❌ Unknown face"})

@app.route("/register_student", methods=["POST"])
def register_student():
    data = request.get_json()
    name = data['name']
    img_data = base64.b64decode(data['image'].split(',')[1])
    file_path = f"known_faces/{name}.jpg"
    with open(file_path, "wb") as f:
        f.write(img_data)
    return jsonify({"message": f"✅ {name} registered successfully!"})

@app.route("/export_excel")
def export_excel():
    df = pd.read_csv(attendance_file)
    excel_path = "attendance/attendance.xlsx"
    df.to_excel(excel_path, index=False)
    return send_file(excel_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

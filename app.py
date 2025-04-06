import cv2
import dlib
import numpy as np
import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# URL to download the predictor model
MODEL_URL = "https://huggingface.co/RushabhShah/facial_landmark_detection_model/resolve/main/landmark_detection.dat"
MODEL_PATH = "landmark_detection.dat"

# Download model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Model downloaded.")

# Load detector and predictor
download_model()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

def get_landmarks(image, face):
    landmarks = predictor(image, face)
    coords = np.zeros((68, 2), dtype=int)
    for i in range(68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return coords

def analyze_asymmetry(landmarks):
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    mouth = landmarks[48:68]
    left_eyebrow = landmarks[17:22]
    right_eyebrow = landmarks[22:27]
    jaw = landmarks[0:17]

    left_eye_height = np.mean(left_eye[:, 1])
    right_eye_height = np.mean(right_eye[:, 1])
    left_eyebrow_height = np.mean(left_eyebrow[:, 1])
    right_eyebrow_height = np.mean(right_eyebrow[:, 1])
    left_mouth_corner = mouth[0]
    right_mouth_corner = mouth[6]

    eye_asymmetry = abs(left_eye_height - right_eye_height)
    eyebrow_asymmetry = abs(left_eyebrow_height - right_eyebrow_height)
    mouth_asymmetry = abs(left_mouth_corner[1] - right_mouth_corner[1])

    face_width = jaw[-1][0] - jaw[0][0]
    total_asymmetry = (eye_asymmetry + eyebrow_asymmetry + mouth_asymmetry) / face_width

    DROOPING_THRESHOLD = 0.05
    return total_asymmetry, total_asymmetry > DROOPING_THRESHOLD

@app.route('/detect_facial_drooping', methods=['POST'])
def detect_facial_drooping():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image_path = "uploaded_image.jpg"
    file.save(image_path)

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return jsonify({"error": "No face detected"}), 400

    for face in faces:
        landmarks = get_landmarks(gray, face)
        asymmetry_score, is_drooping = analyze_asymmetry(landmarks)

        response = {
            "asymmetry_score": round(asymmetry_score, 4),
            "drooping_detected": bool(is_drooping),
            "message": "Facial Drooping Detected" if is_drooping else "No Drooping"
        }
        return jsonify(response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)

import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import os
import time
import urllib.request
import bz2
from flask import Flask, Response, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Check if the shape predictor file exists. If not, download it.
if not os.path.isfile("shape_predictor_68_face_landmarks.dat"):
    print("Downloading shape_predictor_68_face_landmarks.dat.bz2...")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
    print("Extracting shape_predictor_68_face_landmarks.dat.bz2...")
    with bz2.BZ2File("shape_predictor_68_face_landmarks.dat.bz2") as file:
        with open("shape_predictor_68_face_landmarks.dat", "wb") as f_out:
            f_out.write(file.read())

# Load models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants
EAR_THRESHOLD = 0.25
TIME_THRESHOLD = 10  # Time in seconds to trigger popup
POPUP_MESSAGE = "Warning! Eye strain detected."

FRAME_COUNT_THRESHOLD = 4  # Number of consecutive frames with low EAR to trigger tired state
BLINK_DURATION = 1  # Minimum duration (seconds) for a blink to be counted

# Variables
blink_count = 0
last_blink_time = time.time()
blinks_per_minute = 0
blink_start_time = time.time()
ear_below_threshold_start_time = None
popup_triggered = False
avg_ear = 0
screen_start_time = time.time()
tired_state_detected = False
frame_count = 0
system_paused = False  # Global variable to track system state

# Function to calculate EAR
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Gamma correction to adjust brightness
def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Preprocessing for low-light conditions
def preprocess_frame(frame):
    frame = gamma_correction(frame, gamma=1.5)  # Adjust brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray = cv2.equalizeHist(gray)  # Enhance contrast
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply GaussianBlur to reduce noise
    return gray

def generate_frames():
    global blink_count, last_blink_time, blinks_per_minute, blink_start_time, tired_state_detected, frame_count, ear_below_threshold_start_time, popup_triggered, avg_ear, system_paused
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        if system_paused:
            time.sleep(1)  # Pause processing
            continue

        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess the frame for low-light conditions
        gray = preprocess_frame(frame)

        # Detect faces in the frame
        faces = face_detector(gray)

        for face in faces:
            landmarks = landmark_predictor(gray, face)

            # Extract eye landmarks
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

            # Calculate EAR
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Blink Detection
            if avg_ear < EAR_THRESHOLD:
                if time.time() - last_blink_time >= BLINK_DURATION:
                    blink_count += 1
                    last_blink_time = time.time()

            # Check for tiredness based on EAR and blink frequency within first 3 minutes
            if avg_ear < EAR_THRESHOLD:
                if ear_below_threshold_start_time is None:
                    ear_below_threshold_start_time = time.time()

                elapsed_ear_time = time.time() - ear_below_threshold_start_time
                current_elapsed_time = time.time() - screen_start_time

                if elapsed_ear_time >= TIME_THRESHOLD or current_elapsed_time <= 10:
                    if blinks_per_minute < 10 or blinks_per_minute > 30:
                        popup_triggered = True
                        tired_state_detected = True
                        cv2.putText(frame, POPUP_MESSAGE, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(frame, "Tired (EAR & Blinking Risk)", (face.left(), face.top() - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        tired_state_detected = False
                        popup_triggered = False
                else:
                    tired_state_detected = False
                    popup_triggered = False
            else:
                # EAR is normal – reset tracking
                ear_below_threshold_start_time = None
                popup_triggered = False
                tired_state_detected = False
                cv2.putText(frame, "Alert", (face.left(), face.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display EAR value and blinking frequency
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Draw landmarks
            for (x, y) in left_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Calculate blinks per minute
        elapsed_time = time.time() - blink_start_time
        if elapsed_time >= 60:
            blinks_per_minute = blink_count
            blink_count = 0
            blink_start_time = time.time()

        # Display blinking frequency
        cv2.putText(frame, f"Blinks/min: {blinks_per_minute}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Convert the frame to a JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as part of the response for the video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global avg_ear, blinks_per_minute, tired_state_detected
    elapsed_time = int(time.time() - screen_start_time)  # Calculate screen time in seconds
    minutes = elapsed_time // 60  # Get minutes
    seconds = elapsed_time % 60  # Get remaining seconds

    return jsonify({
        "ear": avg_ear,
        "blinksPerMinute": blinks_per_minute,
        "status": "tired" if tired_state_detected else "alert",
        "screen_time": f"{minutes} min {seconds} sec"
    })

@app.route('/check_popup')
def check_popup():
    return jsonify({"show_popup": popup_triggered})

@app.route('/screen_time')
def screen_time():
    elapsed_time = int(time.time() - screen_start_time)
    return jsonify({
        "minutes": elapsed_time // 60,
        "seconds": elapsed_time % 60,
        "total_seconds": elapsed_time  # Add this for clarity
    })

@app.route('/reset_screen_time', methods=['POST'])
def reset_screen_time():
    global screen_start_time
    screen_start_time = time.time()
    return jsonify({"success": True})

@app.route('/toggle_system', methods=['POST'])
def toggle_system():
    global system_paused
    system_paused = not system_paused
    return jsonify({"success": True, "system_paused": system_paused})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

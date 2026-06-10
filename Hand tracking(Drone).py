import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import urllib.request
import os
import time
from djitellopy import Tello
import threading

# --- Download model if needed ---
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        model_path
    )
    print("Download complete.")

# --- Tello Setup ---
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

# Calibrate before takeoff
print("Place drone on flat surface. Calibrating in 5 seconds...")
time.sleep(5)
tello.send_command_with_return("command")
tello.send_command_with_return("imu")
print("Calibration complete.")

tello.streamon()
tello.takeoff()

# --- MediaPipe Setup ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.4
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# --- Gesture thresholding ---
THRESHOLD_FRAMES = 3
gesture_buffer = []
landed = False
last_confirmed = None

# --- Helpers ---

def fingers_up(lm):
    up = []
    up.append(lm[4].x < lm[3].x)
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        up.append(lm[tip].y < lm[pip].y)
    return up

def classify_gesture(lm):
    up = fingers_up(lm)
    count = sum(up)
    if count == 0:
        return "FIST"
    if count == 5 and lm[12].z < -0.07:
        return "PALM UP"
    if count == 5:
        return "OPEN PALM"
    if up[1] and not any(up[2:]):
        return "POINT UP"
    if up[1] and up[2] and not any(up[3:]):
        return "PEACE"
    if up[0] and not any(up[1:]):
        return "THUMBS UP"
    return f"{count} FINGERS"

def stable_gesture(raw_gesture):
    gesture_buffer.append(raw_gesture)
    if len(gesture_buffer) > THRESHOLD_FRAMES:
        gesture_buffer.pop(0)
    if len(gesture_buffer) == THRESHOLD_FRAMES and len(set(gesture_buffer)) == 1:
        return gesture_buffer[0]
    return None

def send_drone_command(gesture, pinch_px):
    global landed
    if pinch_px is not None and pinch_px < 40 and gesture != "PALM UP":
        tello.send_rc_control(0, 0, 0, 50)
        return
    if gesture == "OPEN PALM":
        tello.send_rc_control(0, 0, 0, 0)
    elif gesture == "FIST":
        tello.land()
        landed = True
    elif gesture == "PEACE":
        tello.send_rc_control(0, 0, 0, -50)
    elif gesture == "THUMBS UP":
        tello.send_rc_control(0, 0, 30, 0)
    elif gesture == "PALM UP":
        tello.send_rc_control(0, 0, -30, 0)
    elif gesture == "POINT UP":
        tello.send_rc_control(0, 20, 0, 0)
    else:
        tello.send_rc_control(0, 0, 0, 0)

def draw_landmarks(frame, lm, w, h):
    points = [(int(l.x * w), int(l.y * h)) for l in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 200, 150), 2)
    for pt in points:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)

def pinch_distance(lm, w, h):
    x1, y1 = int(lm[4].x * w), int(lm[4].y * h)
    x2, y2 = int(lm[8].x * w), int(lm[8].y * h)
    return math.hypot(x2 - x1, y2 - y1), (x1, y1), (x2, y2)

# --- Tello camera thread ---
tello_frame = None
tello_lock = threading.Lock()

def tello_camera_thread():
    global tello_frame
    frame_reader = tello.get_frame_read()
    while True:
        frame = frame_reader.frame
        if frame is not None:
            with tello_lock:
                tello_frame = frame.copy()

thread = threading.Thread(target=tello_camera_thread, daemon=True)
thread.start()

print("HandPilot running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    gesture = "No hand"
    pinch_px = None

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        draw_landmarks(frame, lm, w, h)
        gesture = classify_gesture(lm)

        dist, pt1, pt2 = pinch_distance(lm, w, h)
        pinch_px = int(dist)
        color = (0, 200, 100) if dist < 40 else (200, 200, 200)
        cv2.line(frame, pt1, pt2, color, 2)
        cv2.circle(frame, pt1, 6, color, -1)
        cv2.circle(frame, pt2, 6, color, -1)

    confirmed = stable_gesture(gesture)
    if confirmed and not landed:
        if confirmed != last_confirmed:
            send_drone_command(confirmed, pinch_px)
            last_confirmed = confirmed

    # Webcam HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 130), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Gesture: {gesture}",
                (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Confirmed: {confirmed or 'waiting...'}",
                (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2)

    if pinch_px is not None:
        pinch_label = "PINCHED" if pinch_px < 40 else f"{pinch_px}px"
        cv2.putText(frame, f"Pinch: {pinch_label}",
                    (12, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, "Q = quit",
                (12, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    cv2.imshow("HandPilot - Control", frame)

    with tello_lock:
        if tello_frame is not None:
            cv2.imshow("HandPilot - Drone View", tello_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if not landed:
            tello.land()
        break

cap.release()
cv2.destroyAllWindows()
tello.streamoff()

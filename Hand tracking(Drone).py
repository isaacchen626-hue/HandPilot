import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import urllib.request
import os

# --- Download model if needed ---
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        model_path
    )
    print("Download complete.")

# --- Setup ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.6
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# Hand connections to draw the skeleton
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

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

print("Hand Tracker running. Press Q to quit.")

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
        print(f"Wrist z: {lm[0].z:.3f}  Middle fingertip z: {lm[12].z:.3f}")
        draw_landmarks(frame, lm, w, h)
        gesture = classify_gesture(lm)

        dist, pt1, pt2 = pinch_distance(lm, w, h)
        pinch_px = int(dist)
        color = (0, 200, 100) if dist < 40 else (200, 200, 200)
        cv2.line(frame, pt1, pt2, color, 2)
        cv2.circle(frame, pt1, 6, color, -1)
        cv2.circle(frame, pt2, 6, color, -1)

    # HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 110), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Gesture: {gesture}",
                (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if pinch_px is not None:
        pinch_label = "PINCHED" if pinch_px < 40 else f"{pinch_px}px"
        cv2.putText(frame, f"Pinch:   {pinch_label}",
                    (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, "Q = quit",
                (12, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    cv2.imshow("Hand Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
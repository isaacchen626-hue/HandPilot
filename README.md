# HandPilot
A real-time hand gesture drone control system that translates natural hand movements into flight commands using computer vision.

## Demo
*Demo video coming soon*

## How It Works
HandPilot uses MediaPipe's hand landmark model to detect 21 points on your hand in real time via webcam. Each frame, the system classifies your hand gesture based on which fingers are extended and the relative positions of key landmarks. Detected gestures are mapped to drone flight commands, allowing intuitive control without a physical controller.

A pinch detection system runs in parallel, calculating the pixel distance between the thumb and index fingertip to detect fine motor gestures that finger-state classification alone can't reliably distinguish.

## Gesture Map
| Gesture | Drone Command |
|---|---|
| Open Palm | Hover / Stop |
| Fist | Land |
| Point Up | Move Forward |
| Peace | Rotate Counter-clockwise |
| Thumbs Up | Ascend |
| Palm Up | Descend |
| Pinch | Rotate Clockwise |

## Tech Stack
- Python
- MediaPipe
- OpenCV
- djitellopy

## Installation
```bash
pip install mediapipe opencv-python djitellopy
python handpilot.py
```
The hand landmark model downloads automatically on first run.

## Current Status
Gesture recognition and DJI Tello integration complete. Currently tuning gesture reliability and flight stability.

## Future Plans
- Autonomous waypoint navigation
- Object tracking and follow mode
- Multi-gesture sequence commands


# HandPilot

A real-time hand gesture drone control system that translates natural hand movements into flight commands using computer vision.

## Demo
*Demo video coming soon — Tello integration in progress*

## How It Works

HandPilot uses MediaPipe's hand landmark model to detect 21 points on your hand in real time via webcam. Each frame, the system classifies your hand gesture based on which fingers are extended and the relative positions of key landmarks. Detected gestures are mapped to drone flight commands, allowing intuitive control without a physical controller.

A pinch detection system runs in parallel, calculating the pixel distance between the thumb and index fingertip to detect fine motor gestures that finger-state classification alone can't reliably distinguish.

## Gesture Map

| Gesture | Drone Command |
|---|---|
| Open Palm | Hover / Stop |
| Fist | Land |
| Point Up | Takeoff |
| Peace | Move Forward |
| Thumbs Up | Ascend |
| Palm Up | Descend |
| Pinch | Rotate |

## Tech Stack

- Python
- MediaPipe
- OpenCV

## Installation

```bash
pip install mediapipe opencv-python
python handpilot.py
```

The hand landmark model downloads automatically on first run.

## Current Status

Gesture recognition system complete. DJI Tello integration in progress.

## Future Plans

- DJI Tello SDK integration
- Autonomous waypoint navigation
- Object tracking and follow mode
- Multi-gesture sequence commands


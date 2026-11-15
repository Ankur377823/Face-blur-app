# Real-Time Face Anonymizer

A small web app that blurs every face it sees in a live webcam feed. Faces are located with MediaPipe Face Detection and blurred with OpenCV; the interface is built with Gradio and also works from a phone camera.

## How it works

1. Gradio streams frames from the webcam to the app.
2. MediaPipe Face Detection returns a bounding box for each face in the frame.
3. Each box is clamped to the frame and blurred with a 50 x 50 OpenCV box blur.
4. The processed frame is shown next to the live feed.

## Tech stack

Python, OpenCV, MediaPipe, Gradio, NumPy

## Run locally

```bash
git clone https://github.com/Ankur377823/Face-blur-app.git
cd Face-blur-app
pip install -r requirements.txt gradio
python main.py
```

Gradio prints a local URL and a temporary public link. Open either one and allow camera access.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Face detection, blurring and the Gradio interface |
| `requirements.txt` | Python dependencies |

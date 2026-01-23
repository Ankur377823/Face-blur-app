import cv2
import mediapipe as mp
import gradio as gr
import numpy as np

# --- 1. Setup MediaPipe ---
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# --- 2. Processing Function ---
def process_img(img):
    if img is None:
        return None
    
    # Create a writable copy (Gradio images are read-only)
    img = img.copy()

    H, W, _ = img.shape
    
    # --- FIX IS HERE ---
    # Gradio already gives RGB. MediaPipe wants RGB.
    # We do NOT need to convert colors. We pass 'img' directly.
    out = face_detection.process(img)

    if out.detections:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box

            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w  = int(bbox.width * W)
            h  = int(bbox.height * H)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x1 + w)
            y2 = min(H, y1 + h)

            # Apply Blur
            # Check if region is valid to prevent crashes
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2] = cv2.blur(
                    img[y1:y2, x1:x2], (50, 50) # Increased blur to 50 for better visibility
                )

    return img

# --- 3. Run App ---
# I kept your structure but updated 'source' to 'sources' for compatibility
app = gr.Interface(
    fn=process_img,
    inputs=gr.Image(sources=["webcam"], streaming=True, type="numpy"),
    outputs=gr.Image(type="numpy"),
    live=True,
    title="Real-Time Face Anonymizer",
    description="Webcam & Phone camera face blurring using OpenCV + MediaPipe"
)

if __name__ == "__main__":
    app.launch(share=True)
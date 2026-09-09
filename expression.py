import os
import cv2
import time
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from obswebsocket import obsws, requests, exceptions

# ==========================================
# 1. Download MediaPipe Face Landmarker Model
# ==========================================
MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_FILENAME):
    print("Downloading MediaPipe Face Landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)

# ==========================================
# 2. Load Configuration & OBS Setup
# ==========================================
OBS_HOST = "localhost"
OBS_PORT = 4455
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "SBmb20P8LikJ6neZ")
SCENE_NAME = os.getenv("SCENE_NAME", "express")
SOURCE_NAME = os.getenv("SOURCE_NAME", "brain1")

MOUTH_OPEN_THRESHOLD = 0.40  # Trigger threshold
HOLD_DURATION = 3.0          # Seconds to keep image visible after closing mouth

ws = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD)
is_obs_connected = False

try:
    ws.connect()
    is_obs_connected = True
    print("Successfully connected to OBS Studio!")
except exceptions.ConnectionFailure:
    print("OBS Studio not connected. Running detector locally...")

def set_source_visibility(source_name, visible):
    global is_obs_connected
    if not is_obs_connected:
        try:
            ws.connect()
            is_obs_connected = True
        except Exception:
            return

    try:
        res = ws.call(requests.GetSceneItemId(sceneName=SCENE_NAME, sourceName=source_name))
        item_id = res.getSceneItemId()
        ws.call(requests.SetSceneItemEnabled(sceneName=SCENE_NAME, sceneItemId=item_id, sceneItemEnabled=bool(visible)))
    except Exception as e:
        print(f"OBS Toggle Error: {e}")
        is_obs_connected = False

# ==========================================
# 3. Initialize MediaPipe Tasks Vision Detector
# ==========================================
base_options = python.BaseOptions(model_asset_path=MODEL_FILENAME)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Camera Setup
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
current_meme = None
last_pose_time = 0  # Timestamp when pose was last active

while cap.isOpened():
    success, frame = cap.read()
    if not success or frame is None:
        continue

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)
    pose_active = False

    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        
        # Key mouth landmark indices
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]
        left_corner = landmarks[61]
        right_corner = landmarks[291]

        # Calculate mouth height and width
        vertical_dist = abs(top_lip.y - bottom_lip.y) * h
        horizontal_dist = abs(left_corner.x - right_corner.x) * w

        mar = vertical_dist / max(horizontal_dist, 1)

        if mar > MOUTH_OPEN_THRESHOLD:
            pose_active = True
            last_pose_time = time.time()  # Reset hold timer while holding pose

    # Keep image active if pose is live OR within the hold duration window
    if pose_active or (time.time() - last_pose_time < HOLD_DURATION):
        active_meme = SOURCE_NAME
    else:
        active_meme = None

    # Update OBS visibility only on state changes
    if active_meme != current_meme:
        if current_meme:
            set_source_visibility(current_meme, False)
        if active_meme:
            print(f"Mouth Open! Showing OBS Source: {active_meme}")
            set_source_visibility(active_meme, True)
        current_meme = active_meme

    cv2.imshow("Expression Detector (Press Q to quit)", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if is_obs_connected:
    ws.disconnect()
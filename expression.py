import os
import cv2
import time
import math
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from obswebsocket import obsws, requests, exceptions

# ==========================================
# 1. Download MediaPipe Models
# ==========================================
MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
}
for filename, url in MODELS.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

# ==========================================
# 2. Configuration
# ==========================================
OBS_HOST = os.getenv("OBS_HOST", "localhost")
OBS_PORT = int(os.getenv("OBS_PORT", "4455"))
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "")      # set via environment variable
SCENE_NAME = os.getenv("SCENE_NAME", "express")
SOURCE_MOUTH_OPEN = os.getenv("SOURCE_MOUTH_OPEN", "brain1")
SOURCE_FACE_COVER = os.getenv("SOURCE_FACE_COVER", "brain2")
OBS_RETRY_SECONDS = 5.0

MOUTH_OPEN_THRESHOLD = 0.40
HAND_NEAR_MOUTH_RATIO = 0.35

# --- Pose-based detector (PRIMARY: works even when hands overlap / face is hidden) ---
# All distances are relative to shoulder width (pixels).
POSE_WRISTS_NEAR_FACE = 0.60    # both wrists within this of the nose/mouth centre
POSE_WRISTS_TOGETHER = 0.50     # wrist-to-wrist distance
POSE_INDEX_ABOVE_WRIST = 0.03   # index points sit this much above the wrists
POSE_ELBOWS_BELOW_WRIST = 0.10  # forearms vertical: elbows lower than wrists

# --- Hand-landmark detector (SECONDARY) ---
REQUIRE_LEFT_AND_RIGHT = False  # pressed hands are often labelled the same side twice
VERTICAL_MIN_RATIO = 0.25
VERTICAL_MAX_ANGLE_DEG = 40
TIP_TO_NOSE_RATIO = 0.30
MCP_TOGETHER_RATIO = 0.50
WRIST_BELOW_CHIN_RATIO = 0.15
MIN_WRIST_SEPARATION_RATIO = 0.03

FACE_CACHE_SECONDS = 3.0
CONFIRM_FRAMES = 2
HOLD_DURATION = 1.5

# ==========================================
# 3. OBS helpers
# ==========================================
ws = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD)
is_obs_connected = False
last_obs_attempt = 0.0


def check_obs_setup():
    """Print what OBS actually has, so name mismatches are obvious."""
    try:
        scenes = [s["sceneName"] for s in ws.call(requests.GetSceneList()).datain["scenes"]]
        print(f"OBS scenes: {scenes}")
        if SCENE_NAME not in scenes:
            print(f"!! Scene '{SCENE_NAME}' NOT FOUND. Set SCENE_NAME to one of the scenes above.")
            return
        items = [i["sourceName"] for i in
                 ws.call(requests.GetSceneItemList(sceneName=SCENE_NAME)).datain["sceneItems"]]
        print(f"Sources in scene '{SCENE_NAME}': {items}")
        for s in (SOURCE_MOUTH_OPEN, SOURCE_FACE_COVER):
            if s not in items:
                print(f"!! Source '{s}' is NOT directly in scene '{SCENE_NAME}' "
                      f"(check spelling, or whether it sits inside a group).")
    except Exception as e:
        print(f"OBS setup check failed: {e}")


def try_connect_obs(verbose=False):
    global is_obs_connected, last_obs_attempt
    now = time.time()
    if now - last_obs_attempt < OBS_RETRY_SECONDS:
        return is_obs_connected
    last_obs_attempt = now
    try:
        ws.connect()
        is_obs_connected = True
        print("Successfully connected to OBS Studio!")
        check_obs_setup()
    except exceptions.ConnectionFailure:
        is_obs_connected = False
        if verbose:
            print("OBS Studio not connected (check port/password). Running detector locally...")
    except Exception as e:
        is_obs_connected = False
        if verbose:
            print(f"OBS connection error: {e}")
    return is_obs_connected


def set_source_visibility(source_name, visible):
    global is_obs_connected
    if not is_obs_connected and not try_connect_obs():
        return
    try:
        res = ws.call(requests.GetSceneItemId(sceneName=SCENE_NAME, sourceName=source_name))
        item_id = res.datain["sceneItemId"]
        ws.call(requests.SetSceneItemEnabled(
            sceneName=SCENE_NAME, sceneItemId=item_id, sceneItemEnabled=bool(visible)))
    except Exception as e:
        print(f"OBS Toggle Error on '{source_name}': {e}")
        is_obs_connected = False
        try:
            ws.disconnect()
        except Exception:
            pass


if try_connect_obs(verbose=True):
    set_source_visibility(SOURCE_MOUTH_OPEN, False)
    set_source_visibility(SOURCE_FACE_COVER, False)

# ==========================================
# 4. MediaPipe detectors (VIDEO mode = tracking)
# ==========================================
face_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
)
hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
)
pose_detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="pose_landmarker_full.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
)

# ==========================================
# 5. Geometry helpers (pixel space)
# ==========================================


def pt(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def dist(a, b):
    return float(np.linalg.norm(a - b))


# ---------- Pose detector ----------
def detect_pose_cover(pose_lms, w, h):
    """Both wrists pressed together in front of nose/mouth, index points up, forearms vertical.
    Pose landmarks: 0 nose, 9/10 mouth, 11/12 shoulders, 13/14 elbows, 15/16 wrists, 19/20 index.
    Returns (is_pose, any_wrist_near_face, debug_str)."""
    P = lambda i: pt(pose_lms[i], w, h)
    sw = dist(P(11), P(12))
    if sw < 20:
        return False, False, "pose: shoulders not visible"

    nose = P(0)
    mouth = (P(9) + P(10)) / 2
    face_c = (nose + mouth) / 2
    lw, rw = P(15), P(16)
    le, re = P(13), P(14)
    li, ri = P(19), P(20)

    d_l, d_r = dist(lw, face_c) / sw, dist(rw, face_c) / sw
    near = d_l < POSE_WRISTS_NEAR_FACE and d_r < POSE_WRISTS_NEAR_FACE
    any_near = min(d_l, d_r) < POSE_WRISTS_NEAR_FACE
    together = dist(lw, rw) / sw < POSE_WRISTS_TOGETHER
    idx_up = ((lw[1] + rw[1]) / 2 - (li[1] + ri[1]) / 2) / sw > POSE_INDEX_ABOVE_WRIST
    elbows_down = (le[1] - lw[1]) / sw > POSE_ELBOWS_BELOW_WRIST and \
                  (re[1] - rw[1]) / sw > POSE_ELBOWS_BELOW_WRIST
    wrists_low = lw[1] > nose[1] and rw[1] > nose[1]

    dbg = (f"pose near={near}({d_l:.2f}/{d_r:.2f}) tog={together}({dist(lw, rw) / sw:.2f}) "
           f"idxUp={idx_up} elbows={elbows_down} low={wrists_low}")
    return (near and together and idx_up and elbows_down and wrists_low), any_near, dbg


# ---------- Hand-landmark detector ----------
def face_metrics(face, w, h):
    chin = pt(face[152], w, h)
    forehead = pt(face[10], w, h)
    return {
        "nose": pt(face[6], w, h),
        "chin": chin,
        "mouth": (pt(face[13], w, h) + pt(face[14], w, h)) / 2,
        "face_h": max(abs(chin[1] - forehead[1]), 1.0),
        "face_w": max(abs(face[454].x - face[234].x) * w, 1.0),
        "t": time.time(),
    }


def get_hands(hand_result):
    hands = []
    if hand_result.hand_landmarks and hand_result.handedness:
        for lms, hd in zip(hand_result.hand_landmarks, hand_result.handedness):
            hands.append((lms, hd[0].category_name, hd[0].score))
    hands.sort(key=lambda x: -x[2])
    if REQUIRE_LEFT_AND_RIGHT:
        best = {}
        for item in hands:
            best.setdefault(item[1], item)
        return list(best.values())
    return hands[:2]


def hand_points_px(hand, w, h):
    return np.array([[lm.x * w, lm.y * h] for lm in hand], dtype=np.float32)


def is_hand_upright(hand, w, h, face_h):
    vec = pt(hand[8], w, h) - pt(hand[0], w, h)
    rise = -vec[1]
    if rise <= face_h * VERTICAL_MIN_RATIO:
        return False
    return math.degrees(math.atan2(abs(vec[0]), rise)) <= VERTICAL_MAX_ANGLE_DEG


def detect_two_hand_cover(hands, m, w, h):
    if len(hands) < 2:
        return False, f"hands={len(hands)} (need 2)"
    labels = sorted(x[1] for x in hands)
    if REQUIRE_LEFT_AND_RIGHT and labels != ["Left", "Right"]:
        return False, f"labels={labels}"
    a, b = hands[0][0], hands[1][0]
    face_h, face_w = m["face_h"], m["face_w"]
    if dist(pt(a[0], w, h), pt(b[0], w, h)) < face_h * MIN_WRIST_SEPARATION_RATIO:
        return False, "duplicate hand"
    upright = is_hand_upright(a, w, h, face_h) and is_hand_upright(b, w, h, face_h)
    together = abs(pt(a[5], w, h)[0] - pt(b[5], w, h)[0]) < face_w * MCP_TOGETHER_RATIO
    d_a = dist(pt(a[8], w, h), m["nose"]) / face_h
    d_b = dist(pt(b[8], w, h), m["nose"]) / face_h
    tips = d_a < TIP_TO_NOSE_RATIO and d_b < TIP_TO_NOSE_RATIO
    limit = m["chin"][1] - face_h * WRIST_BELOW_CHIN_RATIO
    wrists = pt(a[0], w, h)[1] >= limit and pt(b[0], w, h)[1] >= limit
    return (upright and together and tips and wrists), \
        f"hands {labels} up={upright} tog={together} tips={tips}({d_a:.2f}/{d_b:.2f}) wr={wrists}"


def any_hand_near_mouth(hand_result, m, w, h):
    if not hand_result.hand_landmarks:
        return False
    for hand in hand_result.hand_landmarks:
        if np.linalg.norm(hand_points_px(hand, w, h) - m["mouth"], axis=1).min() < m["face_h"] * HAND_NEAR_MOUTH_RATIO:
            return True
    return False


# ==========================================
# 6. Main loop
# ==========================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

start_time = time.time()
last_ts = -1
cached_face = None

current_meme = None
last_pose_time = 0.0
active_source_target = None
cover_streak = 0

print("Keys: Q quit | 1 = test mouth overlay in OBS | 2 = test face-cover overlay in OBS")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            time.sleep(0.01)
            continue

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        ts = int((time.time() - start_time) * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts

        face_result = face_detector.detect_for_video(mp_image, ts)
        hand_result = hand_detector.detect_for_video(mp_image, ts)
        pose_result = pose_detector.detect_for_video(mp_image, ts)

        # ---- Pose signal ----
        pose_cover, pose_wrist_near, pose_dbg = False, False, "pose: none"
        if pose_result.pose_landmarks:
            pose_lms = pose_result.pose_landmarks[0]
            pose_cover, pose_wrist_near, pose_dbg = detect_pose_cover(pose_lms, w, h)
            for i in (13, 14, 15, 16, 19, 20):
                p = pt(pose_lms[i], w, h)
                cv2.circle(frame, (int(p[0]), int(p[1])), 5, (255, 128, 0), -1)

        # ---- Face geometry (live or cached) ----
        face_live = bool(face_result.face_landmarks)
        face = face_result.face_landmarks[0] if face_live else None
        metrics, face_src = None, "none"
        if face_live:
            metrics, face_src = face_metrics(face, w, h), "live"
        elif cached_face and time.time() - cached_face["t"] < FACE_CACHE_SECONDS:
            metrics, face_src = cached_face, "cached"

        hands = get_hands(hand_result)
        hands_near_face = pose_wrist_near
        hand_cover, hand_dbg = False, "hands: no face geometry"
        mar = 0.0

        if metrics is not None:
            hands_near_face = hands_near_face or any_hand_near_mouth(hand_result, metrics, w, h)
            if face_live and not hands_near_face:
                cached_face = metrics
            hand_cover, hand_dbg = detect_two_hand_cover(hands, metrics, w, h)

        is_exact_face_cover = pose_cover or hand_cover

        is_mouth_open = False
        if face_live and not hands_near_face and not is_exact_face_cover:
            vertical = abs(face[13].y - face[14].y) * h
            horizontal = abs(face[61].x - face[291].x) * w
            mar = vertical / max(horizontal, 1.0)
            is_mouth_open = mar > MOUTH_OPEN_THRESHOLD

        cover_streak = cover_streak + 1 if is_exact_face_cover else 0
        cover_confirmed = cover_streak >= CONFIRM_FRAMES

        detected_pose = None
        if cover_confirmed:
            detected_pose = SOURCE_FACE_COVER
        elif is_mouth_open:
            detected_pose = SOURCE_MOUTH_OPEN

        if detected_pose:
            active_source_target = detected_pose
            last_pose_time = time.time()

        active_meme = active_source_target if time.time() - last_pose_time < HOLD_DURATION else None

        if active_meme != current_meme:
            if current_meme:
                set_source_visibility(current_meme, False)
            if active_meme:
                print(f"Triggered Pose! Showing OBS Source: {active_meme}")
                set_source_visibility(active_meme, True)
            current_meme = active_meme

        # ---- Debug overlay ----
        for lms, label, score in hands:
            for p in hand_points_px(lms, w, h):
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)
        status = "COVER!" if cover_confirmed else ("MOUTH" if is_mouth_open else "-")
        cv2.putText(frame, f"{status} face={face_src} hands={len(hands)} streak={cover_streak} mar={mar:.2f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, pose_dbg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        cv2.putText(frame, hand_dbg, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

        cv2.imshow("Expression Detector (Press Q to quit)", frame)
        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            active_source_target, last_pose_time = SOURCE_MOUTH_OPEN, time.time()
        elif key == ord("2"):
            active_source_target, last_pose_time = SOURCE_FACE_COVER, time.time()
finally:
    if current_meme:
        set_source_visibility(current_meme, False)
    cap.release()
    cv2.destroyAllWindows()
    if is_obs_connected:
        try:
            ws.disconnect()
        except Exception:
            pass
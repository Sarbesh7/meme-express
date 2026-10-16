import os
import cv2
import time
import math
import urllib.request
from contextlib import ExitStack
from pathlib import Path
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
BASE_DIR = Path(__file__).resolve().parent


def download_models():
    for filename, url in MODELS.items():
        path = BASE_DIR / filename
        if not path.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, path)

# ==========================================
# 2. Configuration
# ==========================================
OBS_HOST = os.getenv("OBS_HOST", "localhost")
OBS_PORT = int(os.getenv("OBS_PORT", "4455"))
OBS_PASSWORD = os.getenv("OBS_PASSWORD", "")      # set via environment variable
SCENE_NAME = os.getenv("SCENE_NAME", "express")
SOURCE_MOUTH_OPEN = os.getenv("SOURCE_MOUTH_OPEN", "brain1")
SOURCE_FACE_COVER = os.getenv("SOURCE_FACE_COVER", "brain2")
SOURCE_TALKING_WALL = os.getenv("SOURCE_TALKING_WALL", "talking_wall") # NEW MEME SOURCE
SOURCE_CONFUSED_TILT = os.getenv("SOURCE_CONFUSED_TILT", "meme_confused")
SOURCE_WINK = os.getenv("SOURCE_WINK", "meme_rizz")
SOURCE_SHUSH = os.getenv("SOURCE_SHUSH", "meme_shush")
ALL_SOURCES = (
    SOURCE_MOUTH_OPEN, SOURCE_FACE_COVER, SOURCE_TALKING_WALL,
    SOURCE_CONFUSED_TILT, SOURCE_WINK, SOURCE_SHUSH,
)
OBS_RETRY_SECONDS = 5.0

MOUTH_OPEN_THRESHOLD = 0.40
HAND_NEAR_MOUTH_RATIO = 0.35

# --- Pose-based detector thresholds ---
POSE_WRISTS_NEAR_FACE = 0.60    # both wrists within this of the nose/mouth centre
POSE_WRISTS_TOGETHER = 0.50     # wrist-to-wrist distance
POSE_INDEX_ABOVE_WRIST = 0.03   # index points sit this much above the wrists
POSE_ELBOWS_BELOW_WRIST = 0.10  # forearms vertical: elbows lower than wrists

# --- Open Hands Talking Pose Thresholds ---
POSE_OPEN_HANDS_MIN_DIST = 1.0   # Wrists must be separated by at least 1.0x shoulder width
POSE_WRIST_HEIGHT_MIN = 0.20     # Wrists must be at or above chest level relative to shoulders

# --- Hand-landmark detector ---
REQUIRE_LEFT_AND_RIGHT = False
VERTICAL_MIN_RATIO = 0.25
VERTICAL_MAX_ANGLE_DEG = 40
TIP_TO_NOSE_RATIO = 0.30
MCP_TOGETHER_RATIO = 0.50
WRIST_BELOW_CHIN_RATIO = 0.15
MIN_WRIST_SEPARATION_RATIO = 0.03

FACE_CACHE_SECONDS = 3.0
CONFIRM_FRAMES = 2
HOLD_DURATION = 1.5

# New gestures use live landmarks and timed confirmation, independent of FPS.
# Tune using EAR (eye aspect ratio), roll, and shush distance in the preview.
TILT_MIN_DEG = 18.0
TILT_MAX_DEG = 45.0
TILT_CONFIRM_SECONDS = 0.35
EYE_CLOSED_EAR = 0.17
EYE_OPEN_EAR = 0.23
WINK_MIN_EAR_DIFFERENCE = 0.10
WINK_MAX_ROLL_DEG = 15.0
WINK_CONFIRM_SECONDS = 0.22
FACE_MIN_EYE_WIDTH_PX = 8.0
FACE_MIN_EYE_WIDTH_RATIO = 0.60  # reject strong side profiles
FACE_MAX_NOSE_OFFSET_RATIO = 0.30
SHUSH_MAX_ROLL_DEG = 25.0
SHUSH_MOUTH_DISTANCE_RATIO = 0.10
SHUSH_INDEX_MIN_ANGLE_DEG = 155.0
SHUSH_FOLDED_MAX_ANGLE_DEG = 135.0
SHUSH_VERTICAL_MAX_ANGLE_DEG = 25.0
SHUSH_CONFIRM_SECONDS = 0.30
MAX_CONFIRM_GAP_SECONDS = 0.50

# ==========================================
# 3. OBS helpers
# ==========================================
ws = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD, timeout=2)
is_obs_connected = False
last_obs_attempt = float("-inf")
scene_item_ids = {}
source_visibility = {}


def check_obs_setup():
    """Print what OBS actually has, so name mismatches are obvious."""
    scene_item_ids.clear()
    source_visibility.clear()
    try:
        scenes = [s["sceneName"] for s in ws.call(requests.GetSceneList()).datain["scenes"]]
        print(f"OBS scenes: {scenes}")
        if SCENE_NAME not in scenes:
            print(f"!! Scene '{SCENE_NAME}' NOT FOUND. Set SCENE_NAME to one of the scenes above.")
            return
        items = ws.call(requests.GetSceneItemList(sceneName=SCENE_NAME)).datain["sceneItems"]
        for item in items:
            scene_item_ids.setdefault(item["sourceName"], []).append(item["sceneItemId"])
        print(f"Sources in scene '{SCENE_NAME}': {list(scene_item_ids)}")
        for s in ALL_SOURCES:
            if s not in scene_item_ids:
                print(f"!! Source '{s}' is NOT directly in scene '{SCENE_NAME}' "
                      f"(check spelling, or whether it sits inside a group).")
    except Exception as e:
        print(f"OBS setup check failed: {e}")


def try_connect_obs(verbose=False):
    global is_obs_connected, last_obs_attempt
    if is_obs_connected:
        return True
    now = time.monotonic()
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
    if not is_obs_connected or source_name not in scene_item_ids:
        return False
    try:
        for item_id in scene_item_ids[source_name]:
            res = ws.call(requests.SetSceneItemEnabled(
                sceneName=SCENE_NAME, sceneItemId=item_id, sceneItemEnabled=bool(visible)))
            if not res.status:
                raise RuntimeError("OBS rejected the visibility change; check scene/source names")
        source_visibility[source_name] = bool(visible)
        return True
    except Exception as e:
        print(f"OBS Toggle Error on '{source_name}': {e}")
        is_obs_connected = False
        try:
            ws.disconnect()
        except Exception:
            pass
        return False


def sync_obs_sources(active_source):
    """Reapply all six states after reconnect; hide old memes before showing a new one."""
    if not try_connect_obs():
        return
    for source in ALL_SOURCES:
        if source != active_source and source_visibility.get(source) is not False:
            if source in scene_item_ids and not set_source_visibility(source, False):
                return
    if active_source and source_visibility.get(active_source) is not True:
        set_source_visibility(active_source, True)

# ==========================================
# 4. MediaPipe detectors (VIDEO mode = tracking)
# ==========================================
def create_detectors(stack):
    """ExitStack also closes earlier detectors if a later one fails to initialize."""
    face_detector = stack.enter_context(vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(BASE_DIR / "face_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    ))
    hand_detector = stack.enter_context(vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(BASE_DIR / "hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    ))
    pose_detector = stack.enter_context(vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(BASE_DIR / "pose_landmarker_full.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
    ))
    return face_detector, hand_detector, pose_detector

# ==========================================
# 5. Geometry helpers (pixel space)
# ==========================================


def pt(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def dist(a, b):
    return float(np.linalg.norm(a - b))


class TimedConfirmation:
    """Require consecutive positive samples spanning a minimum duration."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.since = None
        self.last_sample = None

    def update(self, detected, now):
        if self.last_sample is not None and now - self.last_sample > MAX_CONFIRM_GAP_SECONDS:
            self.since = None
        self.last_sample = now
        if not detected:
            self.since = None
            return False
        if self.since is None:
            self.since = now
        return now - self.since >= self.seconds


def eye_aspect_ratio(face, indices, w, h):
    """Two lid gaps divided by twice the eye width; invariant to in-plane roll."""
    a, b, c, d, e, f = [pt(face[i], w, h) for i in indices]
    return (dist(b, f) + dist(c, e)) / max(2.0 * dist(a, d), 1.0)


def expression_geometry(face, w, h):
    """Pixel-space geometry avoids aspect-ratio distortion in non-square video."""
    if face is None or len(face) < 468:
        return None
    right_width = dist(pt(face[33], w, h), pt(face[133], w, h))
    left_width = dist(pt(face[362], w, h), pt(face[263], w, h))
    if min(left_width, right_width) < FACE_MIN_EYE_WIDTH_PX:
        return None
    right_eye = (pt(face[33], w, h) + pt(face[133], w, h)) / 2
    left_eye = (pt(face[362], w, h) + pt(face[263], w, h)) / 2
    axis = left_eye - right_eye
    eye_span = dist(left_eye, right_eye)
    if eye_span < 20.0:
        return None
    roll = (math.degrees(math.atan2(axis[1], axis[0])) + 90) % 180 - 90
    midpoint = (left_eye + right_eye) / 2
    nose_offset = abs(float(np.dot(pt(face[1], w, h) - midpoint, axis / eye_span))) / eye_span
    frontal = (min(left_width, right_width) / max(left_width, right_width)
               >= FACE_MIN_EYE_WIDTH_RATIO and nose_offset <= FACE_MAX_NOSE_OFFSET_RATIO)
    return {
        "roll": roll,
        "left_ear": eye_aspect_ratio(face, (362, 385, 387, 263, 373, 380), w, h),
        "right_ear": eye_aspect_ratio(face, (33, 160, 158, 133, 153, 144), w, h),
        "mar": dist(pt(face[13], w, h), pt(face[14], w, h))
               / max(dist(pt(face[61], w, h), pt(face[291], w, h)), 1.0),
        "frontal": frontal,
        "mouth": (pt(face[13], w, h) + pt(face[14], w, h)) / 2,
        "nose": pt(face[1], w, h),
        "face_h": dist(pt(face[10], w, h), pt(face[152], w, h)),
    }


def detect_confused_tilt(g):
    """Math-is-not-mathing: tilt either way, with both eyes open and mouth relaxed."""
    return bool(g and g["frontal"] and TILT_MIN_DEG <= abs(g["roll"]) <= TILT_MAX_DEG
                and min(g["left_ear"], g["right_ear"]) >= EYE_OPEN_EAR
                and g["mar"] < MOUTH_OPEN_THRESHOLD)


def detect_wink(g):
    """Rizz: one eye closed and the other open. A two-eye blink is rejected."""
    if not g or not g["frontal"] or abs(g["roll"]) > WINK_MAX_ROLL_DEG:
        return False
    closed, opened = sorted((g["left_ear"], g["right_ear"]))
    return (closed <= EYE_CLOSED_EAR and opened >= EYE_OPEN_EAR
            and opened - closed >= WINK_MIN_EAR_DIFFERENCE
            and g["mar"] < MOUTH_OPEN_THRESHOLD)


def joint_angle(a, b, c):
    """Angle at b in degrees; a straight finger is close to 180 degrees."""
    ba, bc = a - b, c - b
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator < 1e-6:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))))


def point_segment_distance(point, a, b):
    segment = b - a
    length_sq = float(np.dot(segment, segment))
    if length_sq < 1e-6:
        return dist(point, a)
    t = float(np.clip(np.dot(point - a, segment) / length_sq, 0.0, 1.0))
    return dist(point, a + t * segment)


def detect_shush(hands, g, w, h):
    """Let-him-cook: one upright index at the lips, other three fingers curled.

    This is a 2D alignment heuristic, not a measurement of physical lip contact.
    Require a live face; cached geometry is reserved for the old cover detector.
    """
    if not g or not g["frontal"] or abs(g["roll"]) > SHUSH_MAX_ROLL_DEG:
        return False, "shush: need live frontal face"
    if len(hands) != 1 or g["face_h"] < 30:
        return False, "shush: need one hand"
    points = hand_points_px(hands[0][0], w, h)
    wrist, base, pip, dip, tip = points[[0, 5, 6, 7, 8]]
    straight = (joint_angle(base, pip, dip) >= SHUSH_INDEX_MIN_ANGLE_DEG
                and joint_angle(pip, dip, tip) >= SHUSH_INDEX_MIN_ANGLE_DEG)
    folded = all(
        joint_angle(points[mcp], points[joint], points[end]) < SHUSH_FOLDED_MAX_ANGLE_DEG
        and dist(points[end], wrist) < 1.15 * dist(points[joint], wrist)
        for mcp, joint, end in ((9, 10, 12), (13, 14, 16), (17, 18, 20))
    )
    rise = float(base[1] - tip[1])
    upright = (rise > 0.18 * g["face_h"] and
               math.degrees(math.atan2(abs(float(tip[0] - base[0])), max(rise, 1e-6)))
               <= SHUSH_VERTICAL_MAX_ANGLE_DEG)
    gap = point_segment_distance(g["mouth"], pip, tip) / g["face_h"]
    at_lips = (gap <= SHUSH_MOUTH_DISTANCE_RATIO
               and g["nose"][1] - 0.08 * g["face_h"] <= tip[1]
               <= g["mouth"][1] + 0.03 * g["face_h"]
               and base[1] > g["mouth"][1] and wrist[1] > base[1])
    active = straight and folded and upright and at_lips
    return active, f"shush={active} gap={gap:.2f} straight={straight} curl={folded} up={upright}"


def choose_meme(cover, talking, mouth, tilt, wink, shush):
    """Specific hand gestures win; only one source is selected per frame."""
    for detected, source in (
        (cover, SOURCE_FACE_COVER), (shush, SOURCE_SHUSH),
        (talking, SOURCE_TALKING_WALL), (wink, SOURCE_WINK),
        (tilt, SOURCE_CONFUSED_TILT), (mouth, SOURCE_MOUTH_OPEN),
    ):
        if detected:
            return source
    return None


# ---------- Pose detectors ----------
def detect_pose_cover(pose_lms, w, h):
    """Both wrists pressed together in front of nose/mouth, index points up, forearms vertical."""
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


def detect_open_hands_talking(pose_lms, w, h):
    """Detects wrists spaced wide open apart at or above chest level (explaining/talking to wall meme)."""
    P = lambda i: pt(pose_lms[i], w, h)
    sw = dist(P(11), P(12))
    if sw < 20:
        return False, "talking: shoulders not visible"

    lw, rw = P(15), P(16)
    ls, rs = P(11), P(12)
    shoulder_y = (ls[1] + rs[1]) / 2.0

    wrist_dist_ratio = dist(lw, rw) / sw
    hands_wide = wrist_dist_ratio > POSE_OPEN_HANDS_MIN_DIST
    
    # Check if wrists are raised up around chest level or higher (y is inverted in screen space)
    hands_up = (lw[1] < shoulder_y + sw * POSE_WRIST_HEIGHT_MIN) and \
               (rw[1] < shoulder_y + sw * POSE_WRIST_HEIGHT_MIN)

    is_talking_wall = hands_wide and hands_up
    dbg = f"talking_wall={is_talking_wall} (dist_ratio={wrist_dist_ratio:.2f}, hands_up={hands_up})"
    return is_talking_wall, dbg


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
        "t": time.monotonic(),
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
def run_detection(cap, face_detector, hand_detector, pose_detector):
    start_time = time.monotonic()
    last_ts = -1
    cached_face = None
    current_meme = None
    last_pose_time = float("-inf")
    active_source_target = None
    cover_streak = 0
    read_failures = 0
    manual_source, manual_until = None, 0.0
    tilt_timer = TimedConfirmation(TILT_CONFIRM_SECONDS)
    wink_timer = TimedConfirmation(WINK_CONFIRM_SECONDS)
    shush_timer = TimedConfirmation(SHUSH_CONFIRM_SECONDS)
    test_keys = {ord(str(i)): source for i, source in enumerate(ALL_SOURCES, start=1)}

    print("Keys: Q quit | 1 mouth | 2 cover | 3 talking wall | 4 confused | 5 wink | 6 shush")
    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            read_failures += 1
            if read_failures >= 30:
                raise RuntimeError("Camera stopped returning frames. Check camera access/index.")
            time.sleep(0.01)
            continue
        read_failures = 0
        now = time.monotonic()

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        ts = int((now - start_time) * 1000)
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts

        face_result = face_detector.detect_for_video(mp_image, ts)
        hand_result = hand_detector.detect_for_video(mp_image, ts)
        pose_result = pose_detector.detect_for_video(mp_image, ts)

        # ---- Pose signals ----
        pose_cover, pose_wrist_near, pose_dbg = False, False, "pose: none"
        talking_wall_active, talking_wall_dbg = False, "talking: none"
        if pose_result.pose_landmarks:
            pose_lms = pose_result.pose_landmarks[0]
            pose_cover, pose_wrist_near, pose_dbg = detect_pose_cover(pose_lms, w, h)
            talking_wall_active, talking_wall_dbg = detect_open_hands_talking(pose_lms, w, h)
            for i in (11, 12, 13, 14, 15, 16, 19, 20):
                p = pt(pose_lms[i], w, h)
                cv2.circle(frame, (int(p[0]), int(p[1])), 5, (255, 128, 0), -1)

        # ---- Face geometry (live or cached) ----
        face_live = bool(face_result.face_landmarks)
        face = face_result.face_landmarks[0] if face_live else None
        metrics, face_src = None, "none"
        if face_live:
            metrics, face_src = face_metrics(face, w, h), "live"
        elif cached_face and now - cached_face["t"] < FACE_CACHE_SECONDS:
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

        # New expressions NEVER use cached face landmarks. Hands near the mouth
        # suppress eye/tilt guesses, but are necessary for the shush gesture.
        geometry = expression_geometry(face, w, h)
        face_unobstructed = not hands_near_face and not is_exact_face_cover
        tilt_confirmed = tilt_timer.update(
            face_unobstructed and detect_confused_tilt(geometry), now)
        wink_confirmed = wink_timer.update(
            face_unobstructed and detect_wink(geometry), now)
        shush_raw, shush_dbg = detect_shush(hands, geometry, w, h)
        shush_confirmed = shush_timer.update(shush_raw and not is_exact_face_cover, now)

        # ---- Determine which pose/meme is active ----
        detected_pose = choose_meme(
            cover_confirmed, talking_wall_active, is_mouth_open,
            tilt_confirmed, wink_confirmed, shush_confirmed)

        if manual_source and now < manual_until:
            active_meme = manual_source
        else:
            manual_source = None
            if detected_pose:
                active_source_target = detected_pose
                last_pose_time = now
            active_meme = active_source_target if now - last_pose_time < HOLD_DURATION else None

        if active_meme != current_meme:
            if active_meme:
                print(f"Triggered Pose! Showing OBS Source: {active_meme}")
            current_meme = active_meme
        sync_obs_sources(active_meme)

        # ---- Debug overlay ----
        for lms, label, score in hands:
            for p in hand_points_px(lms, w, h):
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)
        face_dbg = "new gestures: no live face geometry"
        if geometry:
            face_dbg = (f"roll={geometry['roll']:.1f} EAR L/R={geometry['left_ear']:.2f}/"
                        f"{geometry['right_ear']:.2f} front={geometry['frontal']} "
                        f"tilt={tilt_confirmed} wink={wink_confirmed}")
        lines = (
            f"OBS={'ON' if is_obs_connected else 'OFF'} active={active_meme or '-'} "
            f"face={face_src} hands={len(hands)} mar={mar:.2f}",
            pose_dbg, talking_wall_dbg, face_dbg, shush_dbg,
            "Q quit | 1 mouth 2 cover 3 wall 4 tilt 5 wink 6 shush",
        )
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

        cv2.imshow("Expression Detector (Press Q to quit)", frame)
        key = cv2.waitKey(5) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in test_keys:
            manual_source = test_keys[key]
            manual_until = time.monotonic() + HOLD_DURATION
            active_source_target = None
            # A manual test stays visible even when another gesture is detected.
            sync_obs_sources(manual_source)


def main():
    if len(set(ALL_SOURCES)) != len(ALL_SOURCES):
        raise ValueError("All six SOURCE_* settings must use distinct OBS source names.")
    download_models()
    cap = None
    try:
        with ExitStack() as stack:
            detectors = create_detectors(stack)
            camera_index = int(os.getenv("CAMERA_INDEX", "0"))
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise RuntimeError("Cannot open camera. Check CAMERA_INDEX and other camera apps.")
            try_connect_obs(verbose=True)
            sync_obs_sources(None)
            run_detection(cap, *detectors)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if is_obs_connected:
            sync_obs_sources(None)
        try:
            ws.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()

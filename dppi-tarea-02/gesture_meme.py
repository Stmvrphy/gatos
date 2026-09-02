"""
Webcam gesture -> album detector (desktop version).

Opens two windows, side by side:
  - "Camera": webcam feed with hand landmarks and debug HUD (yaw/pitch)
  - "Meme": album cover corresponding to the detected gesture

Gestures:
  1. crossFingers        -> memes/BLIZZARD_OZZY.webp
  2. twoHandsCoverFace   -> memes/BOOTLEG_DYLAN.jpg
  3. heroesBowie         -> memes/HEROES_BOWIE.jpg
  4. queenShoulders      -> memes/QUEEN2_QUEEN.jpeg
  5. madonnaTrueBlue     -> memes/MADONNA_TRUE_BLUE.jpeg
  6. theBendsRadiohead   -> memes/THEBENDS_RADIOHEAD.jpeg

Press q or ESC to quit.
"""

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MEMES = ROOT / "memes"

GESTURE_MEMES = {
    "crossFingers": ["BLIZZARD_OZZY.webp"],
    "twoHandsCoverFace": ["BOOTLEG_DYLAN.jpg"],
    "heroesBowie": ["HEROES_BOWIE.jpg"],
    "queenShoulders": ["QUEEN2_QUEEN.jpeg"],
    "madonnaTrueBlue": ["MADONNA_TRUE_BLUE.jpeg"],
    "theBendsRadiohead": ["THEBENDS_RADIOHEAD.jpeg"],
    "default": ["rick_cover.jpg"],
}

VIDEO_GESTURES = set()

STABLE_FRAMES_REQUIRED = 5
DEFAULT_FALLBACK_MS = 600
FACE_STALE_MS = 1200

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ---- geometry helpers (ported from the JS version) -----------------------
def p3(lm):
    return np.array([lm.x, lm.y, lm.z])


def dist(a, b):
    return float(np.linalg.norm(a - b))


def angle_deg(v1, v2):
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-9 or m2 < 1e-9:
        return 180.0
    cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def finger_extended(pts, mcp, pip, tip):
    v1 = pts[pip] - pts[mcp]
    v2 = pts[tip] - pts[pip]
    return angle_deg(v1, v2) < 45


def head_pose_angles(matrix):
    """Extract yaw, pitch, roll in degrees from MediaPipe's facial
    transformation matrix."""
    r = np.asarray(matrix)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        return 0.0, 0.0, 0.0
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def classify_hand(landmarks):
    pts = [p3(lm) for lm in landmarks]
    hand_scale = dist(pts[0], pts[9]) or 1e-6

    index_up = finger_extended(pts, 5, 6, 8)
    middle_up = finger_extended(pts, 9, 10, 12)
    ring_up = finger_extended(pts, 13, 14, 16)
    pinky_up = finger_extended(pts, 17, 18, 20)

    thumb_pinky_spread = dist(pts[4], pts[17]) / hand_scale
    thumb_out = thumb_pinky_spread > 1.05

    curled_count = sum(1 for v in (index_up, middle_up, ring_up, pinky_up) if not v)

    return {
        "indexUp": index_up,
        "middleUp": middle_up,
        "ringUp": ring_up,
        "pinkyUp": pinky_up,
        "thumbOut": thumb_out,
        "curledCount": curled_count,
        "handScale": hand_scale,
        "indexTip": pts[8],
        "indexPip": pts[6],
        "indexMcp": pts[5],
        "wrist": pts[0],
        "palmCenter": pts[9],
    }


def is_pointing(h):
    return h["indexUp"] and not h["middleUp"] and not h["ringUp"] and not h["pinkyUp"]


def is_cross_gesture(h0, h1):
    if not (is_pointing(h0) and is_pointing(h1)):
        return False
    avg_scale = (h0["handScale"] + h1["handScale"]) / 2
    v0 = h0["indexTip"] - h0["indexMcp"]
    v1 = h1["indexTip"] - h1["indexMcp"]
    ang = angle_deg(v0, v1)
    is_perpendicular = 50.0 <= ang <= 130.0
    v0_horiz = abs(v0[0]) > abs(v0[1]) * 0.8
    v0_vert = abs(v0[1]) > abs(v0[0]) * 0.8
    v1_horiz = abs(v1[0]) > abs(v1[1]) * 0.8
    v1_vert = abs(v1[1]) > abs(v1[0]) * 0.8
    one_h_one_v = (v0_horiz and v1_vert) or (v0_vert and v1_horiz)

    mid0 = (h0["indexMcp"] + h0["indexTip"]) / 2
    mid1 = (h1["indexMcp"] + h1["indexTip"]) / 2
    d_mid = dist(mid0, mid1) / avg_scale
    d_tip0_mid1 = dist(h0["indexTip"], mid1) / avg_scale
    d_tip1_mid0 = dist(h1["indexTip"], mid0) / avg_scale
    is_close = min(d_mid, d_tip0_mid1, d_tip1_mid0) < 1.5

    return is_perpendicular and one_h_one_v and is_close


def is_bowie_heroes_gesture(hands, last_face):
    if len(hands) != 2:
        return False
    if any(h["curledCount"] >= 3 for h in hands):
        return False

    h_raised = min(hands, key=lambda h: h["palmCenter"][1])
    h_lower = max(hands, key=lambda h: h["palmCenter"][1])

    y_diff = h_lower["palmCenter"][1] - h_raised["palmCenter"][1]
    if y_diff < 0.08:
        return False

    if last_face is not None:
        mouth_center, face_width, _, _, _, _ = last_face
        lower_under_face = h_lower["palmCenter"][1] > mouth_center[1] - face_width * 0.3
        lower_near_x = abs(h_lower["palmCenter"][0] - mouth_center[0]) / face_width < 1.8
        lower_dist = dist(h_lower["palmCenter"], mouth_center) / face_width < 3.0

        raised_at_head_height = (
            mouth_center[1] - face_width * 2.0 < h_raised["palmCenter"][1] < mouth_center[1] + face_width * 0.7
        )
        raised_at_side = abs(h_raised["palmCenter"][0] - mouth_center[0]) / face_width > 0.5
        raised_dist = dist(h_raised["palmCenter"], mouth_center) / face_width < 3.0

        return (
            lower_under_face
            and lower_near_x
            and lower_dist
            and raised_at_head_height
            and raised_at_side
            and raised_dist
        )
    x_diff = abs(h_raised["palmCenter"][0] - h_lower["palmCenter"][0])
    return y_diff > 0.10 and x_diff > 0.10


def is_queen_shoulders_gesture(hands, last_face):
    if len(hands) != 2:
        return False
    if any(h["curledCount"] >= 3 for h in hands):
        return False

    y_diff = abs(hands[0]["palmCenter"][1] - hands[1]["palmCenter"][1])
    if y_diff > 0.18:
        return False

    h_left = min(hands, key=lambda h: h["palmCenter"][0])
    h_right = max(hands, key=lambda h: h["palmCenter"][0])
    x_spread = h_right["palmCenter"][0] - h_left["palmCenter"][0]

    if not (0.12 <= x_spread <= 0.48):
        return False

    if last_face is not None:
        mouth_center, face_width, _, _, _, _ = last_face
        both_below_mouth = all(
            h["palmCenter"][1] > mouth_center[1] + face_width * 0.15 for h in hands
        )
        both_above_waist = all(
            h["palmCenter"][1] < mouth_center[1] + face_width * 3.2 for h in hands
        )
        left_on_left = h_left["palmCenter"][0] < mouth_center[0] + face_width * 0.2
        right_on_right = h_right["palmCenter"][0] > mouth_center[0] - face_width * 0.2

        return both_below_mouth and both_above_waist and left_on_left and right_on_right
    avg_y = (hands[0]["palmCenter"][1] + hands[1]["palmCenter"][1]) / 2
    return avg_y > 0.35


class GestureState:
    def __init__(self):
        self.last_face = None  # (mouth_center, face_width, mouth_open, yaw_deg, pitch_deg, t)
        self.face_seen_this_frame = False
        self.last_yaw_debug = 0.0
        self.last_pitch_debug = 0.0

    def update_face(self, face_result):
        now = time.time() * 1000
        saw_face = bool(face_result.face_landmarks)

        if saw_face:
            f = face_result.face_landmarks[0]
            upper_lip, lower_lip = p3(f[13]), p3(f[14])
            right_cheek, left_cheek = p3(f[234]), p3(f[454])
            mouth_center = (upper_lip + lower_lip) / 2
            face_width = dist(right_cheek, left_cheek)
            mouth_open = dist(upper_lip, lower_lip) / face_width

            yaw_deg, pitch_deg = 0.0, 0.0
            if face_result.facial_transformation_matrixes:
                yaw_deg, pitch_deg, _ = head_pose_angles(face_result.facial_transformation_matrixes[0])

            self.last_face = (mouth_center, face_width, mouth_open, yaw_deg, pitch_deg, now)
            self.last_yaw_debug = yaw_deg
            self.last_pitch_debug = pitch_deg
        self.face_seen_this_frame = saw_face

    def decide(self, hand_result):
        now = time.time() * 1000
        face_is_fresh = self.last_face is not None and now - self.last_face[5] < FACE_STALE_MS

        if not hand_result.hand_landmarks:
            if face_is_fresh:
                _, _, _, yaw_deg, pitch_deg, _ = self.last_face
                if abs(pitch_deg) > 10.0:
                    if abs(yaw_deg) > 18.0:
                        return "madonnaTrueBlue"
                    return "theBendsRadiohead"
            return "default"

        hands = [classify_hand(lm) for lm in hand_result.hand_landmarks]

        if len(hands) == 2:
            if is_cross_gesture(hands[0], hands[1]):
                return "crossFingers"

            if is_bowie_heroes_gesture(hands, self.last_face if face_is_fresh else None):
                return "heroesBowie"

            if is_queen_shoulders_gesture(hands, self.last_face if face_is_fresh else None):
                return "queenShoulders"

            if face_is_fresh:
                mouth_center, face_width, _, _, _, _ = self.last_face
                cover_threshold = 1.4 if not self.face_seen_this_frame else 1.2
                both_covering = all(
                    dist(h["palmCenter"], mouth_center) / face_width < cover_threshold
                    for h in hands
                )
                if both_covering:
                    return "twoHandsCoverFace"

        if face_is_fresh:
            _, _, _, yaw_deg, pitch_deg, _ = self.last_face
            if abs(pitch_deg) > 10.0:
                if abs(yaw_deg) > 18.0:
                    return "madonnaTrueBlue"
                return "theBendsRadiohead"

        return "default"


def load_memes():
    cache = {}
    for gesture, files in GESTURE_MEMES.items():
        imgs = []
        for name in files:
            img = cv2.imread(str(MEMES / name))
            if img is None:
                raise FileNotFoundError(f"missing meme file: {MEMES / name}")
            imgs.append(img)
        cache[gesture] = imgs
    return cache


def draw_debug_hud(frame, state, gesture):
    lines = [
        f"Gesto: {gesture}",
        f"Yaw (giro): {state.last_yaw_debug:+.1f} deg",
        f"Pitch (atras): {state.last_pitch_debug:+.1f} deg",
    ]
    for i, line in enumerate(lines):
        y = 26 + i * 24
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 1, cv2.LINE_AA)


def draw_landmarks(frame, hand_result):
    h, w = frame.shape[:2]
    for hand in hand_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (60, 140, 255), -1)


def fit_to_height(img, height):
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def main():
    hand_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
        )
    )
    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
    )

    memes = load_memes()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")

    cv2.namedWindow("Camera")
    cv2.namedWindow("Meme")
    cv2.moveWindow("Camera", 40, 80)
    cv2.moveWindow("Meme", 720, 80)

    state = GestureState()
    current_gesture = "default"
    candidate_gesture = "default"
    candidate_streak = 0
    last_non_default_at = time.time() * 1000
    current_meme = random.choice(memes["default"]) if "default" in memes else None

    start_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror, like a selfie cam

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
            face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
            state.update_face(face_result)

            gesture = state.decide(hand_result)

            now = time.time() * 1000
            if gesture == candidate_gesture:
                candidate_streak += 1
            else:
                candidate_gesture = gesture
                candidate_streak = 1

            if candidate_streak >= STABLE_FRAMES_REQUIRED and gesture != current_gesture:
                current_gesture = gesture
                if current_gesture in memes:
                    current_meme = random.choice(memes[current_gesture])
                else:
                    current_meme = None

            if gesture != "default":
                last_non_default_at = now
            elif now - last_non_default_at > DEFAULT_FALLBACK_MS and current_gesture != "default":
                current_gesture = "default"
                current_meme = random.choice(memes["default"]) if "default" in memes else None

            draw_landmarks(frame, hand_result)
            draw_debug_hud(frame, state, current_gesture)

            if current_meme is not None:
                meme_view = fit_to_height(current_meme, frame.shape[0])
            else:
                h, w = frame.shape[:2]
                meme_view = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(
                    meme_view,
                    "Esperando gesto...",
                    (w // 5, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (100, 100, 100),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Camera", frame)
            cv2.imshow("Meme", meme_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()


if __name__ == "__main__":
    main()

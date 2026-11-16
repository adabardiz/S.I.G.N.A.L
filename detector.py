import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import numpy as np
import mediapipe as mp
import time
import math
import threading
import os
import joblib
import re
from collections import deque
from functools import lru_cache
from types import SimpleNamespace
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

try:
    from train_model import extract_hand_features
except ImportError:
    def extract_hand_features(landmarks):
        if len(landmarks) == 0:
            return [0.0] * 63
        wrist = landmarks[0]
        norm = landmarks - wrist
        scale = np.linalg.norm(landmarks[9] - wrist)
        if scale > 0:
            norm = norm / scale
        return norm.flatten().tolist()


INK         = (30, 26, 26)        
INK_SOFT    = (48, 44, 42)        
HAIR        = (84, 86, 92)        
IVORY       = (226, 235, 240)     
TAUPE       = (146, 154, 164)    
GOLD        = (118, 170, 205)     
GOLD_LIGHT  = (150, 198, 228)     
GOLD_DEEP   = (78, 124, 160)      
SAGE        = (150, 176, 128)     # live/active
ROSE        = (122, 122, 200)     # stop/recording
ROSE_LIGHT  = (150, 150, 222)

FONT_SANS  = cv2.FONT_HERSHEY_SIMPLEX   
FONT_SERIF = cv2.FONT_HERSHEY_TRIPLEX   

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

KEY_FACE_INDICES = [
    1,
    33, 133, 159, 145,
    362, 263, 386, 374,
    70, 63, 105, 66,
    300, 293, 334, 296,
    61, 291, 0, 17, 13, 14,
    78, 308, 82, 312
]

KEY_ARM_INDICES = [11, 12, 13, 14, 15, 16]
ARM_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

EMOTION_WORDS = {
    "HAPPY", "SAD", "ANGRY", "MAD", "SCARED", "AFRAID", "SURPRISED",
    "EXCITED", "TIRED", "ANNOYED", "DISGUSTED", "FEAR", "FEARFUL",
    "JOY", "JOYFUL", "FURIOUS", "NERVOUS", "UPSET", "PROUD", "JEALOUS",
    "WORRIED", "LOVE", "HATE", "DEPRESSED", "CALM", "LONELY"
}

PUNCT_OPTIONS = {
    "PERIOD":      [("Neutral", ".", "neutral"), ("Sad", ".", "sad"), ("Sarcastic", ".", "sarcastic")],
    "EXCLAMATION": [("Happy", "!", "happy"), ("Angry", "!", "angry")],
    "QUESTION":    [("Surprised", "?", "surprised"), ("Sarcastic", "?", "sarcastic")],
}
PUNCT_TITLES = {
    None:          ("SELECT PUNCTUATION", "How should the sentence end?"),
    "PERIOD":      ("PERIOD", "Choose the tone of voice"),
    "EXCLAMATION": ("EXCLAMATION", "Choose the tone of voice"),
    "QUESTION":    ("QUESTION", "Choose the tone of voice"),
}

def extract_face_features(face_landmarks):
    if not face_landmarks:
        return [0.0] * (len(KEY_FACE_INDICES) * 3)
    nose_tip = np.array([face_landmarks[1].x, face_landmarks[1].y, face_landmarks[1].z])
    face_feats = []
    for idx in KEY_FACE_INDICES:
        lm = face_landmarks[idx]
        face_feats.extend([lm.x - nose_tip[0], lm.y - nose_tip[1], lm.z - nose_tip[2]])
    return face_feats

def extract_arm_features(pose_landmarks):
    if not pose_landmarks:
        return [0.0] * (len(KEY_ARM_INDICES) * 3)
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    shoulder_mid = np.array([
        (left_shoulder.x + right_shoulder.x) / 2.0,
        (left_shoulder.y + right_shoulder.y) / 2.0,
        (left_shoulder.z + right_shoulder.z) / 2.0
    ])
    arm_feats = []
    for idx in KEY_ARM_INDICES:
        lm = pose_landmarks[idx]
        arm_feats.extend([lm.x - shoulder_mid[0], lm.y - shoulder_mid[1], lm.z - shoulder_mid[2]])
    return arm_feats

def calculate_facial_intensity(face_landmarks):
    if not face_landmarks:
        return 1.0, "NEUTRAL"

    def get_pt(idx):
        lm = face_landmarks[idx]
        return np.array([lm.x, lm.y])

    left_eye_outer = get_pt(33)
    right_eye_outer = get_pt(263)
    face_scale = np.linalg.norm(left_eye_outer - right_eye_outer)
    if face_scale == 0:
        return 1.0, "NEUTRAL"

    upper_lip = get_pt(13)
    lower_lip = get_pt(14)
    mouth_height = np.linalg.norm(upper_lip - lower_lip) / face_scale

    left_eyebrow = get_pt(105)
    left_eye = get_pt(159)
    eyebrow_dist = np.linalg.norm(left_eyebrow - left_eye) / face_scale

    left_eye_top = get_pt(159)
    left_eye_bottom = get_pt(145)
    eye_openness = np.linalg.norm(left_eye_top - left_eye_bottom) / face_scale

    neutral_mouth = 0.08
    neutral_eyebrow = 0.22
    neutral_eye = 0.12

    mouth_delta = max(0.0, mouth_height - neutral_mouth)
    eyebrow_delta = abs(eyebrow_dist - neutral_eyebrow)
    eye_delta = max(0.0, eye_openness - neutral_eye)

    raw_score = (mouth_delta * 2.5) + (eyebrow_delta * 3.0) + (eye_delta * 2.0)
    intensity_multiplier = 1.0 + min(max(raw_score, 0.0), 1.5)

    if intensity_multiplier < 1.25:
        level = "NEUTRAL"
    elif intensity_multiplier < 1.60:
        level = "MODERATE"
    else:
        level = "EXTREME"

    return round(intensity_multiplier, 2), level

def is_valid_hand_shape(landmarks):
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    middle_mcp = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
    palm_size = np.linalg.norm(wrist - middle_mcp)
    return 0.01 < palm_size < 0.65

def is_open_hand(hand_landmarks):
    wrist = hand_landmarks[0]
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]
    open_fingers = 0
    for tip, mcp in zip(tips, mcps):
        dist_tip = np.linalg.norm([hand_landmarks[tip].x - wrist.x, hand_landmarks[tip].y - wrist.y])
        dist_mcp = np.linalg.norm([hand_landmarks[mcp].x - wrist.x, hand_landmarks[mcp].y - wrist.y])
        if dist_tip > dist_mcp:
            open_fingers += 1
    return open_fingers >= 4

def is_two_open_hands(hand1, hand2):
    return is_open_hand(hand1) and is_open_hand(hand2)

def aggregate_sequence(sequence_matrix):
    seq = np.array(sequence_matrix, dtype=np.float32)
    mean_f = np.mean(seq, axis=0)
    std_f = np.std(seq, axis=0)
    delta_f = seq[-1] - seq[0]
    max_f = np.max(seq, axis=0)
    min_f = np.min(seq, axis=0)
    return np.hstack([mean_f, std_f, delta_f, max_f, min_f])

def transform_tense(sentence, tense_target):
    clean_sent = sentence.strip()
    if not clean_sent:
        return ""
    if tense_target == "NOW":
        return f"{clean_sent[:-1] if clean_sent[-1] in '.!?' else clean_sent} right now."
    elif tense_target == "PAST":
        return f"Previously, {clean_sent.lower()}"
    elif tense_target == "FUTURE":
        return f"Will {clean_sent.lower()}"
    return clean_sent


mouse_click_pos = None
mouse_pos = None

def on_mouse_event(event, x, y, flags, param):
    global mouse_click_pos, mouse_pos
    mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_pos = (x, y)

def hit(mx, my, rect):
    return rect[0] <= mx <= rect[2] and rect[1] <= my <= rect[3]

@lru_cache(maxsize=4096)
def _char_w(ch, font, scale, thick):
    return cv2.getTextSize(ch, font, scale, thick)[0][0]

def text_w(text, font=FONT_SANS, scale=0.5, thick=1, spacing=0):
    if spacing == 0:
        return cv2.getTextSize(text, font, scale, thick)[0][0]
    total = sum(_char_w(c, font, scale, thick) + spacing for c in text)
    return max(0, total - spacing)

def put_text(img, text, org, font=FONT_SANS, scale=0.5, color=IVORY,
             thick=1, spacing=0, anchor="left"):
    """Anti-aliased text with optional letter-spacing and alignment."""
    tw = text_w(text, font, scale, thick, spacing)
    x, y = org
    if anchor == "center":
        x -= tw // 2
    elif anchor == "right":
        x -= tw
    if spacing == 0:
        cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)
    else:
        for ch in text:
            cv2.putText(img, ch, (x, y), font, scale, color, thick, cv2.LINE_AA)
            x += _char_w(ch, font, scale, thick) + spacing

def label(img, text, org, color=TAUPE, anchor="left", scale=0.34):
    """Small tracked uppercase label."""
    put_text(img, text.upper(), org, FONT_SANS, scale, color, 1, spacing=2, anchor=anchor)

def fit_tail(text, font, scale, thick, max_w):
    if text_w(text, font, scale, thick) <= max_w:
        return text
    while text and text_w("..." + text, font, scale, thick) > max_w:
        text = text[1:]
    return "..." + text

def wrap_lines(text, font, scale, thick, max_w):
    lines, cur = [], ""
    for word in text.split(" "):
        test = f"{cur} {word}" if cur else word
        if cv2.getTextSize(test, font, scale, thick)[0][0] <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines

def draw_wrapped(img, text, pos, max_w, font, scale, color, thick,
                 line_h, max_lines, tail=True):
    lines = wrap_lines(text, font, scale, thick, max_w)
    if len(lines) > max_lines:
        if tail:
            lines = lines[-max_lines:]
        else:
            lines = lines[:max_lines]
            last = lines[-1]
            while last and text_w(last + "...", font, scale, thick) > max_w:
                last = last[:-1]
            lines[-1] = last + "..."
    x, y = pos
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (x, y + i * line_h), font, scale, color, thick, cv2.LINE_AA)

def fill_alpha(img, rect, color, alpha):
    x1, y1, x2, y2 = rect
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)

def frosted(img, strength=0.5):
    """Cheap frosted-glass backdrop: downscale, blur, upscale, tint."""
    H, W = img.shape[:2]
    small = cv2.resize(img, (max(1, W // 8), max(1, H // 8)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    blurred = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    tint = np.full_like(blurred, INK)
    img[:] = cv2.addWeighted(blurred, 1.0 - strength, tint, strength, 0)

def draw_panel(img, rect, alpha=0.78, accent=True):
    fill_alpha(img, rect, INK, alpha)
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), HAIR, 1, cv2.LINE_AA)
    if accent:
        cv2.line(img, (x1, y1), (x1 + 34, y1), GOLD, 2, cv2.LINE_AA)

def draw_hairline(img, p1, p2, color=HAIR):
    cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)

def draw_dot(img, center, color, radius=5):
    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)

def pulse_color(color, base=INK, speed=4.0):
    p = 0.55 + 0.45 * math.sin(time.time() * speed)
    return tuple(int(c * p + b * (1 - p)) for c, b in zip(color, base))

def draw_progress(img, rect, ratio, color=GOLD):
    x1, y1, x2, y2 = rect
    fill_alpha(img, rect, INK_SOFT, 0.9)
    fx = x1 + int((x2 - x1) * min(max(ratio, 0.0), 1.0))
    if fx > x1:
        cv2.rectangle(img, (x1, y1), (fx, y2), color, -1)

def draw_button(img, rect, text, style="ghost", scale=0.42):
    x1, y1, x2, y2 = rect
    hover = mouse_pos is not None and hit(mouse_pos[0], mouse_pos[1], rect)

    if style == "solid":
        fill, fa, border, tc = (GOLD_LIGHT if hover else GOLD), 0.97, GOLD, INK
    elif style == "alert":
        fill, fa, border, tc = (ROSE_LIGHT if hover else ROSE), 0.97, ROSE, INK
    elif style == "active":
        fill, fa, border, tc = INK, 0.85, GOLD, GOLD
    else:  #ghost
        fill, fa = INK_SOFT, (0.90 if hover else 0.62)
        border, tc = (GOLD if hover else HAIR), (GOLD_LIGHT if hover else IVORY)

    fill_alpha(img, rect, fill, fa)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)

    text = text.upper()
    s = scale
    while text_w(text, FONT_SANS, s, 1, 2) > (x2 - x1 - 16) and s > 0.28:
        s -= 0.02
    th = cv2.getTextSize("H", FONT_SANS, s, 1)[0][1]
    put_text(img, text, ((x1 + x2) // 2, (y1 + y2) // 2 + th // 2),
             FONT_SANS, s, tc, 1, spacing=2, anchor="center")

def draw_modal(img, box, title, subtitle=None):
    draw_panel(img, box, alpha=0.90, accent=False)
    x1, y1, x2, y2 = box
    mx = (x1 + x2) // 2
    cv2.line(img, (mx - 18, y1), (mx + 18, y1), GOLD, 2, cv2.LINE_AA)
    put_text(img, title, (mx, y1 + 42), FONT_SANS, 0.52, GOLD, 1, spacing=3, anchor="center")
    if subtitle:
        put_text(img, subtitle, (mx, y1 + 72), FONT_SERIF, 0.46, TAUPE, 1, anchor="center")

def modal_box(cx, cy, bw, bh):
    return (cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2)

def row_rects(box, y_off, n, bw=200, bh=48, gap=18):
    mx = (box[0] + box[2]) // 2
    total = n * bw + (n - 1) * gap
    x0 = mx - total // 2
    y = box[1] + y_off
    return [(x0 + i * (bw + gap), y, x0 + i * (bw + gap) + bw, y + bh) for i in range(n)]

def compute_layout(w, h):
    cx, cy = w // 2, h // 2

    widths, gap, x = [150, 130, 130, 110, 110], 10, w - 24
    top = []
    for bw in reversed(widths):
        top.append((x - bw, 18, x, 54))
        x -= bw + gap
    top.reverse()

    syn_box = modal_box(cx, cy, 700, 210)
    pun_box = modal_box(cx, cy, 740, 220)
    tense_box = modal_box(cx, cy, 540, 270)
    fin_box = modal_box(cx, h - 180, 820, 260)
    voice_box = modal_box(cx, cy, 460, 300)

    return SimpleNamespace(
        cx=cx, cy=cy,
        top=top,
        voice_btn=(24, h - 100, 224, h - 64),
        controls_btn=(24, h - 56, 224, h - 20),
        syn_box=syn_box,
        syn_row=lambda n: row_rects(syn_box, 108, n, bw=200),
        pun_box=pun_box,
        pun_row=lambda n: row_rects(pun_box, 112, n, bw=210),
        pun_back=(pun_box[0] + 24, pun_box[3] - 52, pun_box[0] + 124, pun_box[3] - 20),
        tense_box=tense_box,
        tense_btns=row_rects(tense_box, 104, 2, bw=220) + row_rects(tense_box, 104 + 48 + 16, 2, bw=220),
        fin_box=fin_box,
        fin_btns=row_rects(fin_box, 186, 3, bw=220),
        voice_box=voice_box,
        voice_btns=[(cx - 190, voice_box[1] + 92 + i * 58, cx + 190, voice_box[1] + 92 + i * 58 + 46)
                    for i in range(3)],
    )

def draw_top_bar(img, w):
    fill_alpha(img, (0, 0, w, 72), INK, 0.82)
    draw_hairline(img, (0, 72), (w, 72), GOLD_DEEP)
    put_text(img, "S.I.G.N.A.L", (28, 38), FONT_SERIF, 0.85, IVORY, 1)
    put_text(img, "SIGN INTERPRETATION & GESTURE NETWORK ALGORITHM", (28, 58),
             FONT_SANS, 0.28, TAUPE, 1, spacing=2)

def draw_stat_card(img, rect, title, value, value_color=IVORY, value_scale=0.62, ratio=None):
    draw_panel(img, rect, alpha=0.74)
    x1, y1, x2, y2 = rect
    label(img, title, (x1 + 16, y1 + 22))
    put_text(img, value, (x1 + 16, y1 + 50), FONT_SERIF, value_scale, value_color, 1)
    if ratio is not None:
        draw_progress(img, (x1 + 16, y2 - 8, x2 - 16, y2 - 6), ratio, GOLD)

def draw_hands_and_arms(frame, valid_hands, pose_lms, w, h):
    if pose_lms:
        for s_idx, e_idx in ARM_CONNECTIONS:
            s_lm, e_lm = pose_lms[s_idx], pose_lms[e_idx]
            if s_lm.visibility > 0.5 and e_lm.visibility > 0.5:
                cv2.line(frame, (int(s_lm.x * w), int(s_lm.y * h)),
                         (int(e_lm.x * w), int(e_lm.y * h)), TAUPE, 2, cv2.LINE_AA)
        for idx in KEY_ARM_INDICES:
            lm = pose_lms[idx]
            if lm.visibility > 0.5:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, GOLD_DEEP, -1, cv2.LINE_AA)
    for hand in valid_hands:
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame,
                     (int(hand[a].x * w), int(hand[a].y * h)),
                     (int(hand[b].x * w), int(hand[b].y * h)),
                     IVORY, 2, cv2.LINE_AA)
        for i, lm in enumerate(hand):
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pt, 5 if i in (4, 8, 12, 16, 20) else 3, GOLD, -1, cv2.LINE_AA)


class TextToSpeechEngine:
    def __init__(self):
        self.voices = ["Alex", "Samantha", "Fred"]
        self.voice_idx = 0
        self.current_voice_label = self.voices[self.voice_idx]

    def speak(self, text, tone="neutral", intensity=1.0):
        intensity_tag = f" [{intensity}x]" if intensity > 1.2 else ""
        print(f"[TTS ({self.current_voice_label} | {tone}){intensity_tag}]: {text}")

        def play_audio():
            clean_text = text.replace('"', '').replace("'", "")
            os.system(f'say -v {self.current_voice_label} "{clean_text}"')

        threading.Thread(target=play_audio, daemon=True).start()

    def select_voice(self, idx):
        if 0 <= idx < len(self.voices):
            self.voice_idx = idx
            self.current_voice_label = self.voices[self.voice_idx]

    def toggle_voice(self):
        self.voice_idx = (self.voice_idx + 1) % len(self.voices)
        self.current_voice_label = self.voices[self.voice_idx]

class BackgroundAIThread(threading.Thread):
    def __init__(self, letter_model=None):
        super().__init__(daemon=True)
        self.letter_model = letter_model
        self.predicted_letter = "-"
        self.latest_features = None
        self.has_new_data = False
        self.lock = threading.Lock()
        self.running = True

    def update_data(self, features, hand_x=None):
        with self.lock:
            self.latest_features = features
            self.has_new_data = True

    def run(self):
        while self.running:
            feats = None
            has_data = False
            with self.lock:
                if self.has_new_data:
                    feats = self.latest_features
                    has_data = True
                    self.has_new_data = False

            if has_data:
                if feats is not None and self.letter_model is not None:
                    try:
                        pred = self.letter_model.predict([feats])[0]
                        with self.lock:
                            self.predicted_letter = str(pred).upper()
                    except Exception:
                        with self.lock:
                            self.predicted_letter = "-"
                else:
                    with self.lock:
                        self.predicted_letter = "-"
            time.sleep(0.01)

    def get_prediction(self):
        with self.lock:
            return self.predicted_letter

class SpeechToTextThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.listening = False
        self.transcript = ""
        self.status = "IDLE"
        self.lock = threading.Lock()

        if HAS_SR:
            self.recognizer = sr.Recognizer()
        else:
            self.recognizer = None

    def start_listening(self):
        with self.lock:
            self.listening = True
            self.status = "LISTENING..."

    def stop_listening(self):
        with self.lock:
            self.listening = False
            self.status = "PAUSED"

    def clear_transcript(self):
        with self.lock:
            self.transcript = ""

    def delete_last_word(self):
        with self.lock:
            words = self.transcript.strip().split()
            if words:
                self.transcript = " ".join(words[:-1])
            else:
                self.transcript = ""

    def delete_last_char(self):
        with self.lock:
            if self.transcript:
                self.transcript = self.transcript[:-1]

    def append_text(self, char):
        with self.lock:
            self.transcript += char

    def get_data(self):
        with self.lock:
            return self.transcript, self.status, self.listening

    def run(self):
        if not HAS_SR:
            with self.lock:
                self.status = "Install 'speech_recognition' & 'pyaudio'"
            while self.running:
                time.sleep(0.5)
            return

        while self.running:
            is_active = False
            with self.lock:
                is_active = self.listening

            if is_active:
                try:
                    with sr.Microphone() as source:
                        with self.lock:
                            self.status = "LISTENING TO HEARING PERSON..."
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                        audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=6)

                    with self.lock:
                        self.status = "PROCESSING SPEECH..."
                    text = self.recognizer.recognize_google(audio)

                    with self.lock:
                        if self.transcript:
                            self.transcript += " " + text
                        else:
                            self.transcript = text
                        self.status = "LISTENING TO HEARING PERSON..."
                except sr.WaitTimeoutError:
                    pass
                except sr.UnknownValueError:
                    pass
                except Exception:
                    with self.lock:
                        self.status = "MIC ACTIVE (Awaiting Speech)"
            else:
                time.sleep(0.1)


def main():
    global mouse_click_pos

    asl_word_model = None
    if os.path.exists("asl_word_model.pkl"):
        asl_word_model = joblib.load("asl_word_model.pkl")

    letter_model = None
    if os.path.exists("asl_model.pkl"):
        letter_model = joblib.load("asl_model.pkl")
    elif os.path.exists("asl_letter_model.pkl"):
        letter_model = joblib.load("asl_letter_model.pkl")
    elif os.path.exists("model.pkl"):
        letter_model = joblib.load("model.pkl")

    bg_ai = BackgroundAIThread(letter_model=letter_model)
    bg_ai.start()

    stt_thread = SpeechToTextThread()
    stt_thread.start()

    tts = TextToSpeechEngine()

    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.60,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.60
    )
    detector = vision.HandLandmarker.create_from_options(options)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50
    )

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "S.I.G.N.A.L  |  ASL Gesture Translation"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_event)

    app_view = "SIGN"
    keyboard_editing_active = False

    current_mode = "SPELL"
    word_sequence_buffer = deque(maxlen=30)
    last_word_pred_time = 0.0
    WORD_COOLDOWN_DURATION = 3.0

    is_recording = False
    selecting_punctuation = False
    punct_sub = None
    selecting_tense = False
    is_converting_tense = False
    temp_sentence = ""

    selecting_synonym = False
    synonym_options = []

    show_controls_popup = False
    show_voices_popup = False

    selected_tone = "neutral"

    current_word = ""
    finished_word = ""
    word_history = []

    space_start_time = None
    space_triggered = False

    letter_hold_start_time = None
    current_holding_letter = None
    prev_wrist_pos = None

    HOLD_LETTER_DURATION = 2.0
    ACTION_GESTURE_DURATION = 1.0

    start_time_ms = int(time.time() * 1000)
    last_timestamp_ms = 0
    frame_counter = 0

    cached_face_lms = None
    cached_face_feats = [0.0] * (len(KEY_FACE_INDICES) * 3)
    cached_pose_lms = None
    cached_arm_feats = [0.0] * (len(KEY_ARM_INDICES) * 3)
    cached_intensity_mult = 1.0
    cached_intensity_label = "NEUTRAL"

    smoothed_landmarks = None
    alpha = 0.70

    TARGET_FPS = 30
    TARGET_FRAME_TIME = 1.0 / TARGET_FPS

    while True:
        loop_start_time = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_counter += 1
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        L = compute_layout(w, h)
        cx, cy = L.cx, L.cy

        predicted_letter = bg_ai.get_prediction()

        if mouse_click_pos is not None:
            mx, my = mouse_click_pos
            mouse_click_pos = None

            if hit(mx, my, L.voice_btn):
                show_voices_popup = not show_voices_popup
                show_controls_popup = False

            elif hit(mx, my, L.controls_btn):
                show_controls_popup = not show_controls_popup
                show_voices_popup = False

            elif show_controls_popup:
                show_controls_popup = False

            elif show_voices_popup:
                clicked = next((i for i, r in enumerate(L.voice_btns) if hit(mx, my, r)), None)
                if clicked is not None:
                    tts.select_voice(clicked)
                    show_voices_popup = False
                elif not hit(mx, my, L.voice_box):
                    show_voices_popup = False

            elif hit(mx, my, L.top[0]):
                if app_view == "SIGN":
                    app_view = "TRANSCRIPTION"
                    stt_thread.start_listening()
                else:
                    app_view = "SIGN"
                    stt_thread.stop_listening()
                    keyboard_editing_active = False

            elif hit(mx, my, L.top[1]):
                if app_view == "SIGN":
                    is_recording = not is_recording
                    if is_recording:
                        current_word = ""
                        finished_word = ""
                        selecting_punctuation = False
                        punct_sub = None
                        selecting_tense = False
                        selecting_synonym = False
                        word_sequence_buffer.clear()
                        last_word_pred_time = 0.0
                    else:
                        if current_word.strip():
                            selecting_punctuation = True
                            punct_sub = None
                        else:
                            finished_word = ""
                else:
                    keyboard_editing_active = False
                    live_text, live_status, is_listening = stt_thread.get_data()
                    if is_listening:
                        stt_thread.stop_listening()
                    else:
                        stt_thread.start_listening()

            elif hit(mx, my, L.top[2]):
                if app_view == "SIGN":
                    current_mode = "WORD" if current_mode == "SPELL" else "SPELL"
                    selecting_synonym = False
                    word_sequence_buffer.clear()
                    last_word_pred_time = 0.0
                else:
                    stt_thread.delete_last_word()

            elif hit(mx, my, L.top[3]):
                if app_view == "SIGN":
                    current_word = current_word[:-1]
                else:
                    keyboard_editing_active = not keyboard_editing_active
                    if keyboard_editing_active:
                        stt_thread.stop_listening()

            elif hit(mx, my, L.top[4]):
                if app_view == "SIGN":
                    current_word = ""
                    finished_word = ""
                    selecting_punctuation = False
                    punct_sub = None
                    selecting_tense = False
                    selecting_synonym = False
                else:
                    stt_thread.clear_transcript()

            elif selecting_synonym and app_view == "SIGN":
                n_opts = min(len(synonym_options), 3)
                for i, r in enumerate(L.syn_row(max(n_opts, 1))):
                    if i < n_opts and hit(mx, my, r):
                        chosen = synonym_options[i]
                        current_word += f"{chosen} "
                        tts.speak(chosen, tone=selected_tone)
                        selecting_synonym = False
                        synonym_options = []
                        break

            elif selecting_punctuation and app_view == "SIGN":
                if punct_sub is None:
                    for sub, r in zip(("PERIOD", "EXCLAMATION", "QUESTION"), L.pun_row(3)):
                        if hit(mx, my, r):
                            punct_sub = sub
                            break
                elif hit(mx, my, L.pun_back):
                    punct_sub = None
                else:
                    opts = PUNCT_OPTIONS[punct_sub]
                    for (_, mark, tone), r in zip(opts, L.pun_row(len(opts))):
                        if hit(mx, my, r):
                            temp_sentence = current_word.strip() + mark
                            selected_tone = tone
                            selecting_punctuation, punct_sub, selecting_tense = False, None, True
                            break

            elif selecting_tense and not is_converting_tense and app_view == "SIGN":
                chosen_tense = None
                for t, r in zip(("NOW", "PAST", "FUTURE", "ORIGINAL"), L.tense_btns):
                    if hit(mx, my, r):
                        chosen_tense = t
                        break

                if chosen_tense:
                    selecting_tense = False
                    is_converting_tense = True

                    def process_tense_and_speak(t_tense, raw_sent, tone):
                        nonlocal finished_word, is_converting_tense
                        final_sent = transform_tense(raw_sent, t_tense)
                        finished_word = final_sent
                        if final_sent:
                            word_history.append(final_sent)
                            tts.speak(final_sent, tone)
                        is_converting_tense = False

                    threading.Thread(
                        target=process_tense_and_speak,
                        args=(chosen_tense, temp_sentence, selected_tone),
                        daemon=True
                    ).start()

            elif not is_recording and finished_word and app_view == "SIGN":
                b_speak, b_tone, b_dismiss = L.fin_btns
                if hit(mx, my, b_speak):
                    tts.speak(finished_word, selected_tone)
                elif hit(mx, my, b_tone):
                    if word_history and word_history[-1] == finished_word:
                        word_history.pop()
                    finished_word = ""
                    selecting_punctuation, punct_sub = True, None
                elif hit(mx, my, b_dismiss):
                    finished_word = ""
                    current_word = ""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_detected = False
        space_gesture_detected = False
        open_hand = False
        features = None
        valid_hands = []
        pose_lms = None

        if app_view == "SIGN":
            small_rgb = cv2.resize(rgb_frame, (640, 360))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb)

            frame_timestamp_ms = int(time.time() * 1000) - start_time_ms
            if frame_timestamp_ms <= last_timestamp_ms:
                frame_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = frame_timestamp_ms

            detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

            if frame_counter % 3 == 0:
                face_results = face_mesh.process(small_rgb)
                cached_face_lms = face_results.multi_face_landmarks[0].landmark if face_results.multi_face_landmarks else None
                cached_face_feats = extract_face_features(cached_face_lms)

                pose_results = pose.process(small_rgb)
                cached_pose_lms = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None
                cached_arm_feats = extract_arm_features(cached_pose_lms)
                cached_intensity_mult, cached_intensity_label = calculate_facial_intensity(cached_face_lms)

            face_lms = cached_face_lms
            face_feats = cached_face_feats
            pose_lms = cached_pose_lms
            arm_feats = cached_arm_feats
            intensity_mult, intensity_label = cached_intensity_mult, cached_intensity_label

            if detection_result.hand_landmarks:
                valid_hands = [hd for hd in detection_result.hand_landmarks if is_valid_hand_shape(hd)]
                if valid_hands:
                    hand_detected = True
                    primary_hand = valid_hands[0]
                    curr_wrist = np.array([primary_hand[0].x, primary_hand[0].y])
                    hand_movement = np.linalg.norm(curr_wrist - prev_wrist_pos) if prev_wrist_pos is not None else 0.0
                    prev_wrist_pos = curr_wrist

                    if len(valid_hands) >= 2:
                        space_gesture_detected = is_two_open_hands(valid_hands[0], valid_hands[1])

                    if not space_gesture_detected:
                        open_hand = is_open_hand(primary_hand)

                    if is_recording and space_gesture_detected:
                        if space_start_time is None:
                            space_start_time = time.time()
                        elif (time.time() - space_start_time >= ACTION_GESTURE_DURATION) and not space_triggered:
                            current_word += " "
                            space_triggered = True
                    else:
                        space_start_time, space_triggered = None, False

                    raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in primary_hand])
                    if smoothed_landmarks is None:
                        smoothed_landmarks = raw_pts
                    else:
                        smoothed_landmarks = alpha * raw_pts + (1 - alpha) * smoothed_landmarks

                    features = extract_hand_features(smoothed_landmarks)
                    bg_ai.update_data(features, primary_hand[0].x)
                else:
                    smoothed_landmarks = None
                    bg_ai.update_data(None, None)
                    space_start_time, letter_hold_start_time, current_holding_letter, prev_wrist_pos = None, None, None, None
            else:
                smoothed_landmarks = None
                bg_ai.update_data(None, None)
                space_start_time, letter_hold_start_time, current_holding_letter, prev_wrist_pos = None, None, None, None

            draw_hands_and_arms(frame, valid_hands, pose_lms, w, h)
        else:
            bg_ai.update_data(None, None)
            intensity_mult, intensity_label = 1.0, "NEUTRAL"

        if is_recording and not open_hand and not space_gesture_detected and hand_detected and not selecting_synonym and app_view == "SIGN":
            if current_mode == "SPELL":
                if predicted_letter == current_holding_letter:
                    if letter_hold_start_time and (time.time() - letter_hold_start_time >= HOLD_LETTER_DURATION):
                        current_word += predicted_letter
                        tts.speak(predicted_letter, tone="neutral", intensity=1.0)
                        letter_hold_start_time = time.time()
                else:
                    current_holding_letter = predicted_letter
                    letter_hold_start_time = time.time()

            elif current_mode == "WORD" and asl_word_model is not None and features is not None:
                if time.time() - last_word_pred_time >= WORD_COOLDOWN_DURATION:
                    combined_frame_feats = list(features) + list(face_feats) + list(arm_feats)
                    word_sequence_buffer.append(combined_frame_feats)

                    if len(word_sequence_buffer) == 30:
                        aggregated_vec = aggregate_sequence(word_sequence_buffer)
                        try:
                            predicted_word = asl_word_model.predict([aggregated_vec])[0]
                            if predicted_word:
                                raw_word_str = str(predicted_word).strip()
                                word_str = raw_word_str

                                has_emotion_tag = False
                                if word_str.lower().endswith("_e") or "(e)" in word_str.lower():
                                    has_emotion_tag = True
                                    if word_str.lower().endswith("_e"):
                                        word_str = word_str[:-2].strip()
                                    word_str = word_str.replace("(e)", "").replace("(E)", "").strip()

                                word_str = re.sub(r'(_v\d+|_v_\d+|_\d+)$', '', word_str, flags=re.IGNORECASE).strip()

                                if word_str.lower().endswith("_e") or "(e)" in word_str.lower():
                                    has_emotion_tag = True
                                    if word_str.lower().endswith("_e"):
                                        word_str = word_str[:-2].strip()
                                    word_str = word_str.replace("(e)", "").replace("(E)", "").strip()

                                is_emotion_word = has_emotion_tag or (word_str.upper() in EMOTION_WORDS)

                                if is_emotion_word and intensity_label == "EXTREME":
                                    if not word_str.upper().startswith("VERY "):
                                        word_str = f"VERY {word_str}"

                                if "/" in word_str:
                                    synonym_options = [wd.strip() for wd in word_str.split("/") if wd.strip()]
                                    selecting_synonym = True
                                else:
                                    current_word += f"{word_str} "
                                    word_tone = word_str.lower().replace("very ", "") if is_emotion_word else selected_tone
                                    tts.speak(word_str, tone=word_tone, intensity=intensity_mult if is_emotion_word else 1.0)
                        except Exception as e:
                            print("Word prediction error:", e)

                        word_sequence_buffer.clear()
                        last_word_pred_time = time.time()
                else:
                    word_sequence_buffer.clear()
        else:
            letter_hold_start_time, current_holding_letter = None, None


        modal_active = (app_view == "SIGN" and (selecting_synonym or selecting_punctuation or selecting_tense)) \
            or show_controls_popup or show_voices_popup
        if app_view == "TRANSCRIPTION":
            frosted(frame, 0.66)
        elif modal_active:
            frosted(frame, 0.52)

        draw_top_bar(frame, w)

        draw_button(frame, L.voice_btn, f"Voice   {tts.current_voice_label}", "ghost", 0.38)
        draw_button(frame, L.controls_btn, "Controls   C", "ghost", 0.38)

        if app_view == "SIGN":
            draw_button(frame, L.top[0], "View   Sign", "ghost", 0.38)
            if is_recording:
                draw_button(frame, L.top[1], "Stop", "alert", 0.44)
            else:
                draw_button(frame, L.top[1], "Start Rec", "solid", 0.44)
            draw_button(frame, L.top[2], f"Sign   {current_mode}", "ghost", 0.38)
            draw_button(frame, L.top[3], "Delete", "ghost", 0.42)
            draw_button(frame, L.top[4], "Clear", "ghost", 0.42)

            card_x = 24
            if current_mode == "SPELL":
                draw_stat_card(frame, (24, 92, 184, 156), "Current sign", f"{predicted_letter}",
                               GOLD, value_scale=1.0)
                card_x = 196
            exp_ratio = (intensity_mult - 1.0) / 1.5
            exp_color = IVORY if intensity_label == "NEUTRAL" else GOLD
            draw_stat_card(frame, (card_x, 92, card_x + 240, 156), "Expression",
                           f"{intensity_label.title()}  {intensity_mult:.2f}x",
                           exp_color, value_scale=0.6, ratio=exp_ratio)

            if is_recording:
                rp = (24, 168, 464, 252)
                draw_panel(frame, rp, alpha=0.78)
                draw_dot(frame, (rp[0] + 22, rp[1] + 25), pulse_color(ROSE), 5)
                label(frame, f"Recording / {current_mode}", (rp[0] + 38, rp[1] + 29), IVORY)
                cursor = "|" if int(time.time() * 2) % 2 == 0 else " "
                if current_word:
                    disp = fit_tail(current_word + cursor, FONT_SERIF, 0.85, 1, rp[2] - rp[0] - 40)
                    put_text(frame, disp, (rp[0] + 20, rp[3] - 20), FONT_SERIF, 0.85, IVORY, 1)
                else:
                    put_text(frame, "Awaiting your first sign", (rp[0] + 20, rp[3] - 20),
                             FONT_SERIF, 0.65, TAUPE, 1)

            if is_recording and not open_hand and not space_gesture_detected and hand_detected and not selecting_synonym:
                if current_mode == "SPELL" and current_holding_letter and current_holding_letter != "-" and letter_hold_start_time:
                    elapsed_l = min(time.time() - letter_hold_start_time, HOLD_LETTER_DURATION)
                    label(frame, f"Holding  {current_holding_letter}", (24, 276), GOLD)
                    draw_progress(frame, (24, 286, 244, 289), elapsed_l / HOLD_LETTER_DURATION, GOLD)
                elif current_mode == "WORD" and asl_word_model is not None:
                    label(frame, "Reading sign", (24, 276), GOLD)
                    draw_progress(frame, (24, 286, 244, 289), len(word_sequence_buffer) / 30.0, SAGE)

            if is_recording:
                label(frame, "Hold two open palms for a space   /   open hand to pause",
                      (w // 2, h - 26), TAUPE, anchor="center", scale=0.32)

            if is_converting_tense:
                bx = modal_box(cx, h - 90, 320, 44)
                draw_panel(frame, bx, alpha=0.88)
                draw_dot(frame, (bx[0] + 26, (bx[1] + bx[3]) // 2), pulse_color(GOLD, speed=6.0), 5)
                label(frame, "Refining sentence...", (bx[0] + 46, (bx[1] + bx[3]) // 2 + 5), IVORY)

            if selecting_synonym:
                draw_modal(frame, L.syn_box, "SELECT WORD INTENT", "Which meaning did you intend?")
                n_opts = min(len(synonym_options), 3)
                for i, r in enumerate(L.syn_row(max(n_opts, 1))):
                    if i < n_opts:
                        draw_button(frame, r, synonym_options[i], "solid" if i == 0 else "ghost", 0.5)

            if selecting_punctuation:
                title, sub = PUNCT_TITLES[punct_sub]
                draw_modal(frame, L.pun_box, title, sub)
                if punct_sub is None:
                    labels = ["Period   .", "Exclaim   !", "Question   ?"]
                    for lab, r in zip(labels, L.pun_row(3)):
                        draw_button(frame, r, lab, "ghost", 0.46)
                else:
                    opts = PUNCT_OPTIONS[punct_sub]
                    for (lab, _, _), r in zip(opts, L.pun_row(len(opts))):
                        draw_button(frame, r, lab, "ghost", 0.46)
                    draw_button(frame, L.pun_back, "< Back", "ghost", 0.36)

            if selecting_tense:
                draw_modal(frame, L.tense_box, "TRANSFORM TENSE", "Choose the tense for the spoken sentence")
                for lab, r in zip(("Present", "Past", "Future", "Original"), L.tense_btns):
                    draw_button(frame, r, lab, "solid" if lab == "Original" else "ghost", 0.46)

            if not is_recording and finished_word and not selecting_punctuation and not selecting_tense:
                fb = L.fin_box
                draw_panel(frame, fb, alpha=0.86)
                label(frame, "Translation", (fb[0] + 36, fb[1] + 40), GOLD)
                label(frame, f"Tone  {selected_tone}", (fb[2] - 36, fb[1] + 40), TAUPE, anchor="right")
                draw_hairline(frame, (fb[0] + 36, fb[1] + 56), (fb[2] - 36, fb[1] + 56))
                draw_wrapped(frame, finished_word, (fb[0] + 36, fb[1] + 102), fb[2] - fb[0] - 72,
                             FONT_SERIF, 0.78, IVORY, 1, 34, 2, tail=False)
                draw_button(frame, L.fin_btns[0], "Speak again", "solid", 0.44)
                draw_button(frame, L.fin_btns[1], "Change tone", "ghost", 0.44)
                draw_button(frame, L.fin_btns[2], "Dismiss", "ghost", 0.44)

        else:
            live_text, live_status, is_listening = stt_thread.get_data()

            draw_button(frame, L.top[0], "View   Text", "ghost", 0.38)
            if is_listening:
                draw_button(frame, L.top[1], "Pause Mic", "alert", 0.42)
            else:
                draw_button(frame, L.top[1], "Start Mic", "solid", 0.42)
            draw_button(frame, L.top[2], "Del Word", "ghost", 0.42)
            draw_button(frame, L.top[3], "Editing" if keyboard_editing_active else "Keyboard",
                        "active" if keyboard_editing_active else "ghost", 0.40)
            draw_button(frame, L.top[4], "Clear", "ghost", 0.42)

            card = modal_box(cx, cy + 36, 940, 410)
            draw_panel(frame, card, alpha=0.80)
            c_x1, c_y1, c_x2, c_y2 = card

            label(frame, "Live transcription", (c_x1 + 40, c_y1 + 52), GOLD, scale=0.40)
            put_text(frame, "For deaf and hard-of-hearing users", (c_x1 + 40, c_y1 + 82),
                     FONT_SERIF, 0.5, TAUPE, 1)

            status_disp = "KEYBOARD EDIT - PRESS ENTER TO FINISH" if keyboard_editing_active else live_status
            live = is_listening or keyboard_editing_active
            dot_col = pulse_color(SAGE) if live else TAUPE
            sw = text_w(status_disp.upper(), FONT_SANS, 0.34, 1, 2)
            draw_dot(frame, (c_x2 - 40 - sw - 16, c_y1 + 47), dot_col, 5)
            label(frame, status_disp, (c_x2 - 40, c_y1 + 52), SAGE if live else TAUPE, anchor="right")

            draw_hairline(frame, (c_x1 + 40, c_y1 + 104), (c_x2 - 40, c_y1 + 104))

            if keyboard_editing_active:
                disp_transcript = live_text + ("|" if int(time.time() * 2) % 2 == 0 else " ")
            else:
                disp_transcript = live_text
            if disp_transcript.strip():
                draw_wrapped(frame, disp_transcript, (c_x1 + 40, c_y1 + 160), c_x2 - c_x1 - 80,
                             FONT_SERIF, 0.95, IVORY, 1, 48, 5, tail=True)
            else:
                put_text(frame, "Waiting for the hearing person to speak...", (c_x1 + 40, c_y1 + 160),
                         FONT_SERIF, 0.8, TAUPE, 1)

            label(frame, "Press T to return to sign view", (w // 2, h - 26), TAUPE,
                  anchor="center", scale=0.32)

        if show_controls_popup:
            cb = modal_box(cx, cy, 560, 410)
            draw_modal(frame, cb, "CONTROLS", "Keyboard and gesture reference")
            rows = [
                ("T", "Switch sign / transcription view"),
                ("S", "Start or stop recording"),
                ("M", "Toggle spell / word mode"),
                ("V", "Open the voice picker"),
                ("C", "Show or hide this panel"),
                ("Q", "Quit the application"),
                ("TWO PALMS", "Hold open for 1s to add a space"),
                ("HOLD 2S", "Hold a sign to type its letter"),
            ]
            for i, (k, d) in enumerate(rows):
                y = cb[1] + 118 + i * 32
                put_text(frame, k, (cb[0] + 44, y), FONT_SANS, 0.4, GOLD, 1, spacing=2)
                put_text(frame, d, (cb[0] + 190, y), FONT_SERIF, 0.48, IVORY, 1)
                if i < len(rows) - 1:
                    draw_hairline(frame, (cb[0] + 44, y + 12), (cb[2] - 44, y + 12), INK_SOFT)

        if show_voices_popup:
            draw_modal(frame, L.voice_box, "VOICE ENGINE", "Choose how S.I.G.N.A.L speaks")
            for idx, (voice_name, r) in enumerate(zip(tts.voices, L.voice_btns)):
                draw_button(frame, r, voice_name, "active" if tts.voice_idx == idx else "ghost", 0.5)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if app_view == "TRANSCRIPTION" and keyboard_editing_active:
            if key != 255:
                if key in (13, 27):
                    keyboard_editing_active = False
                elif key in (8, 127):
                    stt_thread.delete_last_char()
                elif 32 <= key <= 126:
                    stt_thread.append_text(chr(key))
        else:
            if key == ord('q'):
                bg_ai.running = False
                stt_thread.running = False
                break
            elif key == ord('t'):
                if app_view == "SIGN":
                    app_view = "TRANSCRIPTION"
                    stt_thread.start_listening()
                else:
                    app_view = "SIGN"
                    stt_thread.stop_listening()
                    keyboard_editing_active = False
            elif key == ord('v'):
                show_voices_popup = not show_voices_popup
                show_controls_popup = False
            elif key == ord('c'):
                show_controls_popup = not show_controls_popup
                show_voices_popup = False
            elif key == ord('m') and app_view == "SIGN":
                current_mode = "WORD" if current_mode == "SPELL" else "SPELL"
                word_sequence_buffer.clear()
                last_word_pred_time = 0.0
            elif key == ord('s'):
                if app_view == "SIGN":
                    is_recording = not is_recording
                    if is_recording:
                        current_word = ""
                        finished_word = ""
                        selecting_punctuation = False
                        punct_sub = None
                        selecting_tense = False
                        selecting_synonym = False
                        word_sequence_buffer.clear()
                        last_word_pred_time = 0.0
                    else:
                        if current_word.strip():
                            selecting_punctuation = True
                            punct_sub = None
                        else:
                            finished_word = ""
                else:
                    keyboard_editing_active = False
                    live_text, live_status, is_listening = stt_thread.get_data()
                    if is_listening:
                        stt_thread.stop_listening()
                    else:
                        stt_thread.start_listening()

        elapsed = time.time() - loop_start_time
        if elapsed < TARGET_FRAME_TIME:
            time.sleep(TARGET_FRAME_TIME - elapsed)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
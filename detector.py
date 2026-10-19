import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import os
import joblib
import re
from collections import deque
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

COLOR_BG_CARD = (245, 245, 245)
COLOR_BORDER = (210, 210, 210)
COLOR_TEXT_DARK = (40, 40, 40)
COLOR_TEXT_MUTED = (120, 120, 120)
COLOR_SAGE = (120, 170, 130)
COLOR_ROSE = (120, 120, 220)
COLOR_PURPLE = (180, 80, 140)
COLOR_BLUE = (220, 140, 60)
COLOR_PINK = (180, 130, 240)
COLOR_SAND = (180, 210, 230)
COLOR_OVERLAY_BG = (250, 250, 250)

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

def draw_rect(img, pt1, pt2, color, thickness=-1):
    cv2.rectangle(img, pt1, pt2, color, thickness)

def draw_button(img, pt1, pt2, bg_color, text, text_color=(40, 40, 40), font_scale=0.5):
    draw_rect(img, pt1, pt2, bg_color, thickness=-1)
    draw_rect(img, pt1, pt2, COLOR_BORDER, thickness=1)
    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    cx = (pt1[0] + pt2[0]) // 2
    cy = (pt1[1] + pt2[1]) // 2
    cv2.putText(img, text, (cx - t_size[0] // 2, cy + t_size[1] // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)

def draw_progress_bar(img, pt1, pt2, ratio, color=COLOR_PURPLE):
    draw_rect(img, pt1, pt2, (220, 220, 220), thickness=-1)
    w = pt2[0] - pt1[0]
    fill_x2 = pt1[0] + int(w * min(max(ratio, 0.0), 1.0))
    if fill_x2 > pt1[0] + 2:
        draw_rect(img, pt1, (fill_x2, pt2[1]), color, thickness=-1)

def draw_wrapped_text(img, text, pos, max_width, font, font_scale, color, thickness, line_spacing=32):
    words = text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        t_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        if t_size[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    
    x, y = pos
    for i, line in enumerate(lines[-5:]):
        cv2.putText(img, line, (x, y + i * line_spacing), font, font_scale, color, thickness, cv2.LINE_AA)

class TextToSpeechEngine:
    def __init__(self):
        self.voices = ["Voice 1 (Neutral)", "Voice 2 (Expressive)", "Voice 3 (Soft)"]
        self.voice_idx = 0
        self.current_voice_label = self.voices[self.voice_idx]

    def speak(self, text, tone="neutral", intensity=1.0):
        intensity_tag = f" [{intensity}x]" if intensity > 1.2 else ""
        print(f"[TTS ({self.current_voice_label} | {tone}){intensity_tag}]: {text}")

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

mouse_click_pos = None
def on_mouse_click(event, x, y, flags, param):
    global mouse_click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_pos = (x, y)

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

    window_name = "ASL Gesture Translation System"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)

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
        cx, cy = w // 2, h // 2

        predicted_letter = bg_ai.get_prediction()

        if mouse_click_pos is not None:
            mx, my = mouse_click_pos
            mouse_click_pos = None

            if 20 <= mx <= 230 and (h - 95) <= my <= (h - 60):
                show_voices_popup = not show_voices_popup
                show_controls_popup = False

            elif 20 <= mx <= 230 and (h - 50) <= my <= (h - 15):
                show_controls_popup = not show_controls_popup
                show_voices_popup = False

            elif show_controls_popup:
                show_controls_popup = False

            elif show_voices_popup:
                v_box_w, v_box_h = 420, 260
                v_x1, v_y1 = cx - v_box_w // 2, cy - v_box_h // 2
                if (cx - 180) <= mx <= (cx + 180):
                    if (cy - 30) <= my <= (cy + 5):
                        tts.select_voice(0)
                        show_voices_popup = False
                    elif (cy + 15) <= my <= (cy + 50):
                        tts.select_voice(1)
                        show_voices_popup = False
                    elif (cy + 60) <= my <= (cy + 95):
                        tts.select_voice(2)
                        show_voices_popup = False
                else:
                    show_voices_popup = False

            elif (w - 710) <= mx <= (w - 560) and 20 <= my <= 55:
                if app_view == "SIGN":
                    app_view = "TRANSCRIPTION"
                    stt_thread.start_listening()
                else:
                    app_view = "SIGN"
                    stt_thread.stop_listening()
                    keyboard_editing_active = False

            elif (w - 550) <= mx <= (w - 420) and 20 <= my <= 55:
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

            elif (w - 410) <= mx <= (w - 280) and 20 <= my <= 55:
                if app_view == "SIGN":
                    current_mode = "WORD" if current_mode == "SPELL" else "SPELL"
                    selecting_synonym = False
                    word_sequence_buffer.clear()
                    last_word_pred_time = 0.0
                else:
                    stt_thread.delete_last_word()

            elif (w - 270) <= mx <= (w - 150) and 20 <= my <= 55:
                if app_view == "SIGN":
                    current_word = current_word[:-1]
                else:
                    keyboard_editing_active = not keyboard_editing_active
                    if keyboard_editing_active:
                        stt_thread.stop_listening()

            elif (w - 140) <= mx <= (w - 20) and 20 <= my <= 55:
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
                btn_y1, btn_y2 = cy - 10, cy + 50
                if (cx - 210) <= mx <= (cx - 10) and btn_y1 <= my <= btn_y2 and len(synonym_options) >= 1:
                    chosen = synonym_options[0]
                    current_word += f"{chosen} "
                    tts.speak(chosen, tone=selected_tone)
                    selecting_synonym = False
                    synonym_options = []
                elif (cx + 10) <= mx <= (cx + 210) and btn_y1 <= my <= btn_y2 and len(synonym_options) >= 2:
                    chosen = synonym_options[1]
                    current_word += f"{chosen} "
                    tts.speak(chosen, tone=selected_tone)
                    selecting_synonym = False
                    synonym_options = []

            elif selecting_punctuation and app_view == "SIGN":
                btn_y1, btn_y2 = cy - 10, cy + 80
                if punct_sub is None:
                    if (cx - 320) <= mx <= (cx - 120) and btn_y1 <= my <= btn_y2:
                        punct_sub = "PERIOD"
                    elif (cx - 100) <= mx <= (cx + 100) and btn_y1 <= my <= btn_y2:
                        punct_sub = "EXCLAMATION"
                    elif (cx + 120) <= mx <= (cx + 320) and btn_y1 <= my <= btn_y2:
                        punct_sub = "QUESTION"

                elif punct_sub == "PERIOD":
                    if (cx - 300) <= mx <= (cx - 110) and btn_y1 <= my <= btn_y2:
                        temp_sentence, selected_tone = current_word.strip() + ".", "neutral"
                        selecting_punctuation, punct_sub, selecting_tense = False, None, True
                    elif (cx - 95) <= mx <= (cx + 95) and btn_y1 <= my <= btn_y2:
                        temp_sentence, selected_tone = current_word.strip() + ".", "sad"
                        selecting_punctuation, punct_sub, selecting_tense = False, None, True
                    elif (cx + 110) <= mx <= (cx + 300) and btn_y1 <= my <= btn_y2:
                        temp_sentence, selected_tone = current_word.strip() + ".", "sarcastic"
                        selecting_punctuation, punct_sub, selecting_tense = False, None, True

                elif punct_sub == "EXCLAMATION":
                    if (cx - 200) <= mx <= (cx - 10) and btn_y1 <= my <= btn_y2:
                        temp_sentence, selected_tone = current_word.strip() + "!", "happy"
                        selecting_punctuation, punct_sub, selecting_tense = False, None, True
                    elif (cx + 10) <= mx <= (cx + 200) and btn_y1 <= my <= btn_y2:
                        temp_sentence, selected_tone = current_word.strip() + "!", "angry"
                        selecting_punctuation, punct_sub, selecting_tense = False, None, True

                elif punct_sub == "QUESTION":
                    if (cx - 200) <= mx <= (cx - 10) and btn_y1 <= my <= btn_y2:
                        temp_sentence, selected_tone = current_word.strip() + "?", "surprised"
                        selecting_punctuation, punct_sub, selecting_tense = False, None, True
                    elif (cx + 10) <= mx <= (cx + 200) and btn_y1 <= my <= btn_y2:
                        temp_sentence, selected_tone = current_word.strip() + "?", "sarcastic"
                        selecting_punctuation, punct_sub, selecting_tense = False, None, True

            elif selecting_tense and not is_converting_tense and app_view == "SIGN":
                chosen_tense = None
                if (cy - 10) <= my <= (cy + 35):
                    if (cx - 210) <= mx <= (cx - 10): chosen_tense = "NOW"
                    elif (cx + 10) <= mx <= (cx + 210): chosen_tense = "PAST"
                elif (cy + 50) <= my <= (cy + 95):
                    if (cx - 210) <= mx <= (cx - 10): chosen_tense = "FUTURE"
                    elif (cx + 10) <= mx <= (cx + 210): chosen_tense = "ORIGINAL"

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
                btn_y1, btn_y2 = cy + 60, cy + 105
                if (cx - 305) <= mx <= (cx - 115) and btn_y1 <= my <= btn_y2:
                    tts.speak(finished_word, selected_tone)
                elif (cx - 95) <= mx <= (cx + 95) and btn_y1 <= my <= btn_y2:
                    if word_history and word_history[-1] == finished_word: word_history.pop()
                    finished_word = ""
                    selecting_punctuation, punct_sub = True, None
                elif (cx + 115) <= mx <= (cx + 305) and btn_y1 <= my <= btn_y2:
                    finished_word = ""
                    current_word = ""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        hand_detected = False
        space_gesture_detected = False
        open_hand = False
        features = None

        if app_view == "SIGN":
            small_rgb = cv2.resize(rgb_frame, (640, 360))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb)
            
            frame_timestamp_ms = int(time.time() * 1000) - start_time_ms
            if frame_timestamp_ms <= last_timestamp_ms:
                frame_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = frame_timestamp_ms

            detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

            #run face & pose every 3 frames 
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

            if pose_lms:
                for start_idx, end_idx in ARM_CONNECTIONS:
                    s_lm = pose_lms[start_idx]
                    e_lm = pose_lms[end_idx]
                    if s_lm.visibility > 0.5 and e_lm.visibility > 0.5:
                        start_p = (int(s_lm.x * w), int(s_lm.y * h))
                        end_p = (int(e_lm.x * w), int(e_lm.y * h))
                        cv2.line(frame, start_p, end_p, (200, 200, 200), 3, cv2.LINE_AA)
                for idx in KEY_ARM_INDICES:
                    lm = pose_lms[idx]
                    if lm.visibility > 0.5:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 6, COLOR_PURPLE, -1, cv2.LINE_AA)

            if detection_result.hand_landmarks:
                valid_hands = [h for h in detection_result.hand_landmarks if is_valid_hand_shape(h)]
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
                        if space_start_time is None: space_start_time = time.time()
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

                    for hand_landmarks in valid_hands:
                        for connection in HAND_CONNECTIONS:
                            start_p = (int(hand_landmarks[connection[0]].x * w), int(hand_landmarks[connection[0]].y * h))
                            end_p = (int(hand_landmarks[connection[1]].x * w), int(hand_landmarks[connection[1]].y * h))
                            cv2.line(frame, start_p, end_p, (230, 235, 240), 2, cv2.LINE_AA)
                        for landmark in hand_landmarks:
                            cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 4, COLOR_PURPLE, -1, cv2.LINE_AA)
                else:
                    smoothed_landmarks = None
                    bg_ai.update_data(None, None)
                    space_start_time, letter_hold_start_time, current_holding_letter, prev_wrist_pos = None, None, None, None
            else:
                smoothed_landmarks = None
                bg_ai.update_data(None, None)
                space_start_time, letter_hold_start_time, current_holding_letter, prev_wrist_pos = None, None, None, None
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
                                    synonym_options = [w.strip() for w in word_str.split("/") if w.strip()]
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

        view_bg = COLOR_PURPLE if app_view == "SIGN" else COLOR_SAGE
        draw_button(frame, (w - 710, 20), (w - 560, 55), view_bg, f"MODE: {app_view}", text_color=(255, 255, 255), font_scale=0.45)

        if app_view == "SIGN":
            rec_bg = COLOR_SAGE if not is_recording else COLOR_ROSE
            rec_btn_text = "STOP" if is_recording else "START REC"
            draw_button(frame, (w - 550, 20), (w - 420, 55), rec_bg, rec_btn_text, text_color=(255, 255, 255), font_scale=0.50)

            mode_bg = COLOR_PURPLE if current_mode == "SPELL" else COLOR_SAND
            mode_txt_color = (255, 255, 255) if current_mode == "SPELL" else COLOR_TEXT_DARK
            draw_button(frame, (w - 410, 20), (w - 280, 55), mode_bg, f"SIGN: {current_mode}", text_color=mode_txt_color, font_scale=0.45)

            draw_button(frame, (w - 270, 20), (w - 150, 55), COLOR_BLUE, "DELETE", text_color=(255, 255, 255), font_scale=0.55)
            draw_button(frame, (w - 140, 20), (w - 20, 55), COLOR_PINK, "CLEAR", text_color=(255, 255, 255), font_scale=0.55)

            if current_mode == "SPELL":
                draw_rect(frame, (20, 20), (200, 65), COLOR_BG_CARD, thickness=-1)
                draw_rect(frame, (20, 20), (200, 65), COLOR_BORDER, thickness=1)
                cv2.putText(frame, "CURRENT SIGN", (32, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{predicted_letter}", (32, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

            draw_rect(frame, (215, 20), (415, 65), COLOR_BG_CARD, thickness=-1)
            draw_rect(frame, (215, 20), (415, 65), COLOR_BORDER, thickness=1)
            cv2.putText(frame, "EMOTION INTENSITY", (227, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{intensity_label} ({intensity_mult}x)", (227, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_SAGE if intensity_label == "NEUTRAL" else COLOR_PURPLE, 2, cv2.LINE_AA)

            if is_recording:
                draw_rect(frame, (20, 80), (430, 160), COLOR_BG_CARD, thickness=-1)
                draw_rect(frame, (20, 80), (430, 160), COLOR_BORDER, thickness=1)
                cv2.putText(frame, f"RECORDING ({current_mode})", (32, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_SAGE, 2, cv2.LINE_AA)
                disp_word = f"{current_word}_" if current_word else "..."
                cv2.putText(frame, disp_word, (32, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

            if is_recording and not open_hand and not space_gesture_detected and hand_detected and not selecting_synonym:
                if current_mode == "SPELL" and current_holding_letter and current_holding_letter != "-" and letter_hold_start_time:
                    letter_elapsed = min(time.time() - letter_hold_start_time, HOLD_LETTER_DURATION)
                    l_ratio = letter_elapsed / HOLD_LETTER_DURATION
                    cv2.putText(frame, f"Holding ('{current_holding_letter}')...", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_DARK, 1, cv2.LINE_AA)
                    draw_progress_bar(frame, (20, 188), (220, 200), l_ratio, color=COLOR_PURPLE)

                elif current_mode == "WORD" and asl_word_model is not None:
                    buf_len = len(word_sequence_buffer)
                    w_ratio = buf_len / 30.0
                    cv2.putText(frame, f"Holding (Sign/Word detecting)...", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_DARK, 1, cv2.LINE_AA)
                    draw_progress_bar(frame, (20, 188), (220, 200), w_ratio, color=COLOR_SAGE)

            draw_button(frame, (20, h - 95), (230, h - 60), COLOR_BG_CARD, "Change Voice Engine", text_color=COLOR_TEXT_DARK, font_scale=0.40)
            draw_button(frame, (20, h - 50), (230, h - 15), COLOR_BG_CARD, "Controls ", text_color=COLOR_TEXT_DARK, font_scale=0.40)

            if selecting_synonym:
                box_w, box_h = 520, 160
                m_x1, m_y1 = cx - box_w // 2, cy - box_h // 2
                m_x2, m_y2 = cx + box_w // 2, cy + box_h // 2
                draw_rect(frame, (m_x1, m_y1), (m_x2, m_y2), COLOR_OVERLAY_BG, thickness=-1)
                draw_rect(frame, (m_x1, m_y1), (m_x2, m_y2), COLOR_BORDER, thickness=1)

                title = "SELECT WORD INTENT"
                t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
                cv2.putText(frame, title, (cx - t_size[0] // 2, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

                btn_h = 45
                if len(synonym_options) >= 1:
                    opt1 = synonym_options[0]
                    draw_button(frame, (cx - 210, cy - 10), (cx - 10, cy - 10 + btn_h), COLOR_SAGE, opt1, text_color=(255, 255, 255), font_scale=0.6)

                if len(synonym_options) >= 2:
                    opt2 = synonym_options[1]
                    draw_button(frame, (cx + 10, cy - 10), (cx + 210, cy - 10 + btn_h), COLOR_PURPLE, opt2, text_color=(255, 255, 255), font_scale=0.6)

            if selecting_punctuation:
                box_w, box_h = 700, 180
                m_x1, m_y1 = cx - box_w // 2, cy - box_h // 2
                m_x2, m_y2 = cx + box_w // 2, cy + box_h // 2
                draw_rect(frame, (m_x1, m_y1), (m_x2, m_y2), COLOR_OVERLAY_BG, thickness=-1)
                draw_rect(frame, (m_x1, m_y1), (m_x2, m_y2), COLOR_BORDER, thickness=1)

                if punct_sub is None:
                    title = "SELECT PUNCTUATION"
                    t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
                    cv2.putText(frame, title, (cx - t_size[0] // 2, cy - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

                    draw_button(frame, (cx - 320, cy - 10), (cx - 120, cy + 50), COLOR_SAND, ". (Period)", text_color=COLOR_TEXT_DARK, font_scale=0.55)
                    draw_button(frame, (cx - 100, cy - 10), (cx + 100, cy + 50), COLOR_SAGE, "! (Exclamation)", text_color=(255, 255, 255), font_scale=0.55)
                    draw_button(frame, (cx + 120, cy - 10), (cx + 320, cy + 50), COLOR_PURPLE, "? (Question)", text_color=(255, 255, 255), font_scale=0.55)

                elif punct_sub == "PERIOD":
                    title = "PERIOD (.) INTENT & TONE"
                    t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
                    cv2.putText(frame, title, (cx - t_size[0] // 2, cy - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

                    draw_button(frame, (cx - 300, cy - 10), (cx - 110, cy + 50), COLOR_SAND, "Neutral", text_color=COLOR_TEXT_DARK, font_scale=0.55)
                    draw_button(frame, (cx - 95, cy - 10), (cx + 95, cy + 50), COLOR_ROSE, "Sad", text_color=(255, 255, 255), font_scale=0.55)
                    draw_button(frame, (cx + 110, cy - 10), (cx + 300, cy + 50), COLOR_PURPLE, "Sarcastic", text_color=(255, 255, 255), font_scale=0.55)

                elif punct_sub == "EXCLAMATION":
                    title = "EXCLAMATION (!) TONE"
                    t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
                    cv2.putText(frame, title, (cx - t_size[0] // 2, cy - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

                    draw_button(frame, (cx - 200, cy - 10), (cx - 10, cy + 50), COLOR_SAGE, "Happy", text_color=(255, 255, 255), font_scale=0.55)
                    draw_button(frame, (cx + 10, cy - 10), (cx + 200, cy + 50), COLOR_PURPLE, "Angry", text_color=(255, 255, 255), font_scale=0.55)

                elif punct_sub == "QUESTION":
                    title = "QUESTION (?) TONE"
                    t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
                    cv2.putText(frame, title, (cx - t_size[0] // 2, cy - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

                    draw_button(frame, (cx - 200, cy - 10), (cx - 10, cy + 50), COLOR_SAND, "Surprised", text_color=COLOR_TEXT_DARK, font_scale=0.55)
                    draw_button(frame, (cx + 10, cy - 10), (cx + 200, cy + 50), COLOR_PURPLE, "Sarcastic", text_color=(255, 255, 255), font_scale=0.55)

            if selecting_tense:
                box_w, box_h = 520, 180
                m_x1, m_y1 = cx - box_w // 2, cy - box_h // 2
                m_x2, m_y2 = cx + box_w // 2, cy + box_h // 2
                draw_rect(frame, (m_x1, m_y1), (m_x2, m_y2), COLOR_OVERLAY_BG, thickness=-1)
                draw_rect(frame, (m_x1, m_y1), (m_x2, m_y2), COLOR_BORDER, thickness=1)

                title = "TRANSFORM TENSE"
                t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
                cv2.putText(frame, title, (cx - t_size[0] // 2, cy - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

                draw_button(frame, (cx - 210, cy - 10), (cx - 10, cy + 35), COLOR_SAGE, "PRESENT (NOW)", text_color=(255, 255, 255), font_scale=0.5)
                draw_button(frame, (cx + 10, cy - 10), (cx + 210, cy + 35), COLOR_PURPLE, "PAST", text_color=(255, 255, 255), font_scale=0.5)
                draw_button(frame, (cx - 210, cy + 50), (cx - 10, cy + 95), COLOR_SAND, "FUTURE", text_color=COLOR_TEXT_DARK, font_scale=0.5)
                draw_button(frame, (cx + 10, cy + 50), (cx + 210, cy + 95), COLOR_ROSE, "ORIGINAL", text_color=(255, 255, 255), font_scale=0.5)

            if not is_recording and finished_word and not selecting_punctuation and not selecting_tense:
                box_w, box_h = 650, 120
                m_x1, m_y1 = cx - box_w // 2, cy + 10
                m_x2, m_y2 = cx + box_w // 2, cy + 130
                draw_rect(frame, (m_x1, m_y1), (m_x2, m_y2), COLOR_BG_CARD, thickness=-1)
                draw_rect(frame, (m_x1, m_y1), (m_x2, m_y2), COLOR_BORDER, thickness=1)

                cv2.putText(frame, f"TRANSLATION: \"{finished_word}\"", (m_x1 + 20, m_y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

                draw_button(frame, (cx - 305, cy + 60), (cx - 115, cy + 105), COLOR_SAGE, "SPEAK AGAIN", text_color=(255, 255, 255), font_scale=0.5)
                draw_button(frame, (cx - 95, cy + 60), (cx + 95, cy + 105), COLOR_PURPLE, "CHANGE TONE", text_color=(255, 255, 255), font_scale=0.5)
                draw_button(frame, (cx + 115, cy + 60), (cx + 305, cy + 105), COLOR_ROSE, "DISMISS", text_color=(255, 255, 255), font_scale=0.5)

        else:
            live_text, live_status, is_listening = stt_thread.get_data()

            mic_bg = COLOR_ROSE if is_listening else COLOR_SAGE
            mic_text = "PAUSE MIC" if is_listening else "START MIC"
            draw_button(frame, (w - 550, 20), (w - 420, 55), mic_bg, mic_text, text_color=(255, 255, 255), font_scale=0.50)
            
            draw_button(frame, (w - 410, 20), (w - 280, 55), COLOR_BLUE, "DEL WORD", text_color=(255, 255, 255), font_scale=0.50)

            kbd_bg = COLOR_SAGE if keyboard_editing_active else COLOR_SAND
            kbd_txt_col = (255, 255, 255) if keyboard_editing_active else COLOR_TEXT_DARK
            kbd_label = "EDITING..." if keyboard_editing_active else "KEYBOARD"
            draw_button(frame, (w - 270, 20), (w - 150, 55), kbd_bg, kbd_label, text_color=kbd_txt_col, font_scale=0.45)

            draw_button(frame, (w - 140, 20), (w - 20, 55), COLOR_PINK, "CLEAR", text_color=(255, 255, 255), font_scale=0.55)

            card_w, card_h = 920, 380
            c_x1, c_y1 = cx - card_w // 2, cy - card_h // 2 + 30
            c_x2, c_y2 = cx + card_w // 2, cy + card_h // 2 + 30

            draw_rect(frame, (c_x1, c_y1), (c_x2, c_y2), COLOR_OVERLAY_BG, thickness=-1)
            draw_rect(frame, (c_x1, c_y1), (c_x2, c_y2), COLOR_BORDER, thickness=2)

            cv2.putText(frame, "LIVE TRANSCRIPTION FOR DEAF / HARD OF HEARING USER", (c_x1 + 25, c_y1 + 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_TEXT_DARK, 2, cv2.LINE_AA)
            
            status_disp = "KEYBOARD EDIT ACTIVE (Press ENTER to finish)" if keyboard_editing_active else f"STATUS: {live_status}"
            status_color = COLOR_SAGE if (is_listening or keyboard_editing_active) else COLOR_PURPLE
            cv2.putText(frame, status_disp, (c_x1 + 25, c_y1 + 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 2, cv2.LINE_AA)

            cv2.line(frame, (c_x1 + 25, c_y1 + 95), (c_x2 - 25, c_y1 + 95), COLOR_BORDER, 1, cv2.LINE_AA)

            disp_transcript = live_text
            if keyboard_editing_active:
                disp_transcript += "|"
            elif not live_text:
                disp_transcript = "Waiting for hearing person to speak..."

            txt_color = COLOR_TEXT_DARK if (live_text or keyboard_editing_active) else COLOR_TEXT_MUTED
            draw_wrapped_text(frame, disp_transcript, (c_x1 + 25, c_y1 + 135), card_w - 50, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, txt_color, 2, line_spacing=38)

        if show_controls_popup:
            c_box_w, c_box_h = 480, 340
            c_x1, c_y1 = cx - c_box_w // 2, cy - c_box_h // 2
            c_x2, c_y2 = cx + c_box_w // 2, cy + c_box_h // 2

            draw_rect(frame, (c_x1, c_y1), (c_x2, c_y2), COLOR_OVERLAY_BG, thickness=-1)
            draw_rect(frame, (c_x1, c_y1), (c_x2, c_y2), COLOR_BORDER, thickness=2)

            cv2.putText(frame, "Controls", (c_x1 + 25, c_y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.70, COLOR_TEXT_DARK, 2, cv2.LINE_AA)
            cv2.line(frame, (c_x1 + 25, c_y1 + 48), (c_x2 - 25, c_y1 + 48), COLOR_BORDER, 1, cv2.LINE_AA)

            ctrl_lines = [
                "Toggle view: 't'",
                "Start / stop: press button or 's'",
                "Toggle mode: Click Mode or 'm'",
                "Space: Hold open two palms",
                "Delete: Click delete",
                "Clear all: Press clear",
                "Type letter: hold sign 2s",
                "Toggle voice: 'v'",
                "Quit app: 'q'"
            ]

            for idx, line in enumerate(ctrl_lines):
                y_pos = c_y1 + 75 + (idx * 26)
                cv2.putText(frame, line, (c_x1 + 30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_DARK, 1, cv2.LINE_AA)

        if show_voices_popup:
            v_box_w, v_box_h = 420, 240
            v_x1, v_y1 = cx - v_box_w // 2, cy - v_box_h // 2
            v_x2, v_y2 = cx + v_box_w // 2, cy + v_box_h // 2

            draw_rect(frame, (v_x1, v_y1), (v_x2, v_y2), COLOR_OVERLAY_BG, thickness=-1)
            draw_rect(frame, (v_x1, v_y1), (v_x2, v_y2), COLOR_BORDER, thickness=2)

            cv2.putText(frame, "Pick a voice", (v_x1 + 25, v_y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.70, COLOR_TEXT_DARK, 2, cv2.LINE_AA)
            cv2.line(frame, (v_x1 + 25, v_y1 + 48), (v_x2 - 25, v_y1 + 48), COLOR_BORDER, 1, cv2.LINE_AA)

            for idx, voice_name in enumerate(tts.voices):
                btn_top = cy - 30 + (idx * 45)
                btn_color = COLOR_SAGE if tts.voice_idx == idx else COLOR_SAND
                txt_color = (255, 255, 255) if tts.voice_idx == idx else COLOR_TEXT_DARK
                draw_button(frame, (cx - 180, btn_top), (cx + 180, btn_top + 35), btn_color, voice_name, text_color=txt_color, font_scale=0.52)

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

        #micro sleep to cap loop at 30ish fps
        elapsed = time.time() - loop_start_time
        if elapsed < TARGET_FRAME_TIME:
            time.sleep(TARGET_FRAME_TIME - elapsed)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
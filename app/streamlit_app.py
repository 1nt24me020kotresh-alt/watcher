import streamlit as st
import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av, sys
sys.path.append('.')
from src.feature_extraction import extract_features

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(page_title='Watcher', page_icon='🐦', layout='centered')
st.title('🐦 Watcher')
st.subheader('Driver Cognitive State Detection — Live')
st.markdown('Allow camera access when prompted. Results update in real time.')
st.divider()

# ── Browser voice alert via Web Speech API ───────────────────────────────
# This JavaScript runs in the user's browser — voice comes from their
# speakers, not the server. Works on Chrome, Edge, Safari.
st.components.v1.html("""
<script>
// How often the browser is allowed to speak (milliseconds)
const ALERT_INTERVAL_MS = 4000;
let lastAlertTime = 0;
let alertActive   = false;

function triggerVoiceAlert(message) {
    const now = Date.now();
    if (now - lastAlertTime < ALERT_INTERVAL_MS) return;
    if (window.speechSynthesis.speaking) return;
    lastAlertTime = now;
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate   = 0.9;
    utterance.volume = 1.0;
    utterance.pitch  = 1.0;
    window.speechSynthesis.speak(utterance);
}

// Poll the shared alert flag that Python sets via query params
function checkAlertFlag() {
    const params = new URLSearchParams(window.location.search);
    const danger  = params.get('danger');
    if (danger === '1') {
        triggerVoiceAlert('Warning. Do not drive.');
    }
}
setInterval(checkAlertFlag, 500);
</script>
""", height=0)

# ── Load model and scaler ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('models/watcher_model.pkl')
    scaler = joblib.load('models/watcher_scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ── Load FaceLandmarker ──────────────────────────────────────────────────
@st.cache_resource
def load_detector():
    base_opts = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)

detector = load_detector()

# ── State config ─────────────────────────────────────────────────────────
STATE_COLORS_BGR = {
    'alert':    (0, 220, 0),
    'tired':    (0, 165, 255),
    'stressed': (0, 165, 255),
    'drowsy':   (0, 0, 255),
    'impaired': (0, 0, 255),
    'no_face':  (180, 180, 180),
}

DANGER_WEIGHTS = {
    'alert':    1.0,
    'tired':    0.55,
    'stressed': 0.50,
    'drowsy':   0.10,
    'impaired': 0.05,
}

ALERT_THRESHOLD = 30

def get_fit_score(features_scaled):
    proba   = model.predict_proba(features_scaled)[0]
    classes = list(model.classes_)
    score   = sum(proba[i] * DANGER_WEIGHTS.get(classes[i], 0.5)
                  for i in range(len(classes))) * 100
    state      = classes[int(np.argmax(proba))]
    confidence = float(np.max(proba)) * 100
    return int(score), state, confidence

def draw_overlay(frame, state, score, confidence, ear, mar):
    color = STATE_COLORS_BGR.get(state, (180, 180, 180))
    h, w  = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f'STATE: {state.upper()}  ({confidence:.0f}%)',
        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    bar_x, bar_y, bar_w = 15, 65, 220
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18), (50, 50, 50), -1)
    bar_color = (0, 220, 0) if score > 70 else (0, 165, 255) if score > 40 else (0, 0, 255)
    cv2.rectangle(frame, (bar_x, bar_y),
        (bar_x + int(bar_w * score / 100), bar_y + 18), bar_color, -1)
    cv2.putText(frame, f'Fit-to-drive: {score}%',
        (bar_x + bar_w + 10, bar_y + 13),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1)

    if score < ALERT_THRESHOLD:
        # Red flash overlay
        danger_overlay = frame.copy()
        cv2.rectangle(danger_overlay, (0, 0), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(danger_overlay, 0.15, frame, 0.85, 0, frame)
        cv2.putText(frame, 'DO NOT DRIVE', (15, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.putText(frame, f'EAR {ear:.3f}   MAR {mar:.3f}',
        (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    return frame

# ── Shared state between video processor and Streamlit UI ────────────────
# VideoProcessorBase runs in a thread — we use instance attributes
# to pass the latest score back to the main Streamlit thread.
class WatcherProcessor(VideoProcessorBase):
    def __init__(self):
        self.score      = 100
        self.state      = 'no_face'
        self.confidence = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img  = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        state      = 'no_face'
        score      = 100
        confidence = 0.0
        ear = mar  = 0.0

        if result.face_landmarks:
            lms              = result.face_landmarks[0]
            ear, mar, coords = extract_features(lms, w, h)
            features_raw     = np.array([[ear, mar] + coords])
            features_scaled  = scaler.transform(features_raw)
            score, state, confidence = get_fit_score(features_scaled)

        # Store on instance so Streamlit UI can read it
        self.score      = score
        self.state      = state
        self.confidence = confidence

        img = draw_overlay(img, state, score, confidence, ear, mar)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── WebRTC streamer ───────────────────────────────────────────────────────
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

ctx = webrtc_streamer(
    key="watcher",
    video_processor_factory=WatcherProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
)

# ── Live score panel below the video ─────────────────────────────────────
# Reads score from the processor and updates the UI + triggers JS alert
if ctx.video_processor:
    score      = ctx.video_processor.score
    state      = ctx.video_processor.state
    confidence = ctx.video_processor.confidence

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric('State',            state.upper())
    col2.metric('Fit-to-drive',     f'{score}%')
    col3.metric('Confidence',       f'{confidence:.0f}%')
    st.progress(score / 100)

    if score < ALERT_THRESHOLD:
        st.error('🚫 DO NOT DRIVE — Dangerous state detected')
        # Inject JS to trigger browser voice alert
        st.components.v1.html("""
        <script>
            if (window.speechSynthesis) {
                const u = new SpeechSynthesisUtterance('Warning. Do not drive.');
                u.rate = 0.9; u.volume = 1.0;
                window.speechSynthesis.speak(u);
            }
        </script>
        """, height=0)
    elif score < 55:
        st.warning('⚠️ CAUTION — Signs of fatigue detected')
    else:
        st.success('✅ FIT TO DRIVE')

st.divider()
st.caption('EAR < 0.25 = drowsy threshold · MAR > 0.60 = yawning threshold')

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pyttsx3
import threading
import time

st.title("Fire Event Detection")

# Camera selection in sidebar
st.sidebar.header("Camera Settings")
camera_index = st.sidebar.number_input(
    "Select Camera Index",
    min_value=0,
    max_value=10,
    value=0,
    step=1,
    help="0 = Default webcam, 1 = External camera, etc."
)
st.sidebar.info("💡 Tip: Try 0 for built-in webcam, 1 for external USB camera")

# Load model
model = YOLO("Fire_Event_best.pt")

# Custom CSS for flashy alert
st.markdown("""
    <style>
    .flashy-alert {
        font-size: 28px !important;
        font-weight: bold;
        color: white !important;
        background: linear-gradient(90deg, #ff0000, #ff8000, #ffff00, #ff8000, #ff0000);
        background-size: 300% 300%;
        animation: flashy 1.5s infinite alternate;
        padding: 15px !important;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
        margin: 10px 0;
    }
    @keyframes flashy {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize TTS engine (once) with robust error handling
@st.cache_resource
def get_tts_engine():
    try:
        engine = pyttsx3.init()
        
        # Try to set properties safely - skip voice setting entirely
        try:
            engine.setProperty('rate', 150)  # Slower, more reliable rate
        except:
            pass  # Continue without setting rate
            
        try:
            engine.setProperty('volume', 0.8)  # Lower volume
        except:
            pass  # Continue without setting volume
            
        # Test if engine works by trying a simple operation
        try:
            voices = engine.getProperty('voices')
            # Don't set voice - use default
        except:
            pass
            
        return engine
        
    except Exception as e:
        st.sidebar.warning(f"🔇 TTS not available in cloud environment")
        return None

tts_engine = get_tts_engine()
last_spoken = ""
last_spoken_time = 0

# Placeholders
frame_placeholder = st.empty()
message_placeholder = st.empty()
status_placeholder = st.empty()

# Lock for TTS
tts_lock = threading.Lock()

def speak_async(text):
    global last_spoken, last_spoken_time
    with tts_lock:
        if text == last_spoken and time.time() - last_spoken_time < 5:
            return
        last_spoken = text
        last_spoken_time = time.time()
        
        if tts_engine:
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception as e:
                # Silently handle TTS errors in production
                st.sidebar.info(f"🔊 {text}")
        else:
            # Fallback - display in sidebar
            st.sidebar.info(f"🔊 {text}")

# Initialize camera
cap = None
try:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        status_placeholder.error(f"❌ Failed to open camera {camera_index}. Try a different index!")
        st.stop()
    else:
        status_placeholder.success(f"✅ Using camera {camera_index}")
except Exception as e:
    status_placeholder.error(f"❌ Camera error: {str(e)}")
    st.stop()

# Store last detection state
last_detection = ""

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            status_placeholder.warning("⚠️ Camera disconnected or frame read failed")
            break

        # Run detection
        results = model(frame)
        annotated = results[0].plot()
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Process detections
        names = [results[0].names[int(box.cls[0])] for box in results[0].boxes]
        current_detection = ""

        if names:
            unique_names = list(set(names))
            current_detection = ', '.join(unique_names)
            alert_text = f"🚨 DETECTED: {current_detection} 🚨"
            
            message_placeholder.markdown(
                f'<div class="flashy-alert">{alert_text}</div>', 
                unsafe_allow_html=True
            )

            # Voice alert for new detections
            if current_detection != last_detection:
                speak_async(f"Marine vessel detected: {current_detection}")
                last_detection = current_detection
        else:
            message_placeholder.empty()
            last_detection = ""

        # Small delay to reduce CPU
        time.sleep(0.1)

except KeyboardInterrupt:
    pass
finally:
    if cap is not None:
        cap.release()
    status_placeholder.info("⏹️ Camera released")

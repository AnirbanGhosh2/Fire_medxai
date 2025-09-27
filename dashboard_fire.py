import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.set_page_config(page_title="🔥 Fire Detection App", layout="wide")
st.title("🔥 Fire Event Detection")

# Sidebar input selection
st.sidebar.header("Input Settings")
input_type = st.sidebar.selectbox(
    "Select Input Type",
    ["Live Camera Feed", "Take Photo", "Upload File"],
    help="Choose how you want to provide input"
)

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("Fire_Event_best.pt")

model = load_model()

# Custom CSS for flashy alert
st.markdown("""
    <style>
    .flashy-alert {
        font-size: 24px !important;
        font-weight: bold;
        color: white !important;
        background: linear-gradient(90deg, #ff0000, #ff8000, #ffff00, #ff8000, #ff0000);
        background-size: 300% 300%;
        animation: flashy 1.5s infinite alternate;
        padding: 10px !important;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    @keyframes flashy {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    </style>
""", unsafe_allow_html=True)

frame_placeholder = st.empty()
message_placeholder = st.empty()

# 🔹 Case 1: Live Camera Feed using WebRTC
if input_type == "Live Camera Feed":
    st.info("🎥 Live webcam works directly in browser using WebRTC")

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.last_detection = ""

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            img = frame.to_ndarray(format="bgr24")

            # Run YOLO inference
            results = model(img)
            annotated = results[0].plot()

            # Process detections
            names = [results[0].names[int(box.cls[0])] for box in results[0].boxes]
            if names:
                unique_names = list(set(names))
                self.last_detection = ", ".join(unique_names)
            else:
                self.last_detection = ""

            return annotated

    ctx = webrtc_streamer(
        key="fire-detection",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

    # 🔑 Continuously check detection state and update UI
    if ctx.video_transformer:
        detection_placeholder = st.empty()

        while ctx.state.playing:
            if ctx.video_transformer.last_detection:
                alert_text = f"🚨 DETECTED: {ctx.video_transformer.last_detection} 🚨"
                detection_placeholder.markdown(
                    f'<div class="flashy-alert">{alert_text}</div>',
                    unsafe_allow_html=True
                )
            else:
                detection_placeholder.success("✅ No fire detected in live feed")

            time.sleep(0.5)


# 🔹 Case 2: Take Photo
elif input_type == "Take Photo":
    st.info("📸 Use your webcam to take a snapshot")

    camera_photo = st.camera_input("Take a photo for fire detection")
    if camera_photo:
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        results = model(frame)
        annotated = results[0].plot()
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        names = [results[0].names[int(box.cls[0])] for box in results[0].boxes]
        if names:
            unique_names = list(set(names))
            current_detection = ", ".join(unique_names)
            alert_text = f"🚨 DETECTED: {current_detection} 🚨"
            message_placeholder.markdown(
                f'<div class="flashy-alert">{alert_text}</div>',
                unsafe_allow_html=True
            )
        else:
            message_placeholder.success("✅ No fire detected in photo")

# 🔹 Case 3: Upload File
elif input_type == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image or Video",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov']
    )

    if uploaded_file is None:
        st.info("👆 Please upload an image or video to start detection")
        st.stop()

    if uploaded_file.type.startswith("image"):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        results = model(frame)
        annotated = results[0].plot()
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        names = [results[0].names[int(box.cls[0])] for box in results[0].boxes]
        if names:
            unique_names = list(set(names))
            alert_text = f"🚨 DETECTED: {', '.join(unique_names)} 🚨"
            message_placeholder.markdown(
                f'<div class="flashy-alert">{alert_text}</div>',
                unsafe_allow_html=True
            )
        else:
            message_placeholder.success("✅ No fire detected in image")

    else:
        temp_file = f"temp_video.{uploaded_file.name.split('.')[-1]}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_file)
        st.success("✅ Processing uploaded video")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            names = [results[0].names[int(box.cls[0])] for box in results[0].boxes]
            if names:
                alert_text = f"🚨 DETECTED: {', '.join(set(names))} 🚨"
                message_placeholder.markdown(
                    f'<div class="flashy-alert">{alert_text}</div>',
                    unsafe_allow_html=True
                )
            else:
                message_placeholder.success("✅ No fire detected in video frame")

            time.sleep(0.1)

        cap.release()

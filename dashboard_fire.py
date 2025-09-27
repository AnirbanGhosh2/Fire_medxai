import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time

st.title("🔥 Fire Event Detection")

# Sidebar options
st.sidebar.header("Input Settings")
input_type = st.sidebar.selectbox(
    "Select Input Type",
    ["Take Photo", "Upload File"],  # Removed "Live Camera Feed"
    help="Live webcam feed is not supported on Streamlit Cloud. Use photo or upload instead."
)

# Load model once
@st.cache_resource
def load_model():
    return YOLO("Fire_Event_best.pt")

model = load_model()

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

# Placeholders
frame_placeholder = st.empty()
message_placeholder = st.empty()

# Case 1: Take Photo with browser camera
if input_type == "Take Photo":
    st.info("📸 Use your webcam to take a snapshot (works in browser & Streamlit Cloud)")

    camera_photo = st.camera_input("Take a photo for fire detection")

    if camera_photo is not None:
        # Convert the photo to OpenCV format
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Run YOLO detection
        results = model(frame)
        annotated = results[0].plot()
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Process detections
        names = [results[0].names[int(box.cls[0])] for box in results[0].boxes]
        if names:
            unique_names = list(set(names))
            current_detection = ', '.join(unique_names)
            alert_text = f"🚨 DETECTED: {current_detection} 🚨"
            message_placeholder.markdown(
                f'<div class="flashy-alert">{alert_text}</div>', 
                unsafe_allow_html=True
            )
            st.success(f"🔥 Fire detection result: **{current_detection}**")
        else:
            message_placeholder.success("✅ No fire detected in photo")
            st.success("✅ **No fire detected** - Image appears safe")

# Case 2: Upload File (image or video)
elif input_type == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image or Video",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov']
    )

    if uploaded_file is None:
        st.info("👆 Please upload an image or video file to start detection")
        st.stop()

    if uploaded_file.type.startswith('image'):
        # Process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        results = model(frame)
        annotated = results[0].plot()
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        names = [results[0].names[int(box.cls[0])] for box in results[0].boxes]
        if names:
            unique_names = list(set(names))
            current_detection = ', '.join(unique_names)
            alert_text = f"🚨 DETECTED: {current_detection} 🚨"
            message_placeholder.markdown(
                f'<div class="flashy-alert">{alert_text}</div>', 
                unsafe_allow_html=True
            )
        else:
            message_placeholder.success("✅ No fire detected in image")

    else:
        # Save video temporarily
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
                unique_names = list(set(names))
                current_detection = ', '.join(unique_names)
                alert_text = f"🚨 DETECTED: {current_detection} 🚨"
                message_placeholder.markdown(
                    f'<div class="flashy-alert">{alert_text}</div>', 
                    unsafe_allow_html=True
                )
            else:
                message_placeholder.success("✅ No fire detected in video frame")

            time.sleep(0.1)

        cap.release()

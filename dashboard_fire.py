import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time

st.title("Fire Event Detection")

# Input selection in sidebar
st.sidebar.header("Input Settings")
input_type = st.sidebar.selectbox(
    "Select Input Type",
    ["Live Camera Feed", "Take Photo", "Upload File"],
    help="Choose your preferred input method"
)

if input_type == "Live Camera Feed":
    camera_index = st.sidebar.number_input(
        "Select Camera Index",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="0 = Default webcam, 1 = External camera, etc. (Works locally only)"
    )
    st.sidebar.info("💡 Live feed works when running locally")
    
elif input_type == "Take Photo":
    st.sidebar.info("📸 Use your device camera to take a photo")
    
else:  # Upload File
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image or Video",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
        help="Upload an image or video for fire detection"
    )

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

# Placeholders
frame_placeholder = st.empty()
message_placeholder = st.empty()
status_placeholder = st.empty()

# Initialize input based on selection
if input_type == "Take Photo":
    # Use st.camera_input for web-based camera access
    camera_photo = st.camera_input("Take a photo for fire detection")
    
    if camera_photo is not None:
        # Convert the photo to OpenCV format
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        # Run detection on the photo
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
        
        # Add option to take another photo
        if st.button("📸 Take Another Photo"):
            st.rerun()
            
    else:
        st.info("👆 Click the camera button above to take a photo")
        
    st.stop()  # End here for photo mode

elif input_type == "Live Camera Feed":
    cap = None
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            status_placeholder.error(f"❌ Failed to open camera {camera_index}. Camera only works locally!")
            st.info("💡 **Running on cloud?** Try 'Take Photo' option instead")
            st.stop()
        else:
            status_placeholder.success(f"✅ Using camera {camera_index}")
    except Exception as e:
        status_placeholder.error(f"❌ Camera error: {str(e)}")
        st.info("💡 **Running on cloud?** Try 'Take Photo' option instead")
        st.stop()

else:  # Upload File
    if uploaded_file is None:
        st.info("👆 Please upload an image or video file to start detection")
        st.stop()
    
    # Handle uploaded file
    if uploaded_file.type.startswith('image'):
        # Process single image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        # Run detection on single image
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
        else:
            message_placeholder.success("✅ No fire detected in image")
        
        st.stop()  # End here for image processing
    
    else:
        # Process video file
        temp_file = f"temp_video.{uploaded_file.name.split('.')[-1]}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_file)
        status_placeholder.success("✅ Processing uploaded video")

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

            # Update last detection
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

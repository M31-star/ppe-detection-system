import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from collections import Counter

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PPE Detection Dashboard", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # auto-download model

model = load_model()

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp { background: #020617; color: #e5e7eb; }
h1 { color: white; text-align: center; }
section[data-testid="stSidebar"] { background: #020617; }
.stFileUploader { border: 2px dashed white; padding: 20px; border-radius: 12px; }
button { background-color: white !important; color: black !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>🦺 PPE Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Real-Time Safety Monitoring</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
option = st.sidebar.radio("Select Mode", ["Image Upload", "Video Upload", "Camera"])

# ---------------- ALERT ----------------
def play_alert():
    st.markdown(
        """
        <audio autoplay>
        <source src="https://www.soundjay.com/buttons/beep-01a.mp3" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True,
    )

# ---------------- PROCESS FRAME ----------------
def process_frame(frame):
    results = model(frame)
    classes = []

    for r in results:
        if r.boxes is not None:
            classes = r.boxes.cls.tolist()

    classes = [int(c) for c in classes]
    counts = Counter(classes)
    annotated = results[0].plot()

    return annotated, counts

# ---------------- SAFETY CHECK ----------------
def check_safety(counts):
    helmet = counts.get(0, 0)
    vest = counts.get(2, 0)
    gloves = counts.get(1, 0)
    boots = counts.get(3, 0)
    persons = counts.get(6, 0)

    missing = []
    if helmet < persons: missing.append("Helmet")
    if vest < persons: missing.append("Vest")
    if gloves < persons: missing.append("Gloves")
    if boots < persons: missing.append("Boots")

    return persons, helmet, vest, gloves, boots, missing

# ---------------- DASHBOARD ----------------
def show_dashboard(counts):
    persons, helmet, vest, gloves, boots, missing = check_safety(counts)

    st.subheader("📊 PPE Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👷 Persons", persons)
    c2.metric("🪖 Helmets", helmet)
    c3.metric("🦺 Vests", vest)
    c4.metric("🧤 Gloves", gloves)
    c5.metric("🥾 Boots", boots)

    if persons > 0 and missing:
        st.error(f"⚠️ Missing: {', '.join(missing)}")
        play_alert()
    else:
        st.success("✅ All PPE Present")

# ---------------- IMAGE ----------------
if option == "Image Upload":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        annotated, counts = process_frame(img)

        col1, col2 = st.columns(2)
        col1.image(img, channels="BGR", caption="Original")
        col2.image(annotated, channels="BGR", caption="Detection")

        show_dashboard(counts)

# ---------------- VIDEO ----------------
elif option == "Video Upload":
    video = st.file_uploader("Upload Video", type=["mp4", "avi"])

    if video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_window = st.image([])
        stats = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, counts = process_frame(frame)
            frame_window.image(annotated, channels="BGR")

            persons, helmet, vest, gloves, boots, missing = check_safety(counts)
            stats.markdown(f"""
            👷 {persons} | 🪖 {helmet} | 🦺 {vest} | 🧤 {gloves} | 🥾 {boots}
            """)

            if persons > 0 and missing:
                st.warning(f"Missing: {', '.join(missing)}")
                play_alert()

        cap.release()

# ---------------- CAMERA ----------------
elif option == "Camera":
    st.warning("⚠️ Camera not supported on Streamlit Cloud. Use Image/Video upload.")
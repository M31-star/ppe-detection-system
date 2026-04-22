import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
from collections import Counter

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PPE Detection Dashboard", layout="wide")

# ---------------- UI STYLE (WHITE THEME) ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: #020617;
    color: #e5e7eb;
}

/* Title */
h1 {
    color: white;
    text-align: center;
    font-weight: 700;
}

/* Subtitle */
p {
    text-align: center;
    color: #9ca3af;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1f2937;
}

/* Sidebar options */
div[role="radiogroup"] label {
    background: #111827;
    padding: 10px;
    margin: 6px 0;
    border-radius: 8px;
    cursor: pointer;
    border: 1px solid transparent;
}

div[role="radiogroup"] label:hover {
    border: 1px solid white;
}

/* Selected option */
div[role="radiogroup"] input:checked + div {
    background: white !important;
    color: black !important;
    font-weight: bold;
}

/* File uploader (MAIN HIGHLIGHT) */
.stFileUploader {
    background: #111827;
    border: 2px dashed white;
    padding: 25px;
    border-radius: 15px;
}

/* Buttons */
button {
    background-color: white !important;
    color: black !important;
    border-radius: 10px !important;
    font-weight: bold;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid white;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
}

/* Section headers */
h2, h3 {
    color: white;
}

/* Divider */
hr {
    border: 1px solid #1f2937;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("""
<h1>🦺 PPE Detection System</h1>
<p>Real-Time Safety Monitoring Dashboard</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "runs/detect/ppe_model-2/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found! Check path.")
    st.stop()

MODEL_PATH = "yolov8n.pt"

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Control Panel")

option = st.sidebar.radio(
    "Select Mode",
    ["Image Upload", "Video Upload", "Camera"]
)

# ---------------- SOUND ALERT ----------------
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

    if helmet < persons:
        missing.append("Helmet")
    if vest < persons:
        missing.append("Vest")
    if gloves < persons:
        missing.append("Gloves")
    if boots < persons:
        missing.append("Boots")

    return persons, helmet, vest, gloves, boots, missing

# ---------------- DASHBOARD ----------------
def show_dashboard(counts):
    persons, helmet, vest, gloves, boots, missing = check_safety(counts)

    st.markdown("## 📊 PPE Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👷 Persons", persons)
    c2.metric("🪖 Helmets", helmet)
    c3.metric("🦺 Vests", vest)
    c4.metric("🧤 Gloves", gloves)
    c5.metric("🥾 Boots", boots)

    st.markdown("## 🧠 Safety Status")

    if persons > 0 and missing:
        st.markdown(f"""
        <div style='background:#111827; border:1px solid white; padding:15px; border-radius:10px'>
        ⚠️ <b>UNSAFE:</b> Missing {', '.join(missing)}
        </div>
        """, unsafe_allow_html=True)
        play_alert()
    else:
        st.markdown("""
        <div style='background:#111827; border:1px solid white; padding:15px; border-radius:10px'>
        ✅ <b>SAFE:</b> All PPE present
        </div>
        """, unsafe_allow_html=True)

# ---------------- IMAGE ----------------
if option == "Image Upload":

    st.markdown("### 📤 Upload Image")

    uploaded_file = st.file_uploader("Choose file", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        annotated, counts = process_frame(img)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Original", channels="BGR")

        with col2:
            st.image(annotated, caption="Detection", channels="BGR")

        show_dashboard(counts)

# ---------------- VIDEO ----------------
elif option == "Video Upload":

    st.markdown("### 🎥 Upload Video")

    uploaded_video = st.file_uploader("Choose video", type=["mp4", "avi"])

    if uploaded_video is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        FRAME = st.image([])
        stats = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, counts = process_frame(frame)

            FRAME.image(annotated, channels="BGR")

            persons, helmet, vest, gloves, boots, missing = check_safety(counts)

            stats.markdown(f"""
            ### 📊 Live Stats  
            👷 Persons: {persons}  
            🪖 Helmets: {helmet}  
            🦺 Vests: {vest}  
            🧤 Gloves: {gloves}  
            🥾 Boots: {boots}
            """)

            if persons > 0 and missing:
                st.warning(f"⚠️ Missing: {', '.join(missing)}")
                play_alert()

        cap.release()

# ---------------- CAMERA ----------------
elif option == "Camera":

    st.markdown("### 📸 Live Camera Detection")

    run = st.checkbox("Start Camera")

    FRAME = st.image([])
    stats = st.empty()

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()

        if not ret:
            st.error("❌ Camera not working")
            break

        annotated, counts = process_frame(frame)

        FRAME.image(annotated, channels="BGR")

        persons, helmet, vest, gloves, boots, missing = check_safety(counts)

        stats.markdown(f"""
        ### 📊 Live Stats  
        👷 Persons: {persons}  
        🪖 Helmets: {helmet}  
        🦺 Vests: {vest}  
        🧤 Gloves: {gloves}  
        🥾 Boots: {boots}
        """)

        if persons > 0 and missing:
            st.warning(f"⚠️ Missing: {', '.join(missing)}")
            play_alert()

    cap.release()
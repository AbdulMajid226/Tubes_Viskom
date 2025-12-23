import streamlit as st
import cv2
import av
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- CONFIG PAGE ---
st.set_page_config(page_title="Drowsiness Detection", page_icon="😴")
st.title("😴 Driver Drowsiness Detection")
st.caption("Real-time Inference via WebRTC (Deployment Ready)")

# --- LOAD MODEL ---
# Cache model agar tidak reload setiap ada frame baru
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- PARAMETER ---
confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# --- CALLBACK PROCESSOR ---
# Fungsi ini akan dipanggil untuk setiap frame video yang masuk
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    
    # 1. Inference YOLO
    results = model(image, conf=confidence, verbose=False)
    
    # 2. Gambar Kotak Deteksi
    annotated_frame = results[0].plot()
    
    # 3. Logika Alert (Digambar langsung di Video agar realtime)
    # Kita tidak bisa pakai st.error() di dalam callback webrtc
    if len(results[0].boxes) > 0:
        cls_id = int(results[0].boxes.cls[0])
        label_name = model.names[cls_id]
        
        if label_name == "drowsy": # Sesuaikan dengan label Anda
            # Tambahkan Teks Merah Besar di Layar
            cv2.putText(annotated_frame, "BAHAYA: MENGANTUK!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            # Tambahkan Border Merah
            cv2.rectangle(annotated_frame, (0,0), (image.shape[1], image.shape[0]), (0,0,255), 10)
    
    # Kembalikan frame yang sudah diedit ke browser
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- MAIN INTERFACE ---
st.info("Klik 'START' dan izinkan akses kamera browser Anda.")

webrtc_streamer(
    key="drowsiness-detection",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={  # STUN Server (Agar jalan di HP/Jaringan berbeda)
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

st.markdown("---")
st.write("Dibuat untuk Tugas Besar Visi Komputer.")
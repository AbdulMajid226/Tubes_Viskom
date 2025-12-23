import streamlit as st
import cv2
import av
import numpy as np
import tempfile
import os
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Drowsiness Detection",
    page_icon="😴",
    layout="centered"
)

# --- JUDUL APLIKASI ---
st.title("😴 Driver Drowsiness Detection")
st.caption("Tugas Besar Visi Komputer - Universitas Diponegoro")

# --- SIDEBAR (PANEL KONTROL) ---
st.sidebar.header("Panel Kontrol")
mode = st.sidebar.selectbox("Pilih Mode Input", ["Realtime Webcam", "Upload Gambar/Video"])
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

st.sidebar.divider()
st.sidebar.info(
    "Webcam Mode: Menggunakan WebRTC untuk deteksi real-time.\n\n"
    "Upload Mode: Memproses file video dan menyimpannya untuk diputar ulang."
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Pastikan file best.pt ada di folder yang sama
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- LOGIKA WEBCAM (Hanya Streaming, Tanpa Playback) ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    
    # 1. Inference YOLO
    results = model(image, conf=confidence, verbose=False)
    annotated_frame = results[0].plot()
    
    # 2. Logika Peringatan (Alert)
    if len(results[0].boxes) > 0:
        cls_id = int(results[0].boxes.cls[0])
        label_name = model.names[cls_id]
        
        if label_name == "drowsy": # Pastikan nama label sesuai dataset Anda
            # Tambahkan Teks Merah Besar
            cv2.putText(annotated_frame, "BAHAYA: MENGANTUK!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            # Tambahkan Kotak Merah di Sekeliling Layar
            cv2.rectangle(annotated_frame, (0,0), (image.shape[1], image.shape[0]), (0,0,255), 20)
            
    # Kembalikan frame ke browser (Real-time display)
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- TAMPILAN UTAMA ---

if mode == "Realtime Webcam":
    st.subheader("📡 Real-time Detection")
    st.write("Klik 'START' dan izinkan akses kamera browser.")
    
    webrtc_streamer(
        key="drowsiness-detection",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

elif mode == "Upload Gambar/Video":
    st.subheader("📂 Upload File")
    uploaded_file = st.file_uploader("Pilih file gambar atau video...", type=['jpg', 'jpeg', 'png', 'mp4'])

    if uploaded_file:
        # --- JIKA GAMBAR ---
        if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            results = model(img_array, conf=confidence)
            res_plotted = results[0].plot()
            
            st.image(res_plotted, caption="Hasil Deteksi Gambar", use_container_width=True)

        # --- JIKA VIDEO (ADA PLAYBACK) ---
        elif uploaded_file.type == "video/mp4":
            st.warning("Sedang memproses video... Harap tunggu hingga 100%.")
            
            # Simpan input sementara
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            
            # Setup Output Video
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            output_path = "hasil_deteksi.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Progress Bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Proses Deteksi
                results = model(frame, conf=confidence)
                res_plotted = results[0].plot()
                
                # Simpan ke file output
                out.write(res_plotted)
                
                # Update Progress
                frame_count += 1
                if total_frames > 0:
                    progress_value = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress_value)
                    status_text.text(f"Processing Frame: {frame_count}/{total_frames}")

            cap.release()
            out.release()
            
            progress_bar.empty()
            status_text.empty()
            
            # Tampilkan Hasil Playback
            st.success("Selesai! Berikut hasil deteksinya:")
            
            # Trik agar video terbaca browser: Baca sebagai binary
            try:
                with open(output_path, 'rb') as f:
                    video_bytes = f.read()
                    st.video(video_bytes)
            except Exception as e:
                st.error("Gagal memuat player video. Cek file 'hasil_deteksi.mp4' di folder project Anda.")
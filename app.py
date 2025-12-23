import streamlit as st
import cv2
import av
import numpy as np
import tempfile
import os
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer

# --- CONFIG PAGE ---
st.set_page_config(page_title="Drowsiness Detection", page_icon="😴", layout="centered")

# --- JUDUL & SIDEBAR ---
st.title("😴 Driver Drowsiness Detection")
st.caption("Tugas Besar Visi Komputer - Realtime & Video Playback")

st.sidebar.header("Panel Kontrol")
mode = st.sidebar.selectbox("Pilih Mode Input", ["Realtime Webcam", "Upload Gambar/Video"])
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- LOGIKA WEBCAM (WEBRTC) ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    
    # Inference
    results = model(image, conf=confidence, verbose=False)
    annotated_frame = results[0].plot()
    
    # Alert Logic
    if len(results[0].boxes) > 0:
        cls_id = int(results[0].boxes.cls[0])
        label_name = model.names[cls_id]
        if label_name == "drowsy":
            cv2.putText(annotated_frame, "MENGANTUK!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.rectangle(annotated_frame, (0,0), (image.shape[1], image.shape[0]), (0,0,255), 10)
            
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- MAIN APP LOGIC ---
if mode == "Realtime Webcam":
    st.info("Mode Webcam: Klik 'START' di bawah. Izinkan browser mengakses kamera.")
    webrtc_streamer(
        key="drowsiness-detection",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

elif mode == "Upload Gambar/Video":
    st.info("Mode Upload: Unggah file foto atau video untuk dianalisis.")
    uploaded_file = st.file_uploader("Upload file...", type=['jpg', 'jpeg', 'png', 'mp4'])

    if uploaded_file:
        # --- PROSES GAMBAR ---
        if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
            image = Image.open(uploaded_file)
            st.subheader("Hasil Deteksi:")
            img_array = np.array(image)
            results = model(img_array, conf=confidence)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Processed Image", use_container_width=True)
            
            if len(results[0].boxes) > 0:
                label = model.names[int(results[0].boxes.cls[0])]
                st.write(f"**Terdeteksi:** {label}")

        # --- PROSES VIDEO (DENGAN PLAYBACK) ---
        elif uploaded_file.type == "video/mp4":
            st.write("Sedang memproses video... Harap tunggu.")
            
            # 1. Simpan file input sementara
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            
            # Ambil properti video asli
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 2. Siapkan Video Writer untuk Output
            output_path = "output_hasil_deteksi.mp4"
            # 'mp4v' adalah codec yang paling aman untuk Windows/OpenCV standar
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Placeholder untuk Progress Bar & Frame Preview
            progress_bar = st.progress(0)
            st_frame = st.empty()
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Inference YOLO
                results = model(frame, conf=confidence)
                res_plotted = results[0].plot()
                
                # Tulis frame hasil ke file output
                out.write(res_plotted)
                
                # Tampilkan Preview Live (Optional - biar user gak bosen nunggu)
                st_frame.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption=f"Processing Frame {frame_count}/{total_frames}", use_container_width=True)
                
                # Update Progress Bar
                frame_count += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out.release()
            
            # Bersihkan elemen UI preview
            st_frame.empty()
            progress_bar.empty()
            
            # 3. Tampilkan Hasil Akhir (Playback)
            st.success("Selesai! Berikut hasil videonya:")
            st.divider()
            
            # Konversi file agar kompatibel dengan Browser (Opsional trick)
            # Karena Streamlit kadang rewel dengan codec mp4v, kita baca ulang sebagai bytes
            try:
                video_file = open(output_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
            except Exception as e:
                st.error("Video selesai diproses tapi gagal dimuat di player browser.")
                st.write(f"File tersimpan di folder project dengan nama: {output_path}")
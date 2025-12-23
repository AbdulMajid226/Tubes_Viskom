import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sistem Deteksi Kelelahan",
    page_icon="😴",
    layout="centered"
)

# --- JUDUL & SIDEBAR ---
st.title("😴 Driver Drowsiness Detection")
st.caption("Tugas Besar Visi Komputer - Deteksi Kelelahan Realtime")

st.sidebar.header("Konfigurasi")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
use_webcam = st.sidebar.toggle("Gunakan Webcam", value=False)

# --- LOAD MODEL ---
# Pastikan file best.pt ada di satu folder dengan file ini
try:
    model = YOLO("best.pt")
except Exception as e:
    st.error(f"Error loading model: {e}. Pastikan file 'best.pt' ada di folder yang sama.")
    st.stop()

# --- UTILS: TAMPILAN STATUS ---
def display_status(label):
    if label == "drowsy": # Sesuaikan dengan nama label di Roboflow Anda
        st.error("⚠️ PERINGATAN: TERDETEKSI MENGANTUK!")
    elif label == "awake":
        st.success("✅ Status: Aman (Terjaga)")
    else:
        st.info(f"Terdeteksi: {label}")

# --- MAIN LOGIC (WEBCAM) ---
if use_webcam:
    cap = cv2.VideoCapture(0) # 0 biasanya ID webcam default laptop
    
    if not cap.isOpened():
        st.error("Tidak bisa membuka webcam.")
    
    # Placeholder untuk gambar video
    frame_placeholder = st.empty()
    stop_button = st.button("Stop Deteksi")
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.write("Gagal membaca frame webcam.")
            break
        
        # 1. Lakukan Deteksi dengan YOLO
        results = model(frame, conf=confidence, verbose=False)
        
        # 2. Visualisasi (Gambar Kotak di Frame)
        annotated_frame = results[0].plot()
        
        # 3. Logika Peringatan (Ambil label pertama yang terdeteksi)
        if len(results[0].boxes) > 0:
            # Ambil ID kelas dari deteksi pertama
            cls_id = int(results[0].boxes.cls[0])
            label_name = model.names[cls_id]
            
            # Tampilkan alert di bawah video (hanya 1 baris agar tidak spam)
            with st.sidebar:
                st.write(f"Terdeteksi: **{label_name}**")
                if label_name == "drowsy":
                    st.markdown("# 🚨 BANGUN!")
        
        # 4. Tampilkan ke Streamlit (Convert BGR ke RGB)
        frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
    cap.release()
else:
    st.info("Aktifkan toggle 'Gunakan Webcam' di sebelah kiri untuk memulai demo.")
    
    # --- BONUS: UPLOAD GAMBAR/VIDEO (Syarat Tugas) ---
    st.divider()
    st.subheader("Uji Coba File Gambar/Video")
    uploaded_file = st.file_uploader("Upload gambar/video...", type=['jpg', 'png', 'mp4'])
    
    if uploaded_file:
        # Simpan file sementara
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        if uploaded_file.name.endswith('.mp4'):
            st.video(tfile.name)
            st.write("Untuk video, silakan gunakan mode Webcam untuk realtime inference.")
        else:
            image = cv2.imread(tfile.name)
            results = model(image, conf=confidence)
            res_plotted = results[0].plot()
            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi")
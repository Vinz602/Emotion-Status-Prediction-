import streamlit as st
import requests

# Konfigurasi Halaman UI
st.set_page_config(page_title="Emotion Status Prediction", page_icon="📊", layout="centered")

st.title("📊 Social Media Activity - Emotion Status Prediction")
st.write("Masukkan metrik aktivitas dan data profil pengguna di bawah ini untuk memprediksi status emosi.")
st.markdown("---")

st.subheader("Form Data Pengguna")

# Membuat layout kolom agar tampilan form rapi
col1, col2 = st.columns(2)

with col1:
    # --- FITUR KATEGORIKAL ---
    gender = st.selectbox("Jenis Kelamin (Gender)", options=["Male", "Female", "Non-Binary"])
    
    platform = st.selectbox("Platform Media Sosial", options=["Instagram", "Twitter", "Facebook", "Whatsapp", "YouTube", "Snapchat", "LinkedIn"])
    
    # --- FITUR NUMERIK (Bagian 1) ---
    age = st.number_input("Usia (Age)", min_value=1, max_value=120, value=20, step=1)
    daily_usage = st.number_input("Waktu Penggunaan Harian (Daily_Usage_Time dalam menit)", min_value=0, value=60, step=5)

with col2:
    # --- FITUR NUMERIK (Bagian 2) ---
    posts_per_day = st.number_input("Jumlah Postingan per Hari (Posts_Per_Day)", min_value=0, value=2, step=1)
    likes_received = st.number_input("Likes yang Diterima per Hari (Likes_Received_Per_Day)", min_value=0, value=10, step=1)
    comments_received = st.number_input("Komentar yang Diterima per Hari (Comments_Received_Per_Day)", min_value=0, value=5, step=1)
    messages_sent = st.number_input("Pesan yang Dikirim per Hari (Messages_Sent_Per_Day)", min_value=0, value=15, step=1)

st.markdown("---")

# Tombol untuk memicu prediksi
if st.button("Prediksi Status Emosi", type="primary"):
    with st.spinner("Mengirim data ke FastAPI backend..."):
        try:
            # Sesuaikan URL_API dengan endpoint FastAPI kamu
            URL_API = "http://localhost:8000/predict"
            
            # Payload JSON menyertakan semua fitur numerik dan kategorikal
            # Pastikan KEY (nama string) di bawah ini SAMA PERSIS dengan nama kolom di DataFrame saat kamu training model
            payload = {
                "Gender": gender,
                "Platform": platform,
                "Age": age,
                "Daily_Usage_Time (minutes)": daily_usage,
                "Posts_Per_Day": posts_per_day,
                "Likes_Received_Per_Day": likes_received,
                "Comments_Received_Per_Day": comments_received,
                "Messages_Sent_Per_Day": messages_sent
            }
            
            # Kirim POST request ke FastAPI
            response = requests.post(URL_API, json=payload)
            
            if response.status_code == 200:
                hasil_json = response.json()
                
                # Mengambil hasil prediksi dari FastAPI
                emosi = hasil_json.get("prediction", "Tidak Terdeteksi")
                probabilitas = hasil_json.get("probability", None)
                
                st.success("Analisis Selesai!")
                st.metric(label="Hasil Prediksi Status Emosi:", value=emosi)
                
                if probabilitas is not None:
                    st.info(f"Tingkat Keyakinan Model: {probabilitas:.2f}")
            else:
                st.error(f"Gagal mendapatkan respon dari backend. Status Code: {response.status_code}")
                st.json(response.json()) # Menampilkan log error dari FastAPI untuk debugging
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Gagal terhubung ke Backend FastAPI. Pastikan server uvicorn sudah dijalankan di port 8000.")
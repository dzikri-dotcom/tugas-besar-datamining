import streamlit as st
import pandas as pd
import pickle
import os

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="Prediksi Harga Rumah", layout="wide")

# ==========================================
# LOAD MODEL
# ==========================================
path_model = 'models/model_project.pkl'

if not os.path.exists(path_model):
    st.error("Model belum ditemukan! Jalankan 'python notebooks/training_logic.py' dulu.")
    st.stop()

with open(path_model, 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    feature_names = data['features']

# ==========================================
# SIDEBAR INPUT
# ==========================================
st.sidebar.title("Spesifikasi Rumah")
st.sidebar.write("Masukkan detail rumah:")

input_data = {}

# Mapping nama kolom ke label yang lebih jelas
labels = {
    'LB': 'Luas Bangunan (m2)',
    'LT': 'Luas Tanah (m2)',
    'KT': 'Jumlah Kamar Tidur',
    'KM': 'Jumlah Kamar Mandi',
    'GRS': 'Kapasitas Garasi (Mobil)'
}

for col in feature_names:
    label = labels.get(col, col) # Pakai label jelas jika ada, jika tidak pakai nama kolom asli
    # Nilai default disesuaikan agar logis
    val = 0
    if col == 'LB': val = 100
    if col == 'LT': val = 120
    if col == 'KT': val = 3
    if col == 'KM': val = 2
    if col == 'GRS': val = 1
    
    input_data[col] = st.sidebar.number_input(label, min_value=0, value=val)

input_df = pd.DataFrame([input_data])

# ==========================================
# MAIN PAGE
# ==========================================
st.title("🏡 Aplikasi Prediksi Harga Rumah")
st.write("Berdasarkan data: `DATA_RUMAH.csv`")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Data Masukan")
    st.dataframe(input_df)

with col2:
    st.subheader("Hasil Prediksi")
    if st.button("🔍 Hitung Harga", type="primary"):
        prediksi = model.predict(input_df)[0]
        
        st.metric("Estimasi Harga", f"Rp {prediksi:,.0f}")
        
        # Contoh kategori sederhana (bisa disesuaikan)
        if prediksi < 1000000000:
            st.success("Kategori: Ekonomis / Terjangkau")
        elif prediksi < 5000000000:
            st.warning("Kategori: Menengah")
        else:
            st.error("Kategori: Mewah")
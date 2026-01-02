import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# ==========================================
# KONFIGURASI
# ==========================================
NAMA_FILE_DATA = 'DATA_RUMAH.csv'
TARGET_KOLOM = 'HARGA'
KOLOM_YANG_DIHAPUS = ['NO', 'NAMA RUMAH']

# ==========================================
# 1. LOAD DATA
# ==========================================
path_data = os.path.join(os.path.dirname(__file__), '../data', NAMA_FILE_DATA)
if not os.path.exists(path_data):
    print(f"ERROR: File tidak ditemukan di {path_data}")
    exit()

df = pd.read_csv(path_data)
print(f"Data berhasil dimuat. Total baris: {df.shape[0]}")

# ==========================================
# 2. DATA PREPARATION
# ==========================================
df_clean = df.drop(columns=[col for col in KOLOM_YANG_DIHAPUS if col in df.columns], errors='ignore')
df_clean = df_clean.dropna()

# Bersihkan data object menjadi angka
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean[col] = df_clean[col].astype(str).str.replace(r'[^0-9]', '', regex=True)
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
df_clean = df_clean.dropna()

X = df_clean.drop(columns=[TARGET_KOLOM])
y = df_clean[TARGET_KOLOM]
feature_names = X.columns.tolist()

# ==========================================
# 3. CLUSTERING & VISUALISASI (INI YANG BARU)
# ==========================================
print("Sedang melakukan Clustering...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# --- BAGIAN PLOTTING ---
plt.figure(figsize=(10, 6))
# Sumbu X: Luas Bangunan (LB), Sumbu Y: Harga
plt.scatter(df_clean['LB'], df_clean['HARGA'], c=clusters, cmap='viridis', alpha=0.6)
plt.title('Visualisasi Persebaran Cluster (Harga vs Luas Bangunan)')
plt.xlabel('Luas Bangunan (m2)')
plt.ylabel('Harga (Rp)')
plt.colorbar(label='Cluster')
plt.grid(True, alpha=0.3)

# Simpan Gambar
path_gambar = os.path.join(os.path.dirname(__file__), '../visualisasi_cluster.png')
plt.savefig(path_gambar)
print(f"GAMBAR BERHASIL DISIMPAN DI: {path_gambar}")
# -----------------------

# ==========================================
# 4. MODELING
# ==========================================
print("Sedang melatih Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Akurasi Model (R^2): {model.score(X_test, y_test):.2f}")

# ==========================================
# 5. SIMPAN MODEL
# ==========================================
path_model = os.path.join(os.path.dirname(__file__), '../models', 'model_project.pkl')
with open(path_model, 'wb') as f:
    pickle.dump({'model': model, 'features': feature_names}, f)
print("Selesai!")
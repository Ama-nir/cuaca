import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. BACA DATA HASIL PREPARATION
df = pd.read_excel('env/Cuaca/Cuaca_Gabungan_Bersih.xlsx')
kolom_numerik = ['TAVG', 'RH_AVG', 'RR', 'FF_AVG']

# 2. AGREGASI — rata-rata per stasiun (1 baris = 1 stasiun)
df_stasiun = df.groupby('STASIUN')[kolom_numerik].mean().round(2).reset_index()
print("Data per stasiun:")
print(df_stasiun)

# 3. NORMALISASI
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_stasiun[kolom_numerik])

# 4. ELBOW — cari K optimal
wcss = []
K_range = range(2, len(df_stasiun))  # max K = jumlah stasiun - 1

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(data_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o')
plt.title('Metode Elbow')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.savefig('elbow.png', dpi=150)
plt.show()

# 5. K-MEANS — ganti K sesuai hasil grafik Elbow
K_OPTIMAL = 3  # ← ubah setelah lihat grafik

km = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
df_stasiun['Cluster'] = km.fit_predict(data_scaled)

# 6. EVALUASI — Silhouette Score
skor = silhouette_score(data_scaled, df_stasiun['Cluster'])
print(f"\nSilhouette Score: {skor:.4f}")
print("(Mendekati 1 = bagus, mendekati 0 = jelek)")

# 7. HASIL — stasiun masuk cluster mana
print("\nHasil Clustering:")
print(df_stasiun[['STASIUN', 'Cluster'] + kolom_numerik])

print("\nRata-rata tiap cluster:")
print(df_stasiun.groupby('Cluster')[kolom_numerik].mean().round(2))

# 8. VISUALISASI — scatter plot sederhana
warna = ['red', 'green', 'blue', 'orange', 'purple']

plt.figure(figsize=(8, 6))
for i in range(K_OPTIMAL):
    subset = df_stasiun[df_stasiun['Cluster'] == i]
    plt.scatter(subset['TAVG'], subset['RH_AVG'],
                c=warna[i], label=f'Cluster {i}', s=100)
    for _, row in subset.iterrows():
        plt.annotate(row['STASIUN'], (row['TAVG'], row['RH_AVG']), fontsize=8)

plt.title('Hasil Clustering Stasiun (Suhu vs Kelembaban)')
plt.xlabel('TAVG (°C)')
plt.ylabel('RH_AVG (%)')
plt.legend()
plt.grid(True)
plt.savefig('cluster.png', dpi=150)
plt.show()

# Menyimpan Hasil
df_stasiun.to_excel('Hasil_Clustering.xlsx', index=False)
print("\nSelesai! File disimpan: Hasil_Clustering.xlsx")
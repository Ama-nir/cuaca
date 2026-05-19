import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. BACA DATA HASIL PREPARATION
df = pd.read_excel('env/Cuaca/Cuaca_Gabungan_Bersih.xlsx')
kolom_numerik = ['TAVG', 'RH_AVG', 'RR', 'FF_AVG']

# 2. NORMALISASI LANGSUNG DATA HARIAN (bukan agregasi)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[kolom_numerik])

# 3. ELBOW — cari K optimal
wcss = []
K_range = range(2, 11)

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

# 4. K-MEANS
K_OPTIMAL = 3  # ← ubah setelah lihat grafik

km = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
df['Cluster'] = km.fit_predict(data_scaled)

# 5. EVALUASI — Silhouette Score
skor = silhouette_score(data_scaled, df['Cluster'])
print(f"\nSilhouette Score: {skor:.4f}")
print("(Mendekati 1 = bagus, mendekati 0 = jelek)")

# 6. KARAKTERISTIK TIAP CLUSTER
print("\nRata-rata tiap cluster:")
print(df.groupby('Cluster')[kolom_numerik].mean().round(2))

print("\nDistribusi cluster (% dari total data):")
print(df['Cluster'].value_counts(normalize=True).mul(100).round(1))

# 7. ANALISIS PER STASIUN — stasiun mana dominan di cluster mana
print("\nDistribusi Cluster per Stasiun (%):")
distribusi = df.groupby('STASIUN')['Cluster'].value_counts(normalize=True)
distribusi = distribusi.mul(100).round(1).unstack(fill_value=0)
distribusi.columns = [f'Cluster {c}' for c in distribusi.columns]
print(distribusi)

# 8. VISUALISASI — pola cuaca harian
warna = ['red', 'green', 'blue', 'orange', 'purple']

plt.figure(figsize=(8, 6))
for i in range(K_OPTIMAL):
    subset = df[df['Cluster'] == i]
    plt.scatter(subset['TAVG'], subset['RH_AVG'],
                c=warna[i], label=f'Cluster {i}', s=20, alpha=0.5)

plt.title('Hasil Clustering Data Harian (Suhu vs Kelembaban)')
plt.xlabel('TAVG (°C)')
plt.ylabel('RH_AVG (%)')
plt.legend()
plt.grid(True)
plt.savefig('cluster.png', dpi=150)
plt.show()

# 9. SIMPAN HASIL
df.to_excel('Hasil_Clustering.xlsx', index=False)
print("\nSelesai! File disimpan: Hasil_Clustering.xlsx")

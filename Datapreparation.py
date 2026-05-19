import pandas as pd
import numpy as np
import glob
import os

kolom = ['TANGGAL', 'TAVG', 'RH_AVG', 'RR', 'FF_AVG']
kolom_numerik = ['TAVG', 'RH_AVG', 'RR', 'FF_AVG']
nilai_error = [9999, 8888, 9998, -9999, -99, 99.9]

# Ngebaca dan ngebangun data dari semua file Excel di folder
folder = 'env/Cuaca/'
semua_file = glob.glob(os.path.join(folder, '*.xlsx'))
semua_file.sort()

print(f"File ditemukan: {len(semua_file)} stasiun")

list_df = []

for file in semua_file:
    # Ambil nama stasiun dari nama file
    nama_stasiun = os.path.basename(file).replace('.xlsx', '').replace('Stasiun_', '')

    df = pd.read_excel(file, header=7)
    df = df[kolom].copy()

    # Tambahkan kolom STASIUN
    df.insert(0, 'STASIUN', nama_stasiun)

    list_df.append(df)
    print(f" {nama_stasiun}: {len(df)} baris")

# Gabungkan semua stasiun
df_gabung = pd.concat(list_df, ignore_index=True)
print(f"\nTotal gabungan: {len(df_gabung)} baris dari {len(semua_file)} stasiun")

# 2. CLEANING — HANDLING ERROR VALUES
df_gabung[kolom_numerik] = df_gabung[kolom_numerik].replace(nilai_error, np.nan)

print("\nJumlah NaN per kolom setelah replace error:")
print(df_gabung[kolom_numerik].isnull().sum())

# Isi NaN dengan rata-rata PER STASIUN (lebih akurat)
df_gabung[kolom_numerik] = df_gabung.groupby('STASIUN')[kolom_numerik].transform(
    lambda x: x.fillna(x.mean())
)

# Isi sisa NaN (kalau satu kolom stasiun kosong semua) dengan rata-rata global
df_gabung[kolom_numerik] = df_gabung[kolom_numerik].fillna(df_gabung[kolom_numerik].mean())

# 3. FORMAT TANGGAL
df_gabung['TANGGAL'] = pd.to_datetime(df_gabung['TANGGAL'], dayfirst=True).dt.date

# Menyimpan hasil gabungan dan bersih ke file baru
output_path = 'env/Cuaca/Cuaca_Gabungan_Bersih.xlsx'
df_gabung.to_excel(output_path, index=False)

print(f"\n✅ File berhasil disimpan: {output_path}")
print(f"\nPreview data gabungan:")
print(df_gabung.head(10))
print(f"\nDaftar stasiun: {df_gabung['STASIUN'].unique().tolist()}")

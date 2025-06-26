  # <h1 align="center">Laporan Tugas Besar - STRATEGI ALGORITMA</h1>
<p align="center">Rafaldo Al Maqdis - 2311102099</p>
<p align="center">Muhammad Nasrullah - 2311102044</p>


## Study Case
```python
!pip install --upgrade geopandas osmnx xgboost shapely matplotlib seaborn scikit-learn imbalanced-learn > /dev/null
```
```python
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
import seaborn as sns

from shapely.geometry import box
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
```
```python
warnings.filterwarnings('ignore')
ox.settings.log_console = False
plt.style.use('ggplot')
sns.set_palette("Set2")
pd.set_option('display.precision', 2)
```
```python
ox.settings.use_cache = True
ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api/interpreter"
ox.settings.timeout = 300
```
```python
KECAMATAN_TERPILIH = 
[    "Semarang Tengah",
    "Semarang Timur",
    "Semarang Barat"]
def buat_grid(batas: gpd.GeoDataFrame, ukuran_grid: int = 200) -> gpd.GeoDataFrame:
    """
    Membuat grid polygon dalam area batas administratif

    Parameters:
        batas (gpd.GeoDataFrame): Batas wilayah kecamatan
        ukuran_grid (int): Ukuran grid dalam meter (default: 200m)

    Returns:
        gpd.GeoDataFrame: Grid yang sudah dipotong dengan batas wilayah
    """
    batas_proj = batas.to_crs(epsg=3857)
    minx, miny, maxx, maxy = batas_proj.total_bounds
    sel_grid = [
        box(x0, y0, x0 + ukuran_grid, y0 + ukuran_grid)
        for x0 in range(int(minx), int(maxx), ukuran_grid)
        for y0 in range(int(miny), int(maxy), ukuran_grid)
    ]
    grid = gpd.GeoDataFrame({'geometry': sel_grid}, crs=batas_proj.crs)
    return gpd.overlay(grid, batas_proj, how='intersection').to_crs(batas.crs)
```
```python
def bersihkan_fitur_dalam_batas(gdf: gpd.GeoDataFrame, batas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Memotong fitur geografis sesuai batas wilayah

    Parameters:
        gdf (gpd.GeoDataFrame): Fitur geografis (jalan, bangunan, dll)
        batas (gpd.GeoDataFrame): Batas wilayah kecamatan

    Returns:
        gpd.GeoDataFrame: Fitur yang sudah dipotong dan dibersihkan
    """
    mask = batas.geometry.union_all()
    return gdf.clip(mask).explode(ignore_index=True)
def dapatkan_data_osm(kecamatan: str) -> tuple:
    """
    Mengambil data OSM dan menyimpan cache

    Parameters:
        kecamatan (str): Nama kecamatan lengkap dengan wilayah administrasi

    Returns:
        tuple: (batas, jalan, bangunan, area_hijau)
    """
    folder_cache = "cached_osm"
    nama_file = kecamatan.split(",")[0].replace(" ", "_").lower()
    path_file = os.path.join(folder_cache, f"{nama_file}_osm.pkl")

    if os.path.exists(path_file):
        with open(path_file, "rb") as f:
            print(f"[INFO] Menggunakan data cache untuk {kecamatan}")
            return pickle.load(f)

    print(f"[PROSES] Mengambil data OSM: {kecamatan}")
    try:
        batas = ox.geocode_to_gdf(kecamatan)

        batas_buffered = batas.to_crs(epsg=3857).buffer(500).to_crs(batas.crs)

        graph_jalan = ox.graph_from_polygon(batas_buffered.geometry.iloc[0], network_type='drive')
        jalan = ox.graph_to_gdfs(graph_jalan, nodes=False, edges=True)

        bangunan = ox.features_from_place(kecamatan, tags={'building': True})

        tag_hijau = {
            'leisure': ['park', 'garden'],
            'landuse': ['grass', 'forest', 'farmland', 'meadow', 'orchard', 'cemetery'],
            'natural': ['wood', 'scrub', 'wetland', 'heath']
        }
        area_hijau = ox.features_from_polygon(batas.geometry.iloc[0], tag_hijau)

        jalan_bersih = bersihkan_fitur_dalam_batas(jalan, batas)
        bangunan_bersih = bersihkan_fitur_dalam_batas(bangunan, batas)
        hijau_bersih = bersihkan_fitur_dalam_batas(area_hijau, batas)

        os.makedirs(folder_cache, exist_ok=True)
        with open(path_file, "wb") as f:
            pickle.dump((batas, jalan_bersih, bangunan_bersih, hijau_bersih), f)

        return batas, jalan_bersih, bangunan_bersih, hijau_bersih

    except Exception as e:
        print(f"[WARNING] Coba alternatif query untuk {kecamatan}")
        try:
            nama_sederhana = kecamatan.split(",")[0].strip()
            batas = ox.geocode_to_gdf(f"{nama_sederhana}, Indonesia")

            return batas, jalan_bersih, bangunan_bersih, hijau_bersih
        except:
            print(f"[ERROR] Tetap gagal, skip kecamatan {kecamatan}")
            return None, None, None, None
def ekstrak_fitur_per_grid(grid: gpd.GeoDataFrame, jalan: gpd.GeoDataFrame, bangunan: gpd.GeoDataFrame, hijau: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ekstraksi fitur untuk setiap grid

    Parameters:
        grid (gpd.GeoDataFrame): Grid wilayah
        jalan (gpd.GeoDataFrame): Data jalan
        bangunan (gpd.GeoDataFrame): Data bangunan
        hijau (gpd.GeoDataFrame): Data area hijau

    Returns:
        gpd.GeoDataFrame: DataFrame dengan fitur yang diekstrak
    """
    jalan = jalan.to_crs(epsg=3857)
    bangunan = bangunan.to_crs(epsg=3857)
    hijau = hijau.to_crs(epsg=3857)
    grid = grid.to_crs(epsg=3857)

    fitur = []
    for idx, sel in grid.iterrows():
        luas_km2 = sel.geometry.area / 1e6
        if luas_km2 < 0.001:
            continue

        jalan_dalam = jalan[jalan.intersects(sel.geometry)]
        panjang_jalan_km = jalan_dalam.geometry.length.sum() / 1000

        bangunan_dalam = bangunan[bangunan.intersects(sel.geometry)]
        jumlah_bangunan = len(bangunan_dalam)

        hijau_dalam = hijau[hijau.intersects(sel.geometry)]
        luas_hijau_km2 = hijau_dalam.geometry.area.sum() / 1e6

        fitur.append({
            'grid_id': idx,
            'luas_km2': luas_km2,
            'panjang_jalan_km': panjang_jalan_km,
            'jumlah_bangunan': jumlah_bangunan,
            'hijau_km2': luas_hijau_km2,
            'kepadatan_jalan': panjang_jalan_km / luas_km2,
            'kepadatan_bangunan': jumlah_bangunan / luas_km2,
            'persentase_hijau': (luas_hijau_km2 / luas_km2) * 100,
            'geometry': sel.geometry
        })

    return gpd.GeoDataFrame(fitur, crs=grid.crs).to_crs(epsg=4326)
```
```python
semua_grid = []
for kec in KECAMATAN_TERPILIH:
    try:
        print(f"\n[PROSES] Memproses kecamatan: {kec.split(',')[0]}")

        batas, jalan, bangunan, hijau = dapatkan_data_osm(kec)

        grid = buat_grid(batas, ukuran_grid=100)

        fig, ax = plt.subplots(figsize=(10, 10))
        hijau.plot(ax=ax, color="green", alpha=0.4, label="Area Hijau")
        bangunan.plot(ax=ax, color="blue", alpha=0.5, label="Bangunan")
        jalan.plot(ax=ax, color="gray", linewidth=1, label="Jalan")
        grid.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.6)

        plt.title(f"Peta Overlay Fitur OSM - {kec.split(',')[0]}", fontsize=14)
        plt.legend(loc='upper left')
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        df_grid = ekstrak_fitur_per_grid(grid, jalan, bangunan, hijau)
        df_grid['kecamatan'] = kec.split(",")[0]
        semua_grid.append(df_grid)
        print(f"[SUKSES] {len(df_grid)} grid diproses")
    except Exception as e:
        print(f"[ERROR] Gagal memproses {kec}: {str(e)}")
```
![image](https://github.com/user-attachments/assets/bc36191e-5302-4d01-8bb2-7c6bf12cbd80)
![image](https://github.com/user-attachments/assets/64f09e2d-a86d-4c9d-be6b-329c895f3d12)
![image](https://github.com/user-attachments/assets/5023a14f-ba57-4f8a-aa06-31285b4f7702)


### Data Training

```python
df_semua_grid = pd.concat(semua_grid, ignore_index=True)
df_semua_grid.dropna(
    subset=['kepadatan_jalan', 'kepadatan_bangunan', 'persentase_hijau'],
    inplace=True
)
print("\n[INFO] Data preprocessing selesai")
print(f"Total grid: {len(df_semua_grid)}")
print(f"Distribusi per kecamatan:")
df_semua_grid['kecamatan'].value_counts()
```
![image](https://github.com/user-attachments/assets/39ae62ac-d3f1-4997-8f67-b485d6f3dba7)

## ANALISIS DISTRIBUSI FITUR NUMERIK

```python
fitur_numerik = ["kepadatan_jalan", "kepadatan_bangunan", "persentase_hijau"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, fitur in enumerate(fitur_numerik):
    sns.boxplot(y=fitur, data=df_semua_grid, ax=axes[i], color='skyblue')
    axes[i].set_title(f'Distribusi {fitur}')

plt.suptitle("Analisis Distribusi Fitur Numerik", fontsize=16)
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/68c1100b-5ca4-48ed-b4bd-9c7636541b0f)
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df_semua_grid['kepadatan_jalan'], kde=True, ax=axes[0], color='blue')
sns.histplot(df_semua_grid['kepadatan_bangunan'], kde=True, ax=axes[1], color='green')
sns.histplot(df_semua_grid['persentase_hijau'], kde=True, ax=axes[2], color='red')
plt.suptitle("Distribusi Fitur Utama", fontsize=16)
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/c4dc4b86-9ced-4110-b7c1-a675015d4157)
```python
ALPHA, BETA, GAMMA = 0.3, 0.5, 0.2
df_semua_grid["emisi_skor"] = (
    ALPHA * df_semua_grid["kepadatan_jalan"]
    + BETA * df_semua_grid["kepadatan_bangunan"]
    - GAMMA * df_semua_grid["persentase_hijau"]
)
def label_emisi(row):
    """Memberi label berdasarkan kriteria emisi"""
    if row['persentase_hijau'] >= 30 and row['kepadatan_jalan'] <= 5 and row['kepadatan_bangunan'] <= 1000:
        return 'Normal'
    elif 15 <= row['persentase_hijau'] < 30 and 5 < row['kepadatan_jalan'] <= 10 and 1000 < row['kepadatan_bangunan'] <= 3000:
        return 'Sedang'
    else:
        return 'Parah'

df_semua_grid["label_emisi"] = df_semua_grid.apply(label_emisi, axis=1)
display(df_semua_grid)
```
![image](https://github.com/user-attachments/assets/fd899c72-2ee5-40cd-8ffb-09500e9770ba)

## KRITERIA EMISI KARBON
```python
df_semua_grid['label_emisi'] = df_semua_grid['label_emisi'].replace({
    'Sedang': 'Tinggi',
    'Parah': 'Tinggi'
})
kriteria_emisi = pd.DataFrame({
    "Kategori": ["Normal (Rendah)", "Tinggi (Gabungan Sedang+Parah)"],
    "% Hijau": [">= 30%", "< 30%"],
    "Kepadatan Jalan": ["<= 5 km/km²", "> 5 km/km²"],
    "Kepadatan Bangunan": ["<= 1000 bangunan/km²", "> 1000 bangunan/km²"],
    "Keterangan": [
        "Kota berkelanjutan, banyak RTH, lalu lintas rendah",
        "Perlu perbaikan: Area hijau terbatas dan kepadatan tinggi"
    ]
})
display(kriteria_emisi.style.set_caption("Kriteria Emisi Karbon"))
```
![image](https://github.com/user-attachments/assets/b0589ee1-7c71-4d70-a147-e07a3236be84)

```python
# Iteratif
def cari_nama_iteratif(nama, karyawan):
    for k in karyawan:  # Loop melalui setiap elemen dalam daftar karyawan
        if k.nama == nama:  # Jika nama ditemukan
            return hitung_gaji_manual(k.jabatan, 1)  # Hitung gaji dan kembalikan hasil
    return -1  # Jika nama tidak ditemukan, kembalikan -1
```

### 3. Algoritma Cetak Tabel

```python
def print_comparison_table(data_sizes, iter_times, recur_times):
    if not data_sizes or not iter_times or not recur_times:
        print("Data tidak tersedia untuk ditampilkan. Jalankan simulasi terlebih dahulu.")
        return

    print("\nTabel Perbandingan Waktu Eksekusi:")
    print("------------------------------------------------------------")
    print(f"{'Jumlah Data':>10} | {'Waktu Iteratif (s)':>20} | {'Waktu Rekursif (s)':>20}")
    print("------------------------------------------------------------")

    for i in range(len(data_sizes)):
        print(f"{data_sizes[i]:>10} | {iter_times[i]:>20.8f} | {recur_times[i]:>20.8f}")
    print("------------------------------------------------------------")

```

### 4. Algoritma Cetak Grafik
```python
def plot_graph(data_sizes, iter_times, recur_times):
    plt.figure(figsize=(10, 5))
    plt.plot(data_sizes, iter_times, label="Iteratif")
    plt.plot(data_sizes, recur_times, label="Rekursif")
    plt.title("Perbandingan Waktu Iteratif vs Rekursif")
    plt.xlabel("Jumlah Karyawan")
    plt.ylabel("Waktu Eksekusi (detik)")
    plt.legend()
    plt.grid()
    plt.savefig("execution_time_comparison.png")
    plt.show()------------------------")

```

### 5. Program Lengkap

```python
import time
import sys
import matplotlib.pyplot as plt

# Set batas rekursi lebih tinggi
sys.setrecursionlimit(200000)

# Definisi class Karyawan
class Karyawan:
    def __init__(self, nama, jabatan):
        self.nama = nama
        self.jabatan = jabatan

# Fungsi untuk menghitung gaji manual
def hitung_gaji_manual(jabatan, faktor):
    if jabatan == "Manager":
        return 10000000 * faktor
    elif jabatan == "Staff":
        return 5000000 * faktor
    elif jabatan == "Engineer":
        return 8000000 * faktor
    else:
        return 3000000 * faktor

# Iteratif
def cari_nama_iteratif(nama, karyawan):
    for k in karyawan:
        if k.nama == nama:
            return hitung_gaji_manual(k.jabatan, 1)
    return -1

# Rekursif
def cari_nama_rekursif(nama, karyawan, index):
    if index >= len(karyawan):
        return -1
    if karyawan[index].nama == nama:
        return hitung_gaji_manual(karyawan[index].jabatan, 1)
    return cari_nama_rekursif(nama, karyawan, index + 1)

def cari_nama_rekursif_wrapper(nama, karyawan):
    return cari_nama_rekursif(nama, karyawan, 0)

# Fungsi untuk mengukur waktu eksekusi
def ukur_waktu(fungsi, *args, ulang=5):
    waktu = []
    for _ in range(ulang):
        start = time.perf_counter()
        fungsi(*args)
        waktu.append(time.perf_counter() - start)
    return sum(waktu) / ulang

# Fungsi untuk membuat grafik
def plot_graph(data_sizes, iter_times, recur_times):
    plt.figure(figsize=(10, 5))
    plt.plot(data_sizes, iter_times, label="Iteratif")
    plt.plot(data_sizes, recur_times, label="Rekursif")
    plt.title("Perbandingan Waktu Iteratif vs Rekursif")
    plt.xlabel("Jumlah Karyawan")
    plt.ylabel("Waktu Eksekusi (detik)")
    plt.legend()
    plt.grid()
    plt.savefig("execution_time_comparison.png")
    plt.show()

# Fungsi untuk membuat tabel perbandingan
def print_comparison_table(data_sizes, iter_times, recur_times):
    if not data_sizes or not iter_times or not recur_times:
        print("Data tidak tersedia untuk ditampilkan. Jalankan simulasi terlebih dahulu.")
        return

    print("\nTabel Perbandingan Waktu Eksekusi:")
    print("------------------------------------------------------------")
    print(f"{'Jumlah Data':>10} | {'Waktu Iteratif (s)':>20} | {'Waktu Rekursif (s)':>20}")
    print("------------------------------------------------------------")

    for i in range(len(data_sizes)):
        print(f"{data_sizes[i]:>10} | {iter_times[i]:>20.8f} | {recur_times[i]:>20.8f}")
    print("------------------------------------------------------------")

# Fungsi untuk menampilkan hasil pencarian dalam bentuk tabel
def print_search_table(nama, hasil_iter, durasi_iter, hasil_recur, durasi_recur):
    print("\nHasil Pencarian:")
    print("------------------------------------------------------------")
    print(f"{'Metode':<10} | {'Hasil Gaji':>10} | {'Waktu Eksekusi (s)':>20}")
    print("------------------------------------------------------------")
    print(f"{'Iteratif':<10} | {hasil_iter:>10} | {durasi_iter:>20.8f}")
    print(f"{'Rekursif':<10} | {hasil_recur:>10} | {durasi_recur:>20.8f}")
    print("------------------------------------------------------------")

# Fungsi utama
def main():
    # Dataset karyawan dengan jumlah besar
    karyawan = []
    for i in range(1500000):
        jabatan = "Staff"
        if i % 3 == 0:
            jabatan = "Manager"
        elif i % 3 == 1:
            jabatan = "Engineer"
        karyawan.append(Karyawan(f"Nama{i+1}", jabatan))

    iter_times = []
    recur_times = []
    data_sizes = []

    while True:
        print("\nMenu:")
        print("1. Tampilkan Data Karyawan (Hanya 10 Data Pertama)")
        print("2. Cari Karyawan (Perbandingan Iteratif vs Rekursif)")
        print("3. Bandingkan Waktu dengan Grafik")
        print("4. Tampilkan Tabel Perbandingan dan Buat Grafik")
        print("5. Keluar")
        pilihan = int(input("Pilihan: "))

        if pilihan == 1:
            print("10 Data Karyawan Pertama:")
            for i in range(10):
                print(f"Nama: {karyawan[i].nama}, Jabatan: {karyawan[i].jabatan}")
        elif pilihan == 2:
            nama = input("Masukkan nama karyawan yang dicari: ")

            # Hitung waktu eksekusi
            durasi_iter = ukur_waktu(cari_nama_iteratif, nama, karyawan)
            durasi_recur = ukur_waktu(cari_nama_rekursif_wrapper, nama, karyawan)

            # Cari gaji
            hasil_iter = cari_nama_iteratif(nama, karyawan)
            hasil_recur = cari_nama_rekursif_wrapper(nama, karyawan)

            # Tampilkan hasil pencarian dalam bentuk tabel
            if hasil_iter == -1 and hasil_recur == -1:
                print("Karyawan tidak ditemukan.")
            else:
                print_search_table(nama, hasil_iter, durasi_iter, hasil_recur, durasi_recur)

        elif pilihan == 3:
            # Simulasi waktu eksekusi dengan subset data
            for n in range(100000, 1500001, 100000):
                subset = karyawan[:n]

                iter_times.append(ukur_waktu(cari_nama_iteratif, "Nama1000", subset))
                recur_times.append(ukur_waktu(cari_nama_rekursif_wrapper, "Nama1000", subset))
                data_sizes.append(n)

            plot_graph(data_sizes, iter_times, recur_times)
            print("Grafik telah disimpan sebagai 'execution_time_comparison.png'.")
        elif pilihan == 4:
            if not data_sizes or not iter_times or not recur_times:
                print("Data tidak tersedia untuk ditampilkan. Jalankan simulasi terlebih dahulu.")
            else:
                try:
                    max_data = int(input("Masukkan jumlah data yang ingin ditampilkan: "))
                    if max_data <= 0:
                        print("Jumlah data harus lebih dari 0.")
                    else:
                        # Ambil subset data sesuai jumlah yang diminta
                        limited_data_sizes = data_sizes[:max_data]
                        limited_iter_times = iter_times[:max_data]
                        limited_recur_times = recur_times[:max_data]
                        
                        # Cetak tabel dengan data yang dibatasi
                        print_comparison_table(limited_data_sizes, limited_iter_times, limited_recur_times)
                        
                        # Buat grafik dengan subset data
                        plot_graph(limited_data_sizes, limited_iter_times, limited_recur_times)
                except ValueError:
                    print("Input tidak valid. Masukkan angka untuk jumlah data.")
        elif pilihan == 5:
            print("Keluar program.")
            break
        else:
            print("Pilihan tidak valid.")

if __name__ == "__main__":
    main()

```


### Output Tabel dan Grafik Berdasarkan inputan n

![Screenshot 2024-12-22 194415](https://github.com/user-attachments/assets/33e626b7-2ac9-41c1-a17a-c6fea6eea017) <br/>

![n_2](https://github.com/user-attachments/assets/712607ab-4cb2-4906-b7ad-e4c90198b71b) <br/>

![Screenshot 2024-12-22 194644](https://github.com/user-attachments/assets/1e100f19-f67e-444f-8afc-fb915a674483) <br/>

![n_5](https://github.com/user-attachments/assets/d072b3d7-c5d7-43e2-a49e-07a3dd7597cf) <br/>

![Screenshot 2024-12-22 194931](https://github.com/user-attachments/assets/07a1260d-393b-4abf-b0bb-1aa0f39e27a4)

![n_8](https://github.com/user-attachments/assets/c12c1127-a414-4668-997d-55d4352ac9d4)

![Screenshot 2024-12-22 195120](https://github.com/user-attachments/assets/e5c9f50f-4694-4388-b6dd-22fdc446f665)

![n_10](https://github.com/user-attachments/assets/fd6ac46c-a8bc-4647-9897-10f52527b048)





## Kesimpulan

- Metode Iteratif lebih efisien dan lebih cepat dalam hal waktu eksekusi, terutama untuk dataset besar. Oleh karena itu, jika Anda bekerja dengan jumlah data yang sangat besar, metode iteratif adalah pilihan yang lebih baik. <br/>

- Metode Rekursif memiliki keuntungan dalam hal kejelasan kode dan kesederhanaan pemrograman, namun dapat mengalami masalah dengan skala dan memori ketika ukuran dataset sangat besar.






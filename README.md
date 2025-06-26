  # <h1 align="center">Laporan Tugas Besar - STRATEGI ALGORITMA</h1>
<p align="center">Rafaldo Al Maqdis - 2311102099</p>
<p align="center">Muhammad Nasrullah - 2311102044</p>


## Study Case
```python
!pip install --upgrade geopandas osmnx xgboost shapely matplotlib seaborn scikit-learn imbalanced-learn > /dev/null
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


## 2. Library Yang Digunakan

### Berikut adalah library yang digunakan di dalam program

```python
import time
import sys
import matplotlib.pyplot as plt


```

## 3. Algoritma Rekursif

```python
# Rekursif
def cari_nama_rekursif(nama, karyawan, index):
    if index >= len(karyawan):  # Basis kasus: jika indeks melebihi panjang daftar
        return -1
    if karyawan[index].nama == nama:  # Basis kasus: jika nama ditemukan
        return hitung_gaji_manual(karyawan[index].jabatan, 1)
    # Rekursi: panggil fungsi untuk elemen berikutnya
    return cari_nama_rekursif(nama, karyawan, index + 1)

def cari_nama_rekursif_wrapper(nama, karyawan):
    return cari_nama_rekursif(nama, karyawan, 0)  # Wrapper untuk memulai dari indeks 0


```    
## 4. Algoritma Iteratif

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






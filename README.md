# ANALISIS PRAKTIKUM FACENET

## build_embeddings.py 
### Analisis

Script build_embeddings.py berfungsi membaca dataset wajah dari folder data/train kemudian mendeteksi wajah menggunakan MTCNN. Setelah itu wajah di-resize, lalu diekstraksi menjadi embedding menggunakan model FaceNet (InceptionResnetV1 pretrained on VGGFace2).
Output akhirnya adalah file embeddings.npz yang berisi:

embedding wajah
label (nama folder)
jumlah data
dimensi embedding
Script ini sangat penting sebagai fondasi karena embedding inilah yang digunakan nanti untuk pelatihan KNN atau proses pengenalan wajah.
## convert_npz_to_npy.py 
### Analisis

Script convert_npz_to_npy.py memuat file embeddings.npz kemudian memisahkannya menjadi:
X_train.npy → berisi embedding wajah
y_train.npy → berisi label individu
Tujuannya agar lebih mudah dipakai pada proses training, misalnya untuk algoritma KNN, SVM, atau classifier lainnya.
Proses ini memastikan data berhasil dipisahkan dan tersimpan dengan benar.

## evaluate.py
### Analisis

Script evaluate.py digunakan untuk melakukan evaluasi terhadap model KNN atau classifier lain berdasarkan data validasi.
Langkah yang dilakukan:
Load model KNN yang sudah dilatih
Load dataset validasi dari folder data/val
Deteksi wajah dengan MTCNN
Ekstraksi embedding dengan FaceNet
Prediksi label menggunakan model KNN
Hitung akurasi
Script ini bertujuan mengetahui seberapa baik model dalam mengenali wajah.

## facenet_knn.joblib 
### Analisis

File .joblib adalah hasil penyimpanan model KNN yang sudah dilatih.
Dengan file ini, kamu tidak perlu melatih ulang setiap menjalankan aplikasi — cukup load model dan langsung lakukan prediksi.

## predict_knn.py 
### Analisis

Script predict_knn.py digunakan untuk menguji satu gambar wajah secara langsung. Prosesnya:
Deteksi wajah dengan MTCNN
Ekstraksi embedding dengan FaceNet
Load model KNN dari file .joblib
Lakukan prediksi label dan tampilkan hasilnya
Script ini berguna untuk pengujian cepat (single-image testing).

## svm_model.pkl 
### Analisis

File .pkl ini merupakan model SVM alternatif yang kamu latih selain KNN.
Biasanya digunakan sebagai pembanding — apakah KNN lebih baik atau SVM lebih akurat pada datamu.

## train_classifier.py 
### Analisis

Script ini melatih classifier (SVM atau KNN) dengan data embedding yang sudah disiapkan.
Prosesnya:
Load X_train.npy dan y_train.npy
Memilih jenis classifier (misalnya SVM)
Melatih model menggunakan embedding wajah
Menyimpan model ke file .pkl atau .joblib
Script ini penting karena menjadi tahap training model utama.

## train_knn.py 
### Analisis

Script train_knn.py khusus melatih algoritma KNN menggunakan embedding wajah dari X_train.npy dan y_train.npy.
Keluaran akhirnya adalah model facenet_knn.joblib.
KNN dipilih karena:
sederhana
cocok digunakan bersama embedding FaceNet
hasilnya cukup akurat untuk dataset kecil

## verify_pair.py 
### Analisis

Script ini digunakan untuk face verification, yaitu membandingkan dua gambar apakah mereka adalah orang yang sama.
Langkahnya:
Deteksi wajah dengan MTCNN
Proses embedding dengan FaceNet
Hitung cosine similarity
Tentukan apakah "match" berdasarkan threshold (misal 0.85)
Script ini penting untuk tugas verifikasi wajah, berbeda dengan face recognition.

## utils_facenet.py 
### Analisis

File penting yang berisi fungsi-fungsi pendukung:
deteksi wajah (MTCNN)
alignment wajah
ekstraksi embedding (FaceNet)
fungsi similarity
fungsi normalisasi
Semua script lain bergantung pada file utilitas ini agar proses face recognition berjalan dengan benar.


Bounding box + nama muncul di layar

Ini memungkinkan real-time face recognition.

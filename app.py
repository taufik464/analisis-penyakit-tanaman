import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

# ==============================================================================
# 1. KONFIGURASI DAN PEMUATAN DATA
# ==============================================================================

FILE_PATH = 'dataset_penyakit_tanaman.csv'

# Kolom Kriteria (termasuk Nama Tanaman)
KRITERIA_COLUMNS = ['Nama_Tanaman', 'Warna_Daun', 'Bercak_Daun', 'Daun_Layu', 'Batang_Busuk', 'Pertumbuhan_Terhambat']
GEJALA_COLUMNS = KRITERIA_COLUMNS[1:] # Gejala selain Nama Tanaman
HASIL_COLUMN = 'Penyakit'

# Bobot untuk setiap kriteria (total harus 1.0)
# [Nama_Tanaman, Warna_Daun, Bercak_Daun, Daun_Layu, Batang_Busuk, Pertumbuhan_Terhambat]
WEIGHTS = np.array([0.30, 0.10, 0.15, 0.15, 0.15, 0.15]) 
K_NEIGHBORS = 3

@st.cache_data
def load_and_process_cases(file_path):
    """Memuat data dari CSV, melakukan encoding, dan mengubahnya menjadi format basis kasus."""
    try:
        df = pd.read_csv(file_path)
        # Mengganti Nama_Tai menjadi Nama_Tanaman
        if 'Nama_Tai' in df.columns:
            df = df.rename(columns={'Nama_Tai': 'Nama_Tanaman'})
    except FileNotFoundError:
        st.error(f"File CSV tidak ditemukan: {file_path}. Pastikan file 'a.csv' berada di direktori yang sama.")
        return None, None, None, None, None

    feature_mappings = {}
    
    # 1. Encoding semua kolom kriteria
    for col in KRITERIA_COLUMNS:
        unique_values = df[col].unique()
        # Membuat dictionary mapping: {nilai_unik: index_numerik}
        # Nilai numerik dimulai dari 1
        feature_mappings[col] = {val: i + 1 for i, val in enumerate(unique_values)}
    
    # 2. Encoding data (mengubah nilai teks menjadi numerik)
    df_encoded = df.copy()
    for col, mapping in feature_mappings.items():
        df_encoded[col] = df_encoded[col].map(mapping)
        
    # 3. Membuat Basis Kasus Dictionary & Data Training untuk k-NN
    cases_dict = {}
    X_train = []
    y_train = [] 

    for index, row in df_encoded.iterrows():
        case_id = f"K{int(row['ID']):03d}" 
        kriteria_list = [row[col] for col in KRITERIA_COLUMNS]
        
        cases_dict[case_id] = {
            'tanaman': row['Nama_Tanaman'], 
            'kriteria_numeric': [int(g) for g in kriteria_list],
            'diagnosis': row[HASIL_COLUMN],
            'solusi': f"Penanganan untuk penyakit: {row[HASIL_COLUMN]}. (Kasus ID: {case_id})" 
        }
        X_train.append(cases_dict[case_id]['kriteria_numeric'])
        y_train.append(row[HASIL_COLUMN])

    return cases_dict, feature_mappings, df['Nama_Tanaman'].unique(), np.array(X_train), np.array(y_train)

CASES, FEATURE_MAPPINGS, TANAMAN_LIST, X_TRAIN, Y_TRAIN = load_and_process_cases(FILE_PATH)

if CASES is None:
    st.stop() 

# ==============================================================================
# 2. FUNGSI PERHITUNGAN CBR & K-NN
# ==============================================================================

def calculate_similarity(case_new_kriteria, case_old_kriteria, weights):
    """Menghitung kemiripan menggunakan Weighted Similarity (CBR)."""
    kriteria_new = np.array(case_new_kriteria)
    kriteria_old = np.array(case_old_kriteria)
    match = (kriteria_new == kriteria_old).astype(int)
    similarity_score = np.sum(match * weights)
    return similarity_score

def cbr_diagnosis(new_case_kriteria, weights, cases):
    """Melakukan diagnosis dengan mencari kasus paling mirip."""
    similarity_scores = {}
    for case_id, data in cases.items():
        score = calculate_similarity(new_case_kriteria, data['kriteria_numeric'], weights)
        similarity_scores[case_id] = score
    sorted_scores = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
    best_match_id, best_score = sorted_scores[0]
    best_case = cases[best_match_id]
    return best_case, best_score, sorted_scores

def knn_diagnosis(new_case_kriteria, X_train, Y_train, k, weights):
    """Melakukan diagnosis menggunakan k-Nearest Neighbors (Weighted Euclidean Distance)."""
    new_case = np.array(new_case_kriteria)
    distances = []

    for i, train_case in enumerate(X_train):
        # Weighted Euclidean Distance (karena data kategorikal)
        diff_sq = (new_case - train_case) ** 2
        weighted_diff_sq = diff_sq * weights
        distance = np.sqrt(np.sum(weighted_diff_sq))
        distances.append({'distance': distance, 'label': Y_train[i], 'case_id': f"K{i+1:03d}"})

    # Urutkan berdasarkan jarak terdekat (jarak terkecil)
    distances.sort(key=lambda x: x['distance'])
    
    # Ambil k tetangga terdekat
    neighbors = distances[:k]
    
    # Hitung kelas prediksi (Voting)
    neighbor_labels = [n['label'] for n in neighbors]
    most_common = Counter(neighbor_labels).most_common(1)
    
    predicted_diagnosis = most_common[0][0]
    
    return predicted_diagnosis, neighbors

# ==============================================================================
# 3. ANTARMUKA STREAMLIT
# ==============================================================================

st.set_page_config(page_title="SPK CBR & k-NN Hibrida", layout="wide")

st.title("🌱 SPK Diagnosa Penyakit Tanaman (CBR & k-NN)")
st.markdown("Diagnosa menggunakan dua metode berdasarkan kriteria berbobot.")

st.header("1. Input Data Tanaman")

with st.form("spk_form"):
    
    # --- Input Nama Tanaman (TEXT INPUT MANUAL) ---
    nama_tanaman_input = st.text_input(
        "Nama Tanaman", 
        placeholder="Contoh: Padi, Stroberi, atau nama tanaman baru"
    )
    st.caption("Nama Tanaman wajib diisi dan mempengaruhi perhitungan.")
    st.markdown("---")
    
    # --- Input Dropdown (SelectBox) untuk Gejala ---
    input_values = {}
    st.subheader("Gejala yang Diamati")
    
    for col in GEJALA_COLUMNS:
        options = list(FEATURE_MAPPINGS[col].keys())
        input_values[col] = st.selectbox(
            f"{GEJALA_COLUMNS.index(col)+1}. {col.replace('_', ' ')}", 
            options
        )
    
    submitted = st.form_submit_button("Diagnosa Penyakit")

if submitted:
    
    if not nama_tanaman_input.strip():
        st.error("Nama Tanaman wajib diisi!")
        st.stop()
    
    # --- Pemetaan Kriteria Input ke Nilai Numerik ---
    new_case_kriteria = []
    
    # 1. Pemetaan Nama Tanaman
    tanaman_mapping = FEATURE_MAPPINGS['Nama_Tanaman']
    
    # Normalisasi/Pencocokan nama tanaman (Case-insensitive)
    input_tanaman_norm = nama_tanaman_input.strip()
    
    # Cari kecocokan di mapping
    match_found = False
    for key, val in tanaman_mapping.items():
        if str(key).strip().lower() == input_tanaman_norm.lower():
            numeric_tanaman = val
            match_found = True
            break
    
    if not match_found:
        # Jika input manual TIDAK ADA di data, beri nilai unik tertinggi + 1
        numeric_tanaman = max(tanaman_mapping.values()) + 1
        st.warning(f"Nama Tanaman '{input_tanaman_norm}' tidak ditemukan di basis kasus. Nilai unik **{numeric_tanaman}** digunakan.")

    new_case_kriteria.append(numeric_tanaman)
    
    # 2. Pemetaan Gejala (dari SelectBox)
    for col in GEJALA_COLUMNS:
        selected_text = input_values[col]
        numeric_val = FEATURE_MAPPINGS[col][selected_text]
        new_case_kriteria.append(numeric_val)
    
    # ==========================================================================
    # 4. HASIL DIAGNOSIS
    # ==========================================================================

    # --- CBR Diagnosis ---
    best_case, best_score, sorted_scores = cbr_diagnosis(new_case_kriteria, WEIGHTS, CASES)
    
    # --- k-NN Diagnosis ---
    knn_diagnosis_result, neighbors = knn_diagnosis(new_case_kriteria, X_TRAIN, Y_TRAIN, K_NEIGHBORS, WEIGHTS)

    st.markdown("---")
    st.header("2. Hasil Diagnosis SPK Hibrida")

    col_cbr, col_knn = st.columns(2)
    
    # Tampilan CBR
    with col_cbr:
        st.subheader("🔴 Case-Based Reasoning (CBR)")
        st.metric(label="Tingkat Kemiripan (Similarity Score)", value=f"{best_score*100:.2f}%")
        st.info(f"**Diagnosis CBR:** **{best_case['diagnosis']}**")
        st.caption(f"Kasus Rujukan: **{sorted_scores[0][0]} ({best_case['tanaman']})**.")

    # Tampilan k-NN
    with col_knn:
        st.subheader(f"🔵 k-Nearest Neighbors (k-NN, k={K_NEIGHBORS})")
        st.success(f"**Diagnosis k-NN:** **{knn_diagnosis_result}**")
        
        # Detail k=3 Tetangga Terdekat
        st.markdown("**Detail 3 Tetangga Terdekat:**")
        knn_df = pd.DataFrame([
            {'ID': n['case_id'], 'Penyakit': n['label'], 'Jarak': f"{n['distance']:.4f}"}
            for n in neighbors
        ])
        st.dataframe(knn_df, hide_index=True)
        
    st.markdown("---")
    
    st.subheader(f"Rekomendasi Penanganan untuk **{input_tanaman_norm.capitalize()}**")
    st.success(f"**Solusi (Berdasarkan kasus CBR terdekat):** {best_case['solusi']}")
    
    st.markdown("---")
    
    st.subheader("Detail Perhitungan Kemiripan Kasus (CBR)")
    
    df_similarity = pd.DataFrame([
        {
            'Kasus ID': case_id, 
            'Tanaman Kasus': CASES[case_id]['tanaman'], 
            'Kemiripan': f"{score*100:.2f}%", 
            'Diagnosis': CASES[case_id]['diagnosis']
        }
        for case_id, score in sorted_scores
    ])
    
    st.dataframe(df_similarity, use_container_width=True, hide_index=True)
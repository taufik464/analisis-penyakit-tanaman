import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os
import ast

# ======================================================================
# 1. KONFIGURASI DAN PEMUATAN DATA
# ======================================================================

FILE_PATH = 'dataset_penyakit_tanaman.csv'
HISTORY_FILE = 'riwayat_diagnosis.csv'

KRITERIA_COLUMNS = [
    'Nama_Tanaman',
    'Warna_Daun',
    'Bercak_Daun',
    'Daun_Layu',
    'Batang_Busuk',
    'Pertumbuhan_Terhambat'
]
GEJALA_COLUMNS = KRITERIA_COLUMNS[1:]
HASIL_COLUMN = 'Penyakit'

WEIGHTS = np.array([0.30, 0.10, 0.15, 0.15, 0.15, 0.15])


# ==========================================================================
# Fungsi memuat riwayat
# ==========================================================================

def load_history(filename):
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)

            if 'input' in df.columns:
                df['input'] = df['input'].apply(ast.literal_eval)

            if 'kriteria_numeric' in df.columns:
                df['kriteria_numeric'] = df['kriteria_numeric'].apply(ast.literal_eval)

            records = df.to_dict('records')
            max_id = df['id'].max() if not df.empty else 0
            return records, int(max_id)

        except Exception as e:
            st.warning(f"Gagal membaca file riwayat. Error: {e}")
            return [], 0
    return [], 0


def save_history(history_list, filename):
    df = pd.DataFrame(history_list)
    df.to_csv(filename, index=False)


# ======================================================================
# Load dataset awal + encoding
# ======================================================================

@st.cache_data
def load_and_process_cases(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'Nama_Tai' in df.columns:
            df = df.rename(columns={'Nama_Tai': 'Nama_Tanaman'})
    except:
        st.error(f"Dataset '{file_path}' tidak ditemukan!")
        return None, None, None

    if 'ID' not in df.columns:
        df['ID'] = range(1, len(df) + 1)

    feature_mappings = {}
    for col in KRITERIA_COLUMNS:
        unique_vals = df[col].unique()
        feature_mappings[col] = {val: i + 1 for i, val in enumerate(unique_vals)}

    df_encoded = df.copy()
    for col, mapping in feature_mappings.items():
        df_encoded[col] = df_encoded[col].map(mapping)

    cases_dict = {}
    for _, row in df_encoded.iterrows():
        cid = f"K{int(row['ID']):03d}"
        kriteria = [row[col] for col in KRITERIA_COLUMNS]

        cases_dict[cid] = {
            "tanaman": row["Nama_Tanaman"],
            "kriteria_numeric": [int(x) for x in kriteria],
            "diagnosis": row[HASIL_COLUMN],
            "solusi": f"Penanganan untuk penyakit: {row[HASIL_COLUMN]}"
        }
    return cases_dict, feature_mappings, df['Nama_Tanaman'].unique()


# ======================================================================
# Menggabungkan dataset awal + riwayat
# ======================================================================

def get_combined_cases(initial_cases, history_records):
    combined = initial_cases.copy()

    for record in history_records:
        cid = f"R{record['id']:03d}"
        combined[cid] = {
            "tanaman": record["input"]["Nama_Tanaman"],
            "kriteria_numeric": record["kriteria_numeric"],
            "diagnosis": record["diagnosis_revisi"],
            "solusi": record["solusi_revisi"]
        }

    return combined


# ======================================================================
# Inisialisasi data
# ======================================================================

CASES_INITIAL, FEATURE_MAPPINGS, TANAMAN_LIST = load_and_process_cases(FILE_PATH)
if CASES_INITIAL is None:
    st.stop()

if "history" not in st.session_state:
    h, c = load_history(HISTORY_FILE)
    st.session_state.history = h
    st.session_state.history_counter = c

CASES = get_combined_cases(CASES_INITIAL, st.session_state.history)


# ======================================================================
# Perhitungan CBR
# ======================================================================

def calculate_similarity(k1, k2, weights):
    k1 = np.array(k1)
    k2 = np.array(k2)
    match = (k1 == k2).astype(int)
    return float(np.sum(match * weights))


def cbr_diagnosis(new_kriteria, weights, cases):
    sim_scores = {}
    for cid, case in cases.items():
        if len(case["kriteria_numeric"]) == len(new_kriteria):
            score = calculate_similarity(new_kriteria, case["kriteria_numeric"], weights)
            sim_scores[cid] = score

    if not sim_scores:
        return None, 0.0, []

    sorted_scores = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
    best_id, best_score = sorted_scores[0]
    return cases[best_id], best_score, sorted_scores


# ======================================================================
# Simpan kasu baru (Retain)
# ======================================================================

def update_global_cases():
    global CASES
    CASES = get_combined_cases(CASES_INITIAL, st.session_state.history)


def save_diagnosis(input_data, new_case_kriteria, best_case, best_score):
    st.session_state.history_counter += 1

    record = {
        "id": st.session_state.history_counter,
        "tanggal": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": input_data,
        "kriteria_numeric": new_case_kriteria,
        "diagnosis_cbr": best_case["diagnosis"],
        "solusi_cbr": best_case["solusi"],
        "kemiripan": best_score,
        "diagnosis_revisi": best_case["diagnosis"],
        "solusi_revisi": best_case["solusi"],
        "direvisi": False,
    }

    st.session_state.history.append(record)
    save_history(st.session_state.history, HISTORY_FILE)
    update_global_cases()

    st.success(f"Kasus berhasil disimpan sebagai ID R{record['id']:03d}")
    st.rerun()


# ======================================================================
# Revisi diagnosis (Revise)
# ======================================================================

def revise_diagnosis(record_id, diagnosis, solusi):
    for record in st.session_state.history:
        if record["id"] == record_id:
            record["diagnosis_revisi"] = diagnosis
            record["solusi_revisi"] = solusi
            record["direvisi"] = True

    save_history(st.session_state.history, HISTORY_FILE)
    update_global_cases()
    st.success(f"Revisi kasus R{record_id:03d} berhasil disimpan.")
    st.rerun()


# ======================================================================
# ANTARMUKA
# ======================================================================

st.set_page_config(page_title="SPK CBR Penyakit Tanaman", layout="wide")
st.title("🌱 Sistem Pakar Diagnosa Penyakit Tanaman (CBR Learning System)")
tab1, tab2 = st.tabs(["Diagnosa Baru", "Riwayat & Revisi"])


# ======================================================================
# TAB: DIAGNOSA BARU
# ======================================================================

with tab1:
    st.header("Input Data Tanaman")

    with st.form("form_diagnosa"):
        nama_tanaman = st.text_input("Nama Tanaman")

        input_values = {}
        cols = st.columns(3)
        split = np.array_split(GEJALA_COLUMNS, 3)

        for i, part in enumerate(split):
            with cols[i]:
                for col in part:
                    input_values[col] = st.selectbox(col.replace("_", " "), list(FEATURE_MAPPINGS[col].keys()))

        submit = st.form_submit_button("Diagnosa")

    if submit:
        if not nama_tanaman.strip():
            st.error("Nama tanaman tidak boleh kosong")
            st.stop()

        # ----- Pemetaan nama tanaman -----
        nm = nama_tanaman.strip()
        first_map = FEATURE_MAPPINGS["Nama_Tanaman"]

        if nm in first_map:
            numeric = first_map[nm]
        else:
            all_vals = [case["kriteria_numeric"][0] for case in CASES.values()]
            numeric = max(all_vals) + 1

        new_case_kriteria = [numeric]

        input_data = {"Nama_Tanaman": nm}
        for col in GEJALA_COLUMNS:
            input_data[col] = input_values[col]
            new_case_kriteria.append(FEATURE_MAPPINGS[col][input_values[col]])

        best_case, best_score, sorted_scores = cbr_diagnosis(new_case_kriteria, WEIGHTS, CASES)

        st.metric("Similarity", f"{best_score*100:.2f}%")
        st.success(f"Diagnosis: {best_case['diagnosis']}")
        st.info(f"Solusi: {best_case['solusi']}")

        if st.button("Simpan ke Riwayat"):
            save_diagnosis(input_data, new_case_kriteria, best_case, best_score)


# ======================================================================
# TAB: RIWAYAT
# ======================================================================

with tab2:
    st.header("Riwayat Diagnosis")

    if not st.session_state.history:
        st.info("Belum ada riwayat.")
    else:
        df = pd.DataFrame(st.session_state.history)
        df_display = df.copy()
        df_display["ID Kasus"] = df_display["id"].apply(lambda x: f"R{x:03d}")
        df_display["Kemiripan"] = df_display["kemiripan"].apply(lambda x: f"{x*100:.2f}%")

        st.dataframe(df_display, use_container_width=True)

        st.subheader("Revisi Diagnosis")

        pilih = st.selectbox("Pilih ID Riwayat", df["id"].tolist(), format_func=lambda x: f"R{x:03d}")

        selected = next((r for r in st.session_state.history if r["id"] == pilih), None)

        if selected:
            st.write("Input awal:", selected["input"])
            st.write("Diagnosis CBR:", selected["diagnosis_cbr"])

            with st.form("rev_form"):
                new_diag = st.text_input("Diagnosis Revisi", selected["diagnosis_revisi"])
                new_sol = st.text_area("Solusi Revisi", selected["solusi_revisi"])
                ok = st.form_submit_button("Simpan Revisi")

            if ok:
                revise_diagnosis(pilih, new_diag, new_sol)

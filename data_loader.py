import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, "dataset_penyakit_tanaman.csv")
HISTORY_FILE = os.path.join(BASE_DIR, "riwayat_diagnosis.csv")

# ================= LOAD DATASET =================
def load_dataset():
    df = pd.read_csv(DATASET_FILE)

    df["kriteria_numeric"] = df.apply(kriteria_to_numeric, axis=1)
    return df

# ================= KRITERIA =================
def kriteria_to_numeric(row):
    return [
        1 if row["Warna_Daun"] != "Hijau" else 0,
        1 if row["Bercak_Daun"] == "Ada" else 0,
        1 if row["Daun_Layu"] == "Ya" else 0,
        1 if row["Batang_Busuk"] == "Ya" else 0,
        1 if row["Pertumbuhan_Terhambat"] == "Ya" else 0
    ]

# ================= RIWAYAT =================
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE).to_dict("records")
    return []

def save_history(history):
    pd.DataFrame(history).to_csv(HISTORY_FILE, index=False)

# ================= GABUNG KASUS =================
def get_all_cases(dataset, history):
    cases = []

    # kasus dari dataset
    for _, row in dataset.iterrows():
        cases.append({
            "kriteria_numeric": row["kriteria_numeric"],
            "diagnosis": row["Penyakit"]
        })

    # kasus dari riwayat (hasil revisi dipakai ulang)
    for h in history:
        cases.append({
            "kriteria_numeric": [
                1 if h["warna_daun"] != "Hijau" else 0,
                1 if h["bercak_daun"] == "Ada" else 0,
                1 if h["daun_layu"] == "Ya" else 0,
                1 if h["batang_busuk"] == "Ya" else 0,
                1 if h["pertumbuhan_terhambat"] == "Ya" else 0
            ],
            "diagnosis": h["diagnosis"]
        })

    return cases

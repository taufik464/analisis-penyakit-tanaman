import streamlit as st
import pandas as pd
from datetime import datetime
from cbr import retrieve_case
from data_loader import get_all_cases, save_history

def render_ui(dataset):
    st.title("🌱 Sistem Pakar Diagnosis Penyakit Tanaman (CBR)")

    tab_diagnosa, tab_riwayat = st.tabs(["🔍 Diagnosa", "📚 Riwayat"])

    # ================= TAB DIAGNOSA =================
    with tab_diagnosa:
        st.subheader("Input Gejala")

        nama_tanaman = st.text_input("Nama Tanaman")

        warna_daun = st.selectbox("Warna Daun", ["Hijau", "Kuning", "Coklat"])
        bercak_daun = st.selectbox("Bercak Daun", ["Tidak Ada", "Ada"])
        daun_layu = st.selectbox("Daun Layu", ["Tidak", "Ya"])
        batang_busuk = st.selectbox("Batang Busuk", ["Tidak", "Ya"])
        pertumbuhan = st.selectbox("Pertumbuhan Terhambat", ["Tidak", "Ya"])

        if st.button("🔍 Diagnosa"):
            new_case = [
                1 if warna_daun != "Hijau" else 0,
                1 if bercak_daun == "Ada" else 0,
                1 if daun_layu == "Ya" else 0,
                1 if batang_busuk == "Ya" else 0,
                1 if pertumbuhan == "Ya" else 0
            ]

            cases = get_all_cases(dataset, st.session_state.history)
            best_case, score = retrieve_case(new_case, cases)

            st.session_state.last_result = {
                "id": len(st.session_state.history) + 1,
                "tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "nama_tanaman": nama_tanaman,
                "warna_daun": warna_daun,
                "bercak_daun": bercak_daun,
                "daun_layu": daun_layu,
                "batang_busuk": batang_busuk,
                "pertumbuhan_terhambat": pertumbuhan,
                "diagnosis": best_case["diagnosis"],
                "similarity": round(score, 2)
            }

        if "last_result" in st.session_state:
            r = st.session_state.last_result
            st.success(f"Hasil Diagnosis: **{r['diagnosis']}**")
            st.info(f"Tingkat Kemiripan: **{r['similarity']}**")

            if st.button("💾 Simpan ke Riwayat"):
                st.session_state.history.append(r)
                save_history(st.session_state.history)
                del st.session_state.last_result
                st.success("Riwayat disimpan")
                st.rerun()

    # ================= TAB RIWAYAT =================
    with tab_riwayat:
        st.subheader("Riwayat Diagnosis")

        if not st.session_state.history:
            st.info("Belum ada riwayat")
            return

        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

        st.divider()
        st.subheader("✏️ Revisi Diagnosis")

        selected_id = st.selectbox(
            "Pilih ID",
            options=df["id"].tolist()
        )

        new_diag = st.text_input("Diagnosis Revisi")
        catatan = st.text_input("Catatan Revisi")

        if st.button("💾 Simpan Revisi"):
            for h in st.session_state.history:
                if h["id"] == selected_id:
                    h["diagnosis"] = new_diag
                    h["tanggal_revisi"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    h["catatan_revisi"] = catatan or "Revisi oleh pakar"
                    break

            save_history(st.session_state.history)
            st.success("Diagnosis berhasil direvisi")
            st.rerun()

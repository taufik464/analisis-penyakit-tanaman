import streamlit as st
from data_loader import load_dataset, load_history
from ui import render_ui

st.set_page_config(
    page_title="SPK CBR Penyakit Tanaman",
    layout="wide"
)

dataset = load_dataset()

if "history" not in st.session_state:
    st.session_state.history = load_history()

render_ui(dataset)

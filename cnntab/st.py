import streamlit as st  # type: ignore

from .ftr import extract_one_worker_hour
from .heatmap import Heatmap


def display_heatmap(file_path: str):
    st.title("Heatmap")

    st.markdown("## v1")
    worker, label, timestamp, data, diag = extract_one_worker_hour(file_path)
    st.write(f"Employee Number: {worker}")
    st.write(f"Label: {label}")
    st.write(f"WorkDateTime: {timestamp}")
    st.write(diag)
    h = Heatmap(data=data).create_heatmap()
    st.pyplot(h)

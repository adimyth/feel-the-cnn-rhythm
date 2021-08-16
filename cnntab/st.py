import streamlit as st  # type: ignore

from .ftr import extract_one_worker_hour
from .heatmap import Heatmap


def display_heatmap(file_path: str):
    st.title("Heatmap")

    st.markdown("## v1")
    worker, label, timestamp, data, diag = extract_one_worker_hour(file_path)
    st.write(f"Employee Number: {worker.iloc[0]}")
    st.write(f"Label: {label.iloc[0]}")
    st.write(f"WorkDateTime: {timestamp.iloc[0]}")
    st.write(diag)
    h = Heatmap(data=data).create_heatmap()
    st.pyplot(h)

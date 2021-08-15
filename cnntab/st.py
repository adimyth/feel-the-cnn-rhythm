import streamlit as st
from .ftr import FTR, extract_one_worker, df_to_heatmap, df_to_heatmap_v2
from .heatmap import Heatmap


def display_heatmap(file_path: str):
    st.title("Heatmap")

    raw = extract_one_worker(file_path)

    st.markdown("## v1")
    data, label, timestamp, diag = df_to_heatmap(raw)
    st.write(diag)
    st.write(label)
    st.write(data)
    st.write(timestamp)
    h = Heatmap(data=data).create_heatmap()
    st.pyplot(h)

    st.markdown("## v2")
    data2, label2, timestamp2, diag2 = df_to_heatmap_v2(raw)
    st.write(diag2)
    st.write(label2)
    st.write(data2)
    st.write(timestamp2)
    h2 = Heatmap(data=data2).create_heatmap()
    st.pyplot(h2)

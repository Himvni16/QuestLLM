"""Streamlit entry point for QuestLLM."""

import streamlit as st

from questllm.config import APP_DESCRIPTION, APP_NAME

st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="centered")

st.title(APP_NAME)
st.write(APP_DESCRIPTION)
st.info(
    "QuestLLM is being built incrementally. Paste Text, PDF Upload, and quiz generation "
    "will be introduced in later phases."
)

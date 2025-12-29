import streamlit as st
st.title("AI Predictor")

company_name = st.text_input("Enter your AI company name:")
description = st.text_area("Enter a brief description of your AI company:")

if st.button("Analyze"):
    pass
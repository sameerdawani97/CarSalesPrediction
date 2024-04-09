import streamlit as st


def showHome():
    st.markdown("<h1 style='text-align: center;'>Welcome to the Car Sales Price Prediction App!</h1>\n", unsafe_allow_html=True)

    st.image("download.jpeg", use_column_width=True)
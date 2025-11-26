import streamlit as st
import pandas as pd
from utils import render_sidebar_info

# Page configuration
st.set_page_config(
    page_title="IND320 Weather Data App",
    layout="wide"
)

# Render sidebar data info
render_sidebar_info()

# Main page content
st.title("Weather Data Analysis App")
st.header("Welcome to the IND320 Project")

st.markdown("""
### About This App
This is a Streamlit application for analyzing weather data from Open-Meteo.

The app includes:
- **Map & Selector**: Select location and download data
- **Data Exploration**: View raw data and interactive plots
- **Snow Drift**: Analyze snow transport and fence requirements
- **Advanced Analysis**: STL decomposition and Spectrograms
- **Anomalies**: Detect weather anomalies
- **Correlations**: Analyze Weather vs Energy relationships
- **Forecasting**: Predict future energy production
""")

st.markdown("---")

# Quick Navigation
st.subheader("Quick Navigation")

col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    st.markdown("#### 📊 Data Exploration")
    if st.button("📍 Map Selector", use_container_width=True, key="main_map"):
        st.switch_page("pages/01_Map_Selector.py")
    if st.button("📈 Interactive Plot", use_container_width=True, key="main_plot"):
        st.switch_page("pages/02_Interactive_Plot.py")

with col_nav2:
    st.markdown("#### 🔬 Analysis")
    if st.button("❄️ Snow Drift", use_container_width=True, key="main_snow"):
        st.switch_page("pages/03_Snow_Drift.py")
    if st.button("📊 Advanced Analysis", use_container_width=True, key="main_advanced"):
        st.switch_page("pages/04_Advanced_Analysis.py")
    if st.button("⚠️ Weather Anomalies", use_container_width=True, key="main_anomalies"):
        st.switch_page("pages/05_Weather_Anomalies.py")

with col_nav3:
    st.markdown("#### 🔮 Predictive")
    if st.button("🔗 Correlations", use_container_width=True, key="main_corr"):
        st.switch_page("pages/06_Correlations.py")
    if st.button("📉 Forecasting", use_container_width=True, key="main_forecast"):
        st.switch_page("pages/07_Forecasting.py")

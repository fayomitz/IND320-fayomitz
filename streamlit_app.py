import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="IND320 Weather Data App",
    page_icon="☁",
    layout="wide"
)

# Cache the data loading function for performance
@st.cache_data
def load_data():
    """Load the weather data from CSV file with caching for app speed"""
    df = pd.read_csv('open-meteo-subset.csv')
    df['time'] = pd.to_datetime(df['time'])
    return df

# Main page content
st.title("Weather Data Analysis App")
st.header("Welcome to the IND320 Project")

st.markdown("""
### About This App
This is a Streamlit application for analyzing weather data from Open-Meteo.

The app includes:
- **Home** (this page): Overview and introduction
- **Data Table**: View the dataset with interactive visualizations
- **Interactive Plot**: Explore the data with customizable plots
""")

st.markdown("---")

# Display some basic info about the dataset
st.subheader("Dataset Information")
try:
    data = load_data()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    
    with col2:
        st.metric("Date Range", f"{data['time'].min().date()} to {data['time'].max().date()}")
    
    with col3:
        st.metric("Variables", len(data.columns) - 1)
    
except FileNotFoundError:
    st.error("Data file 'open-meteo-subset.csv' not found. Please ensure the file is in the same directory as this app.")

import streamlit as st
import pandas as pd
import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import requests

# Sidebar CSS styling for grouped navigation
SIDEBAR_CSS = """
<style>
    /* Style the sidebar navigation section headers based on page names */
    [data-testid="stSidebarNav"] li:has(a[href*="Map_Selector"]) span,
    [data-testid="stSidebarNav"] li:has(a[href*="Interactive_Plot"]) span {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%);
        color: white !important;
        padding: 2px 8px;
        border-radius: 4px;
    }
    
    [data-testid="stSidebarNav"] li:has(a[href*="Snow_Drift"]) span,
    [data-testid="stSidebarNav"] li:has(a[href*="Advanced_Analysis"]) span,
    [data-testid="stSidebarNav"] li:has(a[href*="Weather_Anomalies"]) span {
        background: linear-gradient(90deg, #1a472a 0%, #2d6a4f 100%);
        color: white !important;
        padding: 2px 8px;
        border-radius: 4px;
    }
    
    [data-testid="stSidebarNav"] li:has(a[href*="Correlations"]) span,
    [data-testid="stSidebarNav"] li:has(a[href*="Forecasting"]) span {
        background: linear-gradient(90deg, #4a1a6b 0%, #6b2d8a 100%);
        color: white !important;
        padding: 2px 8px;
        border-radius: 4px;
    }
</style>
"""

def render_sidebar_info():
    """Render sidebar styling and current data info"""
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("---")
        # Show current data info if available
        if 'weather_data' in st.session_state and st.session_state.weather_data is not None:
            st.markdown("### 📁 Current Data")
            if 'selected_price_area' in st.session_state and st.session_state.selected_price_area:
                st.info(f"📍 Area: **{st.session_state.selected_price_area}**")
            if 'data_range' in st.session_state:
                start_yr, end_yr = st.session_state.data_range
                st.info(f"📅 Years: **{start_yr}-{end_yr}**")


def check_data_requirements(require_weather=False, require_coordinates=False, require_energy=False):
    """
    Check if required data is available and show appropriate warnings.
    
    Parameters:
        require_weather: Check if weather data has been downloaded
        require_coordinates: Check if a location has been selected on the map
        require_energy: Check if energy data is available
    
    Returns:
        True if all requirements are met, False otherwise
    """
    missing = []
    
    if require_coordinates:
        if 'selected_coordinates' not in st.session_state or st.session_state.selected_coordinates is None:
            missing.append("location selection (click on the map)")
    
    if require_weather:
        if 'weather_data' not in st.session_state or st.session_state.weather_data is None:
            missing.append("weather data download")
    
    if require_energy:
        if 'production_data' not in st.session_state and 'consumption_data' not in st.session_state:
            missing.append("energy data")
    
    if missing:
        st.warning(f"⚠️ Missing: {', '.join(missing)}. Please visit the **Map Selector** page first.")
        return False
    return True


# MongoDB connection
@st.cache_resource
def init_connection():
    """Initialize MongoDB connection using secrets"""
    try:
        uri = st.secrets["URI"]
        client = MongoClient(uri, server_api=ServerApi('1'))
        return client
    except Exception:
        return None

# Load data from MongoDB
@st.cache_data(ttl=600)
def load_production_data():
    """Load production data from MongoDB"""
    client = init_connection()
    if client:
        try:
            db = client['energy_data']
            collection = db['production']
            data = list(collection.find({}))
            df = pd.DataFrame(data)
            df['startTime_parsed'] = pd.to_datetime(df['startTime'], utc=True)
            return df
        except Exception:
            pass
    
    # Mock Data Fallback
    dates = pd.date_range(start='2023-01-01', periods=8760, freq='H')
    df = pd.DataFrame({
        'startTime_parsed': dates,
        'quantityKwh': np.random.normal(100, 20, size=len(dates)) + \
                       np.sin(np.linspace(0, 100, len(dates))) * 50,
        'priceArea': 'NO1',
        'productionGroup': 'Wind'
    })
    return df

def get_production_data():
    """Get production data from session state or load it"""
    if 'production_data' not in st.session_state:
        st.session_state.production_data = load_production_data()
    return st.session_state.production_data

# Download weather data from open-meteo API
@st.cache_data
def download_weather_data(latitude, longitude, start_year, end_year):
    """
    Download historical weather data from open-meteo API for a range of years
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': f"{start_year}-01-01",
        'end_date': f"{end_year}-12-31",
        'hourly': [
            'temperature_2m',
            'precipitation',
            'wind_speed_10m',
            'wind_gusts_10m',
            'wind_direction_10m'
        ],
        'timezone': 'auto'
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}")
    
    data = response.json()
    hourly = data['hourly']
    
    df = pd.DataFrame({
        'time': pd.to_datetime(hourly['time']),
        'temperature_2m (°C)': hourly['temperature_2m'],
        'precipitation (mm)': hourly['precipitation'],
        'wind_speed_10m (m/s)': hourly['wind_speed_10m'],
        'wind_gusts_10m (m/s)': hourly['wind_gusts_10m'],
        'wind_direction_10m (°)': hourly['wind_direction_10m']
    })
    
    return df

# Load consumption data from MongoDB
@st.cache_data(ttl=600)
def load_consumption_data():
    """Load consumption data from MongoDB"""
    client = init_connection()
    if client:
        try:
            db = client['energy_data']
            collection = db['consumption']
            data = list(collection.find({}))
            df = pd.DataFrame(data)
            df['startTime_parsed'] = pd.to_datetime(df['startTime'], utc=True)
            return df
        except Exception:
            pass
    
    # Mock Data Fallback
    dates = pd.date_range(start='2023-01-01', periods=8760, freq='H')
    df = pd.DataFrame({
        'startTime_parsed': dates,
        'quantityKwh': np.random.normal(80, 15, size=len(dates)) + \
                       np.cos(np.linspace(0, 100, len(dates))) * 40,
        'priceArea': 'NO1',
        'consumptionGroup': 'Household'  # Fixed: was incorrectly named 'productionGroup'
    })
    return df

def get_consumption_data():
    """Get consumption data from session state or load it"""
    if 'consumption_data' not in st.session_state:
        st.session_state.consumption_data = load_consumption_data()
    return st.session_state.consumption_data

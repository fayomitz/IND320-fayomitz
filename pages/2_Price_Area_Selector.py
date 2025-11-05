import streamlit as st
import pandas as pd
import requests

# Page configuration
st.set_page_config(
    page_title="Price Area Selector",
    layout="wide"
)

# Norwegian cities with coordinates
CITIES_DATA = {
    'NO1': {'city': 'Oslo', 'latitude': 59.9139, 'longitude': 10.7522},
    'NO2': {'city': 'Kristiansand', 'latitude': 58.1462, 'longitude': 7.9956},
    'NO3': {'city': 'Trondheim', 'latitude': 63.4305, 'longitude': 10.3951},
    'NO4': {'city': 'Tromsø', 'latitude': 69.6492, 'longitude': 18.9553},
    'NO5': {'city': 'Bergen', 'latitude': 60.3913, 'longitude': 5.3221}
}

# Download weather data from open-meteo API
@st.cache_data
def download_weather_data(longitude, latitude, year):
    """
    Download historical weather data from open-meteo API
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': f"{year}-01-01",
        'end_date': f"{year}-12-31",
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

# Main page content
st.title("Price Area Selector & Weather Data")
st.header("Select Norwegian Price Area and Year to Download Weather Data")

st.markdown("""
Select a Norwegian price area and year to download historical weather data. This selection will determine 
which data is used across other pages in this application.
""")

st.markdown("---")

try:
    # Price area selector
    st.subheader("🌍 Select Price Area")
    
    # Radio buttons for price area selection
    price_areas = list(CITIES_DATA.keys())
    selected_area = st.radio(
        "Choose a Norwegian price area:",
        options=price_areas,
        format_func=lambda x: f"{x} - {CITIES_DATA[x]['city']}",
        horizontal=True,
        help="Select a price area to download weather data for the corresponding city"
    )
    
    # Display selected area info
    selected_city = CITIES_DATA[selected_area]['city']
    selected_lat = CITIES_DATA[selected_area]['latitude']
    selected_lon = CITIES_DATA[selected_area]['longitude']
    
    st.info(f"📍 Selected: **{selected_area}** - {selected_city} ({selected_lat:.4f}°N, {selected_lon:.4f}°E)")
    
    # Year selector
    st.markdown("---")
    st.subheader("📅 Select Year")
    
    selected_year = st.selectbox(
        "Choose a year for weather data:",
        options=list(range(2024, 2009, -1)),  # API data available from 2010 onwards
        index=1,  # Default to 2023 (index 1 in the list)
        help="Select any year from 2010 to 2024"
    )
    
    st.info(f"📆 Selected year: **{selected_year}**")
    
    # Download weather data button
    st.markdown("---")
    st.subheader("☁️ Download Weather Data")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"Click the button below to download {selected_year} weather data from Open-Meteo API for the selected price area.")
    with col2:
        download_button = st.button("📥 Download Data", type="primary", width='stretch')
    
    if download_button:
        with st.spinner(f"Downloading weather data for {selected_city} ({selected_year})..."):
            try:
                # Download data
                weather_data = download_weather_data(
                    longitude=selected_lon,
                    latitude=selected_lat,
                    year=selected_year
                )
                
                # Store in session state
                st.session_state.weather_data = weather_data
                st.session_state.selected_area = selected_area
                st.session_state.selected_city = selected_city
                st.session_state.selected_year = selected_year
                
                st.success(f"✅ Successfully downloaded {len(weather_data)} hourly records for {selected_city} ({selected_year})!")
                
                # Display summary
                st.markdown("---")
                st.subheader("📊 Data Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", f"{len(weather_data):,}")
                with col2:
                    st.metric("Start Date", weather_data['time'].min().strftime('%Y-%m-%d'))
                with col3:
                    st.metric("End Date", weather_data['time'].max().strftime('%Y-%m-%d'))
                
                # Show preview
                with st.expander("🔍 Preview Data (First 10 rows)"):
                    preview_data = weather_data.head(10).copy()
                    preview_data['time'] = preview_data['time'].astype(str)
                    st.dataframe(preview_data, width='stretch')
                
                # Show statistics
                with st.expander("📈 Statistical Summary"):
                    # Exclude time column from statistics to avoid Arrow serialization issues
                    numeric_cols = weather_data.select_dtypes(include='number').columns
                    st.dataframe(weather_data[numeric_cols].describe(), width='stretch')
                
            except Exception as e:
                st.error(f"❌ Error downloading weather data: {str(e)}")
                st.exception(e)
    
    # Show current session data if available
    if 'weather_data' in st.session_state and st.session_state.weather_data is not None:
        st.markdown("---")
        st.success("✅ Weather data is currently loaded and available for analysis on other pages")
        loaded_year = st.session_state.get('selected_year', 'Unknown')
        st.info(f"Current data: **{st.session_state.selected_area}** - {st.session_state.selected_city} ({loaded_year})")
        
        # Option to clear data
        if st.button("🗑️ Clear Loaded Data"):
            st.session_state.weather_data = None
            st.session_state.selected_area = None
            st.session_state.selected_city = None
            st.session_state.selected_year = None
            st.rerun()
    
    # Information about data source
    st.markdown("---")
    with st.expander("ℹ️ About the Weather Data"):
        st.markdown("""
        ### Data Source
        
        **Source:** Open-Meteo API  
        **API Endpoint:** `https://archive-api.open-meteo.com/v1/archive`  
        **Model:** ERA5 Reanalysis
        
        **Description:**  
        Historical weather data providing hourly measurements for the selected location and year.
        
        **Variables:**
        - `temperature_2m (°C)`: Air temperature at 2 meters above ground
        - `precipitation (mm)`: Total precipitation
        - `wind_speed_10m (m/s)`: Wind speed at 10 meters
        - `wind_gusts_10m (m/s)`: Maximum wind gusts
        - `wind_direction_10m (°)`: Wind direction (0-360°)
        
        **Coverage:**
        - **Years:** 2010-2024
        - **Frequency:** Hourly
        - **Locations:** Major cities representing each Norwegian price area
        
        **Usage:**
        Once downloaded, the data is stored in the session and can be used across all pages
        in this application for various analyses and visualizations.
        """)
    
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)

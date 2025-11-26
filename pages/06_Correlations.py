import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_production_data, get_consumption_data, download_weather_data, render_sidebar_info, check_data_requirements

st.set_page_config(page_title="Correlations", layout="wide")

# Render sidebar data info
render_sidebar_info()

# --- Coordinate Mapping for Price Areas ---
AREA_COORDINATES = {
    "NO1": (59.91, 10.75),  # Oslo
    "NO2": (58.15, 8.0),    # Kristiansand
    "NO3": (63.43, 10.39),  # Trondheim
    "NO4": (69.65, 18.96),  # Tromsø
    "NO5": (60.39, 5.32)    # Bergen
}

# --- Data Loading ---


# --- Main Page ---

st.title("🔗 Sliding Window Correlation")
st.markdown("Analyze the dynamic relationship between Weather and Energy data.")

try:
    # 1. Check for required data
    if not check_data_requirements(require_weather=True):
        st.stop()

    weather_df = st.session_state.weather_data.copy()
    weather_df['time'] = pd.to_datetime(weather_df['time'], utc=True)

    # 2. Data Type Selection (Production or Consumption)
    col_type, col_rest = st.columns([1, 3])
    
    with col_type:
        data_type = st.radio("Energy Data Type", ["Production", "Consumption"], horizontal=True)
    
    # Load appropriate energy data
    if data_type == "Production":
        energy_df = get_production_data()
        group_column = 'productionGroup'
    else:
        energy_df = get_consumption_data()
        group_column = 'consumptionGroup'
    
    if energy_df is None or energy_df.empty:
        st.error(f"Failed to load {data_type.lower()} data. Please check your data connection.")
        st.stop()

    # 3. Selectors
    col1, col2, col3 = st.columns(3)

    with col1:
        weather_cols = [c for c in weather_df.columns if c not in ['time', 'season', 'year_month']]
        if not weather_cols:
            st.error("No valid weather columns found in the data.")
            st.stop()
        weather_col = st.selectbox("Weather Variable", weather_cols)

    with col2:
        # Filter Energy Data
        areas = sorted(energy_df['priceArea'].dropna().unique())
        groups = sorted(energy_df[group_column].dropna().unique())
        
        if not areas or not groups:
            st.error(f"No valid areas or groups found in {data_type.lower()} data.")
            st.stop()
        
        # Get default area from session state (from map selector)
        default_area = st.session_state.get('selected_price_area', None)
        default_idx = 0
        if default_area and default_area in areas:
            default_idx = areas.index(default_area)
        
        sel_area = st.selectbox("Price Area", areas, index=default_idx)
        sel_group = st.selectbox(f"{data_type} Group", groups)
        
        energy_subset = energy_df[
            (energy_df['priceArea'] == sel_area) & 
            (energy_df[group_column] == sel_group)
        ].copy()
        
        # Check if we need to download weather data for this area
        need_weather_download = False
        stored_area = st.session_state.get('weather_data_area', None)
        
        if sel_area != stored_area:
            need_weather_download = True

    with col3:
        window_size = st.number_input("Window Size (Hours)", min_value=24, value=168)
        lag = st.number_input("Lag (Hours)", min_value=-168, max_value=168, value=0)

    # 4. Merge Data
    if energy_subset.empty:
        st.warning(f"No {data_type.lower()} data found for {sel_area} - {sel_group}. Try different selections.")
        st.stop()
    
    with st.spinner("Processing data..."):
        # Check if we need to download weather data for the selected area
        if need_weather_download:
            # Get year range from session state or default
            data_range = st.session_state.get('data_range', (2021, 2024))
            start_year, end_year = data_range
            
            lat, lon = AREA_COORDINATES.get(sel_area, (59.91, 10.75))
            st.info(f"Downloading weather data for {sel_area} ({start_year}-{end_year})...")
            
            try:
                weather_df = download_weather_data(lat, lon, start_year, end_year)
                weather_df['time'] = pd.to_datetime(weather_df['time'], utc=True)
                st.session_state.weather_data = weather_df
                st.session_state.weather_data_area = sel_area
            except Exception as e:
                st.error(f"Failed to download weather data: {e}")
                st.stop()
        
        # Resample to hourly to ensure alignment
        weather_hourly = weather_df.set_index('time').resample('h').mean()
        energy_hourly = energy_subset.set_index('startTime_parsed')['quantityKwh'].resample('h').sum()

        merged = pd.merge(weather_hourly, energy_hourly, left_index=True, right_index=True, how='inner')
        merged.rename(columns={'quantityKwh': 'Energy'}, inplace=True)

    if merged.empty:
        st.error("No overlapping data found between Weather and Energy datasets.")
        st.info("This could be due to different time ranges in the datasets. Check that both datasets cover the same period.")
        st.stop()

    # 5. Calculate Correlation
    # Apply lag to weather data (positive lag means weather leads energy)
    shifted_weather = merged[weather_col].shift(lag)
    correlation = shifted_weather.rolling(window=window_size).corr(merged['Energy'])
    
    # Check for valid correlation values
    valid_corr = correlation.dropna()
    if valid_corr.empty:
        st.warning("Not enough data points to calculate correlation with the selected window size.")
        st.info(f"Try reducing the window size (currently {window_size} hours) or selecting a larger date range.")
        st.stop()

    # 6. Plotting
    st.subheader("Correlation over Time")
    fig_corr = px.line(x=merged.index, y=correlation, 
                       title=f"Sliding Window Correlation (Window={window_size}h, Lag={lag}h)")
    fig_corr.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_corr.update_layout(
        xaxis_title="Time",
        yaxis_title="Correlation Coefficient"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Display correlation statistics
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Mean Correlation", f"{valid_corr.mean():.3f}")
    with col_stat2:
        st.metric("Max Correlation", f"{valid_corr.max():.3f}")
    with col_stat3:
        st.metric("Min Correlation", f"{valid_corr.min():.3f}")

    st.subheader("Data Overview")
    fig_dual = go.Figure()
    fig_dual.add_trace(go.Scatter(x=merged.index, y=merged[weather_col], name=weather_col, yaxis='y1'))
    fig_dual.add_trace(go.Scatter(x=merged.index, y=merged['Energy'], name=f'{data_type} Energy', yaxis='y2'))

    fig_dual.update_layout(
        yaxis=dict(title=weather_col),
        yaxis2=dict(title="Energy (kWh)", overlaying='y', side='right'),
        title=f"Weather vs {data_type} Energy Time Series",
        xaxis_title="Time"
    )
    st.plotly_chart(fig_dual, use_container_width=True)
    
    st.info(f"Showing {len(merged):,} overlapping data points between weather and {data_type.lower()} energy data.")

except Exception as e:
    st.error(f"An error occurred while processing the data: {str(e)}")
    st.info("Please check your data selections and try again. If the problem persists, try reloading the weather data from the Map Selector page.")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import render_sidebar_info, check_data_requirements

st.set_page_config(page_title="Snow Drift Analysis", layout="wide")

# Render sidebar data info
render_sidebar_info()

# --- Helper Functions (Adapted from Snow_drift.py) ---

def compute_Qupot(hourly_wind_speeds, dt=3600):
    """
    Compute the potential wind-driven snow transport (Qupot) [kg/m]
    by summing hourly contributions using u^3.8.
    """
    total = sum((u ** 3.8) * dt for u in hourly_wind_speeds) / 233847
    return total

def sector_index(direction):
    """Returns index (0-15) for 16-sector division."""
    return int(((direction + 11.25) % 360) // 22.5)

def compute_sector_transport(hourly_wind_speeds, hourly_wind_dirs, dt=3600):
    """Compute cumulative transport for each of 16 wind sectors."""
    sectors = [0.0] * 16
    for u, d in zip(hourly_wind_speeds, hourly_wind_dirs):
        idx = sector_index(d)
        sectors[idx] += ((u ** 3.8) * dt) / 233847
    return sectors

def compute_snow_transport(T, F, theta, Swe, hourly_wind_speeds, dt=3600):
    """
    Compute snow drifting transport components according to Tabler (2003).
    """
    Qupot = compute_Qupot(hourly_wind_speeds, dt)
    Qspot = 0.5 * T * Swe
    Srwe = theta * Swe
    
    if Qupot > Qspot:
        Qinf = 0.5 * T * Srwe
        control = "Snowfall controlled"
    else:
        Qinf = Qupot
        control = "Wind controlled"
    
    Qt = Qinf * (1 - 0.14 ** (F / T))
    
    return Qt, control

# --- Main Page ---
st.title("❄️ Snow Drift Analysis")

# Check for required data (weather data and coordinates from map)
if not check_data_requirements(require_weather=True, require_coordinates=True):
    st.stop()

df = st.session_state.weather_data.copy()

# Parameters
with st.expander("Configuration", expanded=True):
    with st.form("snow_params"):
        st.header("Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            T = st.number_input("Max Transport Distance (T) [m]", value=3000)
        with col2:
            F = st.number_input("Fetch Distance (F) [m]", value=30000)
        with col3:
            theta = st.number_input("Relocation Coefficient (theta)", value=0.5)
        
        # Define Seasons (July 1 - June 30)
        df['season'] = df['time'].apply(lambda dt: dt.year if dt.month >= 7 else dt.year - 1)
        available_seasons = sorted(df['season'].unique())

        # Filter out incomplete first/last seasons if necessary, or just let user choose
        selected_seasons = st.multiselect(
            "Select Seasons (July-June)",
            options=available_seasons,
            default=available_seasons[-3:] if len(available_seasons) > 3 else available_seasons,
            format_func=lambda x: f"{x}-{x+1}"
        )
        
        update_snow = st.form_submit_button("Update Analysis")

if not selected_seasons:
    st.stop()

# --- Calculations ---
annual_results = []
monthly_results = []
sectors_list = []

for s in selected_seasons:
    season_start = pd.Timestamp(year=s, month=7, day=1)
    season_end = pd.Timestamp(year=s+1, month=6, day=30, hour=23, minute=59)
    
    # Filter for this season
    mask = (df['time'] >= season_start) & (df['time'] <= season_end)
    df_season = df[mask].copy()
    
    if df_season.empty:
        continue

    # Calculate Hourly SWE (Precipitation as snow if Temp < 1C)
    df_season['Swe_hourly'] = df_season.apply(
        lambda row: row['precipitation (mm)'] if row['temperature_2m (°C)'] < 1 else 0, axis=1
    )
    
    # 1. Annual Calculation
    total_Swe = df_season['Swe_hourly'].sum()
    wind_speeds = df_season["wind_speed_10m (m/s)"].tolist()
    wind_dirs = df_season["wind_direction_10m (°)"].tolist()
    
    Qt_annual, control = compute_snow_transport(T, F, theta, total_Swe, wind_speeds)
    annual_results.append({
        'Season': f"{s}-{s+1}",
        'Qt (tonnes/m)': Qt_annual / 1000.0,
        'Control': control
    })

    # Wind Rose Data (Sector Transport)
    sectors = compute_sector_transport(wind_speeds, wind_dirs)
    sectors_list.append(sectors)
    
    # 2. Monthly Calculation
    df_season['month_idx'] = df_season['time'].dt.month
    # Sort months to be July -> June
    month_order = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    
    for m in month_order:
        df_month = df_season[df_season['month_idx'] == m]
        if df_month.empty:
            continue
            
        m_swe = df_month['Swe_hourly'].sum()
        m_wind = df_month["wind_speed_10m (m/s)"].tolist()
        
        Qt_month, _ = compute_snow_transport(T, F, theta, m_swe, m_wind)
        
        monthly_results.append({
            'Season': f"{s}-{s+1}",
            'Month': pd.to_datetime(f"2000-{m}-01").strftime('%b'), # Month name
            'Month_Num': m if m >= 7 else m + 12, # For sorting
            'Qt (tonnes/m)': Qt_month / 1000.0
        })

# --- Visualization ---
tab1, tab2 = st.tabs(["Annual Overview", "Monthly Breakdown"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Annual Snow Transport")
        df_annual = pd.DataFrame(annual_results)
        fig_annual = px.bar(df_annual, x='Season', y='Qt (tonnes/m)', color='Control', 
                            title="Total Snow Transport per Season")
        st.plotly_chart(fig_annual, use_container_width=True)

    with col2:
        st.subheader("Wind Rose (Directional Transport)")
        if sectors_list:
            # Average sectors over selected seasons
            avg_sectors = np.mean(sectors_list, axis=0)
            avg_sectors_tonnes = np.array(avg_sectors) / 1000.0
            
            directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                          'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
            
            fig_rose = go.Figure(go.Barpolar(
                r=avg_sectors_tonnes,
                theta=[i * 22.5 for i in range(16)],
                width=[22.5]*16,
                marker_color='blue',
                marker_line_color='black',
                marker_line_width=1,
                opacity=0.8
            ))
            
            fig_rose.update_layout(
                template='plotly_white',
                polar=dict(
                    radialaxis=dict(showticklabels=True, ticks=''),
                    angularaxis=dict(
                        tickmode='array',
                        tickvals=[i * 22.5 for i in range(16)],
                        ticktext=directions,
                        direction='clockwise',
                        rotation=90
                    )
                ),
                title="Avg Directional Distribution (Tonnes/m)"
            )
            st.plotly_chart(fig_rose, use_container_width=True)
        else:
            st.info("No wind data available for Wind Rose.")

with tab2:
    st.subheader("Monthly Snow Transport")
    df_monthly = pd.DataFrame(monthly_results)
    
    # Sort by custom month order
    df_monthly = df_monthly.sort_values(['Season', 'Month_Num'])
    
    fig_monthly = px.line(df_monthly, x='Month', y='Qt (tonnes/m)', color='Season',
                          title="Monthly Snow Transport Comparison", markers=True)
    st.plotly_chart(fig_monthly, use_container_width=True)

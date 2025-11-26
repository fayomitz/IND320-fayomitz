import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.fft import dct, idct
from sklearn.neighbors import LocalOutlierFactor
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import render_sidebar_info, check_data_requirements

# Page configuration
st.set_page_config(
    page_title="Weather Anomalies Detection",
    layout="wide"
)

# Render sidebar data info
render_sidebar_info()

# Load weather data from session state (set by page 2)
def get_weather_data():
    """Get weather data from session state"""
    if 'weather_data' in st.session_state and st.session_state.weather_data is not None:
        return st.session_state.weather_data
    return None

def detect_temperature_outliers(df, freq_cutoff=0.05, n_std=3):
    """
    Detect temperature outliers using DCT and SPC
    """
    # Extract temperature data
    temp = df['temperature_2m (°C)'].ffill().bfill().values
    time = pd.to_datetime(df['time'])
    
    # Apply DCT
    temp_dct = dct(temp, type=2, norm='ortho')
    
    # High-pass filter
    cutoff_index = int(len(temp_dct) * freq_cutoff)
    temp_dct_filtered = temp_dct.copy()
    temp_dct_filtered[:cutoff_index] = 0
    
    # Get SATV
    satv = idct(temp_dct_filtered, type=2, norm='ortho')

    # Get Trend (Low-frequency component)
    # We reconstruct the part we filtered out to get the trend
    temp_dct_trend = np.zeros_like(temp_dct)
    temp_dct_trend[:cutoff_index] = temp_dct[:cutoff_index]
    trend = idct(temp_dct_trend, type=2, norm='ortho')
    
    # Calculate robust statistics
    median_satv = np.median(satv)
    mad_satv = np.median(np.abs(satv - median_satv))
    std_satv = mad_satv * 1.4826
    
    # SPC boundaries (Static for SATV)
    upper_boundary = median_satv + n_std * std_satv
    lower_boundary = median_satv - n_std * std_satv

    # Dynamic boundaries (For Raw Data)
    # Add the trend back to the boundaries so they follow the data
    upper_dynamic = trend + upper_boundary
    lower_dynamic = trend + lower_boundary
    
    # Identify outliers
    outliers_mask = (satv > upper_boundary) | (satv < lower_boundary)
    n_outliers = np.sum(outliers_mask)
    outlier_percentage = (n_outliers / len(temp)) * 100
    
    # Create visualization with Plotly
    fig = go.Figure()
    
    # Plot normal points
    fig.add_trace(go.Scatter(
        x=time[~outliers_mask], 
        y=temp[~outliers_mask],
        mode='lines',
        name='Normal temperature',
        line=dict(color='green', width=1),
        opacity=0.7
    ))
    
    # Plot outliers
    fig.add_trace(go.Scatter(
        x=time[outliers_mask], 
        y=temp[outliers_mask],
        mode='markers',
        name=f'Outliers (n={n_outliers})',
        marker=dict(color='red', size=6, symbol='circle')
    ))

    # Plot Dynamic Boundaries
    fig.add_trace(go.Scatter(
        x=time,
        y=upper_dynamic,
        mode='lines',
        name='Upper Limit',
        line=dict(color='gray', width=1, dash='dash'),
        opacity=0.5
    ))
    
    fig.add_trace(go.Scatter(
        x=time,
        y=lower_dynamic,
        mode='lines',
        name='Lower Limit',
        line=dict(color='gray', width=1, dash='dash'),
        opacity=0.5,
        fill='tonexty' # Optional: fills the area between limits
    ))
    
    fig.update_layout(
        title=f'Temperature Outlier Detection (DCT + SPC) - Cutoff: {freq_cutoff}, ±{n_std}σ',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        template='plotly_white'
    )
    
    results = {
        'figure': fig,
        'n_outliers': n_outliers,
        'outlier_percentage': outlier_percentage,
        'upper_boundary': upper_boundary,
        'lower_boundary': lower_boundary,
        'outlier_dates': time[outliers_mask].tolist(),
        'outlier_values': temp[outliers_mask].tolist()
    }
    
    return results

def detect_precipitation_anomalies(df, outlier_proportion=0.01, n_neighbors=50):
    """
    Detect precipitation anomalies using LOF
    """
    # Extract precipitation data
    precip = df['precipitation (mm)'].values
    time = pd.to_datetime(df['time'])
    
    # Prepare features for LOF
    precip_diff = np.diff(precip, prepend=precip[0])
    X = np.column_stack([precip, precip_diff])
    
    # Add small jitter to handle duplicate values (common with zero precipitation)
    jitter = np.random.RandomState(42).normal(0, 1e-6, X.shape)
    X_jittered = X + jitter
    
    # Fit LOF
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=outlier_proportion)
    predictions = lof.fit_predict(X_jittered)
    lof_scores = lof.negative_outlier_factor_
    
    # Identify anomalies
    anomalies_mask = predictions == -1
    n_anomalies = np.sum(anomalies_mask)
    anomaly_percentage = (n_anomalies / len(precip)) * 100
    
    # Create visualization with Plotly
    fig = go.Figure()
    
    # Plot normal precipitation
    fig.add_trace(go.Bar(
        x=time[~anomalies_mask], 
        y=precip[~anomalies_mask],
        name='Normal precipitation',
        marker_color='blue',
        opacity=0.6
    ))
    
    # Plot anomalies (Bars)
    fig.add_trace(go.Bar(
        x=time[anomalies_mask], 
        y=precip[anomalies_mask],
        name=f'Anomalies (n={n_anomalies})',
        marker_color='red',
        opacity=1
    ))

    # Plot anomalies (Markers for visibility)
    # This helps spot anomalies even when zoomed out (thin bars)
    fig.add_trace(go.Scatter(
        x=time[anomalies_mask],
        y=precip[anomalies_mask],
        mode='markers',
        name='Anomaly Markers',
        marker=dict(color='red', size=8, symbol='circle', line=dict(color='white', width=1)),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f'Precipitation Anomaly Detection (LOF) - Prop: {outlier_proportion*100:.1f}%',
        xaxis_title='Time',
        yaxis_title='Precipitation (mm)',
        template='plotly_white',
        barmode='overlay'
    )
    
    # Get anomaly details
    anomaly_dates_list = time[anomalies_mask].tolist()
    
    results = {
        'figure': fig,
        'n_anomalies': n_anomalies,
        'anomaly_percentage': anomaly_percentage,
        'lof_scores': lof_scores,
        'anomaly_dates': anomaly_dates_list,
        'anomaly_values': precip[anomalies_mask].tolist()
    }
    
    return results

# Main page content
st.title("Weather Anomalies Detection")
st.header("Temperature Outliers and Precipitation Anomalies")

st.markdown("""
Detect unusual weather patterns using advanced statistical methods.
""")

st.markdown("---")

# Check if weather data is available
if not check_data_requirements(require_weather=True):
    st.stop()

weather_df = get_weather_data()

st.success(f"✅ Weather data loaded: {len(weather_df)} records")

# Display selected area info
if 'selected_price_area' in st.session_state:
    st.info(f"📍 Analyzing data for: **{st.session_state.selected_price_area}**")

# Create tabs
tab1, tab2 = st.tabs(["🌡️ Temperature Outliers (SPC)", "🌧️ Precipitation Anomalies (LOF)"])

# Tab 1: Temperature Outliers
with tab1:
    st.subheader("Temperature Outlier Detection using DCT and SPC")
    
    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        freq_cutoff = st.slider(
            "Frequency Cutoff",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Lower = removes more seasonal trends (0.05 recommended)"
        )
    
    with col2:
        n_std = st.slider(
            "Standard Deviations",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="Number of σ for SPC boundaries (3σ = 99.7%)"
        )
    
    # Perform analysis
    if st.button("Detect Temperature Outliers", key="temp_button"):
        with st.spinner("Analyzing temperature data..."):
            results = detect_temperature_outliers(
                weather_df,
                freq_cutoff=freq_cutoff,
                n_std=n_std
            )
            
            st.plotly_chart(results['figure'], use_container_width=True)
            
            # Statistics
            st.markdown("---")
            st.subheader("📊 Outlier Statistics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Outliers", results['n_outliers'])
            with col2:
                st.metric("Percentage", f"{results['outlier_percentage']:.2f}%")
            with col3:
                st.metric("Boundary Range", f"±{n_std}σ")
            
            # Show outlier details
            if results['n_outliers'] > 0:
                with st.expander("🔍 View Outlier Details"):
                    outlier_df = pd.DataFrame({
                        'Date': [d.strftime('%Y-%m-%d %H:%M:%S') if hasattr(d, 'strftime') else str(d) for d in results['outlier_dates']],
                        'Temperature (°C)': [f"{v:.2f}" for v in results['outlier_values']]
                    })
                    st.dataframe(outlier_df, width='stretch', height=300)

# Tab 2: Precipitation Anomalies
with tab2:
    st.subheader("Precipitation Anomaly Detection using LOF")
    
    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        outlier_prop = st.slider(
            "Expected Outlier Proportion",
            min_value=0.001,
            max_value=0.05,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="Expected proportion of anomalies (1% = 0.01)"
        )
    
    with col2:
        n_neighbors = st.slider(
            "Number of Neighbors",
            min_value=20,
            max_value=200,
            value=50,
            step=10,
            help="Higher values reduce impact of duplicate values"
        )
    
    st.info(f"Expecting approximately {int(len(weather_df) * outlier_prop)} anomalies out of {len(weather_df)} data points")
    
    # Perform analysis
    if st.button("Detect Precipitation Anomalies", key="precip_button"):
        with st.spinner("Analyzing precipitation data..."):
            results = detect_precipitation_anomalies(
                weather_df,
                outlier_proportion=outlier_prop,
                n_neighbors=n_neighbors
            )
            
            st.plotly_chart(results['figure'], use_container_width=True)
            
            # Statistics
            st.markdown("---")
            st.subheader("📊 Anomaly Statistics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Anomalies", results['n_anomalies'])
            with col2:
                st.metric("Percentage", f"{results['anomaly_percentage']:.2f}%")
            with col3:
                st.metric("Mean LOF Score", f"{np.mean(results['lof_scores']):.2f}")
            
            # Show anomaly details
            if results['n_anomalies'] > 0:
                with st.expander("🔍 View Anomaly Details (Top 20)"):
                    # Sort by precipitation value
                    anomaly_df = pd.DataFrame({
                        'Date': [d.strftime('%Y-%m-%d %H:%M:%S') if hasattr(d, 'strftime') else str(d) for d in results['anomaly_dates']],
                        'Precipitation (mm)': results['anomaly_values']
                    })
                    anomaly_df = anomaly_df.sort_values('Precipitation (mm)', ascending=False)
                    st.dataframe(anomaly_df.head(20), width='stretch', height=400)

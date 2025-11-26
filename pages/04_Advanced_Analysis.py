import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL
from scipy import signal
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_production_data, render_sidebar_info

# Page configuration
st.set_page_config(
    page_title="Advanced Time Series Analysis",
    layout="wide"
)

# Render sidebar data info
render_sidebar_info()


def stl_analysis(df, price_area, production_group, period=24, seasonal=7, trend=None, robust=False):
    """
    Perform STL decomposition on production data
    """
    # Filter data
    filtered = df[(df['priceArea'] == price_area) & (df['productionGroup'] == production_group)]
    
    if len(filtered) == 0:
        return None, "No data available for selected combination"
    
    # Create time series
    filtered = filtered.sort_values('startTime_parsed')
    ts = pd.Series(
        filtered['quantityKwh'].values,
        index=filtered['startTime_parsed']
    )
    
    # Handle missing values
    ts = ts.ffill().bfill()
    
    # Perform STL decomposition
    stl = STL(ts, period=period, seasonal=seasonal, trend=trend, robust=robust)
    result = stl.fit()
    
    # Create visualization with Plotly
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        subplot_titles=("Original", "Trend", "Seasonal", "Residual"))
    
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Original", line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=result.trend, name="Trend", line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=result.seasonal, name="Seasonal", line=dict(color='green')), row=3, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=result.resid, name="Residual", line=dict(color='red')), row=4, col=1)
    
    fig.update_layout(height=800, title_text=f"STL Decomposition: {production_group} in {price_area}")
    
    return fig, None

def spectrogram_analysis(df, price_area, production_group, window_length=168, window_overlap=84):
    """
    Create spectrogram of production data
    """
    # Filter data
    filtered = df[(df['priceArea'] == price_area) & (df['productionGroup'] == production_group)]
    
    if len(filtered) == 0:
        return None, "No data available for selected combination"
    
    # Sort and extract production values
    filtered = filtered.sort_values('startTime_parsed')
    production = filtered['quantityKwh'].values
    
    # Handle missing values
    production = pd.Series(production).ffill().bfill().values
    
    # Compute STFT
    frequencies, times, Sxx = signal.spectrogram(
        production,
        fs=1.0,
        window='hann',
        nperseg=window_length,
        noverlap=window_overlap
    )
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Create visualization with Plotly
    fig = go.Figure(data=go.Heatmap(
        z=Sxx_db,
        x=times,
        y=frequencies,
        colorscale='Viridis',
        colorbar=dict(title='Power (dB)')
    ))
    
    fig.update_layout(
        title=f'Spectrogram: {production_group} Production in {price_area}',
        xaxis_title='Time (hours from start)',
        yaxis_title='Frequency (cycles/hour)'
    )
    
    return fig, None

# Main page content
st.title("Advanced Time Series Analysis")
st.header("STL Decomposition and Spectrogram Analysis")

st.markdown("""
Analyze electricity production patterns using advanced time series techniques.
""")

st.markdown("---")

try:
    # Load data
    df = get_production_data()
    
    # Get unique values
    price_areas = sorted(df['priceArea'].unique())
    production_groups = sorted(df['productionGroup'].unique())
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 STL Decomposition", "🎵 Spectrogram"])
    
    # Tab 1: STL Decomposition
    with tab1:
        st.subheader("Seasonal-Trend Decomposition using LOESS (STL)")
        
        # Controls in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            stl_price_area = st.selectbox(
                "Price Area",
                options=price_areas,
                key="stl_area"
            )
        
        with col2:
            stl_prod_group = st.selectbox(
                "Production Group",
                options=production_groups,
                key="stl_group"
            )
        
        with col3:
            stl_period = st.number_input(
                "Seasonal Period",
                min_value=2,
                max_value=720,
                value=24,
                help="Length of seasonal cycle (24=daily, 168=weekly)"
            )
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            stl_seasonal = st.slider(
                "Seasonal Smoothing",
                min_value=3,
                max_value=25,
                value=7,
                step=2,
                help="Higher = smoother seasonal component (must be odd)"
            )
        
        with col5:
            stl_robust = st.checkbox(
                "Robust Fitting",
                value=True,
                help="Resistant to outliers"
            )
        
        # Make seasonal odd
        if stl_seasonal % 2 == 0:
            stl_seasonal += 1
        
        # Perform analysis
        if st.button("Run STL Analysis", key="stl_button"):
            with st.spinner("Performing STL decomposition..."):
                fig, error = stl_analysis(
                    df, stl_price_area, stl_prod_group,
                    period=stl_period,
                    seasonal=stl_seasonal,
                    robust=stl_robust
                )
                
                if error:
                    st.error(error)
                else:
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Spectrogram
    with tab2:
        st.subheader("Spectrogram - Frequency-Time Analysis")
        
        # Controls
        col1, col2 = st.columns(2)
        
        with col1:
            spec_price_area = st.selectbox(
                "Price Area",
                options=price_areas,
                key="spec_area"
            )
        
        with col2:
            spec_prod_group = st.selectbox(
                "Production Group",
                options=production_groups,
                key="spec_group"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            spec_window = st.slider(
                "Window Length (hours)",
                min_value=24,
                max_value=720,
                value=168,
                step=24,
                help="Larger = better frequency resolution"
            )
        
        with col4:
            spec_overlap = st.slider(
                "Window Overlap (hours)",
                min_value=0,
                max_value=int(spec_window * 0.9),
                value=int(spec_window * 0.5),
                step=12,
                help="Higher = smoother spectrogram"
            )
        
        # Perform analysis
        if st.button("Create Spectrogram", key="spec_button"):
            with st.spinner("Computing spectrogram..."):
                fig, error = spectrogram_analysis(
                    df, spec_price_area, spec_prod_group,
                    window_length=spec_window,
                    window_overlap=spec_overlap
                )
                
                if error:
                    st.error(error)
                else:
                    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from statsmodels.tsa.seasonal import STL
from scipy import signal
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Advanced Time Series Analysis",
    layout="wide"
)

# MongoDB connection
@st.cache_resource
def init_connection():
    """Initialize MongoDB connection using secrets"""
    uri = st.secrets["URI"]
    client = MongoClient(uri, server_api=ServerApi('1'))
    return client

# Load data from MongoDB
@st.cache_data(ttl=600)
def load_production_data():
    """Load production data from MongoDB"""
    client = init_connection()
    db = client['energy_data']
    collection = db['production']
    
    # Fetch all documents
    data = list(collection.find({}))
    df = pd.DataFrame(data)
    
    # Convert timestamp strings to datetime
    df['startTime_parsed'] = pd.to_datetime(df['startTime'], utc=True)
    df['endTime_parsed'] = pd.to_datetime(df['endTime'], utc=True)
    
    return df

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
    ts = ts.fillna(method='ffill').fillna(method='bfill')
    
    # Perform STL decomposition
    stl = STL(ts, period=period, seasonal=seasonal, trend=trend, robust=robust)
    result = stl.fit()
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    
    # Plot original data
    axes[0].plot(ts.index, ts.values, color='black', linewidth=1)
    axes[0].set_ylabel('Original', fontsize=11, fontweight='bold')
    axes[0].set_title(f'STL Decomposition: {production_group} in {price_area}', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[0].grid(True, alpha=0.3)
    
    # Plot trend component
    axes[1].plot(ts.index, result.trend, color='blue', linewidth=1.5)
    axes[1].set_ylabel('Trend', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot seasonal component
    axes[2].plot(ts.index, result.seasonal, color='green', linewidth=1)
    axes[2].set_ylabel('Seasonal', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Plot residual component
    axes[3].plot(ts.index, result.resid, color='red', linewidth=1)
    axes[3].set_ylabel('Residual', fontsize=11, fontweight='bold')
    axes[3].set_xlabel('Time', fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
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
    production = pd.Series(production).fillna(method='ffill').fillna(method='bfill').values
    
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
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(15, 8))
    
    im = ax.pcolormesh(
        times,
        frequencies,
        Sxx_db,
        shading='gouraud',
        cmap='viridis'
    )
    
    cbar = plt.colorbar(im, ax=ax, label='Power (dB)')
    
    ax.set_xlabel('Time (hours from start)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (cycles/hour)', fontsize=12, fontweight='bold')
    ax.set_title(f'Spectrogram: {production_group} Production in {price_area}\n' +
                 f'Window: {window_length}h, Overlap: {window_overlap}h',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    plt.tight_layout()
    
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
    df = load_production_data()
    
    # Get unique values
    price_areas = sorted(df['priceArea'].unique())
    production_groups = sorted(df['productionGroup'].unique())
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 STL Decomposition", "🎵 Spectrogram"])
    
    # Tab 1: STL Decomposition
    with tab1:
        st.subheader("Seasonal-Trend Decomposition using LOESS (STL)")
        
        st.markdown("""
        STL decomposes a time series into three components:
        - **Trend**: Long-term progression
        - **Seasonal**: Repeating patterns (daily, weekly, etc.)
        - **Residual**: Remainder after removing trend and seasonal
        """)
        
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
                    st.pyplot(fig)
                    
                    st.markdown("---")
                    st.info("""
                    **Interpretation Guide:**
                    - **Trend**: Shows long-term changes in production
                    - **Seasonal**: Reveals daily/weekly patterns
                    - **Residual**: Contains irregular variations and noise
                    """)
    
    # Tab 2: Spectrogram
    with tab2:
        st.subheader("Spectrogram - Frequency-Time Analysis")
        
        st.markdown("""
        A spectrogram shows how frequency content changes over time, revealing:
        - **Daily cycles**: ~1/24 cycles/hour
        - **Weekly patterns**: ~1/168 cycles/hour
        - **Seasonal shifts**: Long-term frequency changes
        """)
        
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
                    st.pyplot(fig)
                    
                    st.markdown("---")
                    st.info("""
                    **Key Frequencies:**
                    - **Daily cycle**: 0.042 cycles/hour (1/24)
                    - **Weekly cycle**: 0.006 cycles/hour (1/168)
                    - **Brighter colors**: Higher power at that frequency/time
                    """)
    
    # Data source info
    st.markdown("---")
    with st.expander("ℹ️ About These Methods"):
        st.markdown("""
        ### STL Decomposition
        STL (Seasonal-Trend decomposition using LOESS) is a versatile and robust method for 
        decomposing time series. It handles any type of seasonality and is resistant to outliers 
        when robust=True.
        
        ### Spectrogram
        Uses Short-Time Fourier Transform (STFT) to show how the frequency spectrum evolves over time.
        Useful for identifying periodic patterns, detecting changes in production cycles, and 
        spotting anomalies in the frequency domain.
        
        **Data Source:** Elhub production data for 2021
        """)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)

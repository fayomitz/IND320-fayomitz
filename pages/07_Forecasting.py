import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_production_data, get_consumption_data, download_weather_data, render_sidebar_info
from datetime import timedelta

st.set_page_config(page_title="Forecasting", layout="wide")

# Render sidebar data info
render_sidebar_info()

# --- Main Page ---

st.title("🔮 Energy Forecasting (SARIMAX)")
st.markdown("Forecast energy production or consumption using Seasonal ARIMA with Exogenous variables.")

# 1. Data Selection & Loading
col_conf1, col_conf2 = st.columns(2)

with col_conf1:
    st.subheader("1. Select Data")
    data_type = st.selectbox("Data Type", ["Production", "Consumption"])
    
    if data_type == "Production":
        energy_df = get_production_data()
        group_col = 'productionGroup'
    else:
        energy_df = get_consumption_data()
        group_col = 'consumptionGroup'

    areas = sorted(energy_df['priceArea'].unique())
    
    # Get default area from session state (from map selector)
    default_area = st.session_state.get('selected_price_area', None)
    default_area_idx = 0
    if default_area and default_area in areas:
        default_area_idx = areas.index(default_area)
    
    sel_area = st.selectbox("Price Area", areas, index=default_area_idx)
    
    groups = sorted(energy_df[group_col].unique())
    sel_group = st.selectbox("Group", groups)

# 2. Timeframe & Horizon
with col_conf2:
    st.subheader("2. Timeframe & Horizon")
    
    # Get data range from session state (from map selector) or use defaults
    data_range = st.session_state.get('data_range', None)
    
    if data_range:
        # Use the years from map selector
        start_year, end_year = data_range
        default_start = pd.to_datetime(f"{start_year}-01-01").date()
        # Default training end is Dec 30th of end year (1 day before end to have test data)
        default_end = pd.to_datetime(f"{end_year}-12-30").date()
    else:
        # Fall back to energy data range
        if not energy_df.empty:
            default_start = energy_df['startTime_parsed'].min().date()
            default_end = energy_df['startTime_parsed'].max().date() - timedelta(days=1)
        else:
            default_start = pd.to_datetime("2021-01-01").date()
            default_end = pd.to_datetime("2024-12-30").date()
    
    # Determine min/max dates from energy data
    if not energy_df.empty:
        min_date = energy_df['startTime_parsed'].min().date()
        max_date = energy_df['startTime_parsed'].max().date()
    else:
        min_date = pd.to_datetime("2020-01-01").date()
        max_date = pd.to_datetime("2024-12-31").date()
    
    # Clamp default values to valid range
    default_start = max(default_start, min_date)
    default_end = min(default_end, max_date)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        train_start = st.date_input("Training Start", default_start, min_value=min_date, max_value=max_date)
    with col_t2:
        train_end = st.date_input("Training End", default_end, min_value=min_date, max_value=max_date)
    
    # Validate date range
    if train_start >= train_end:
        st.error("⚠️ Training start date must be before training end date.")
        st.stop()
    
    forecast_horizon = st.number_input("Forecast Horizon (Hours)", 1, 720, 24)

# 3. Prepare Data
energy_subset = energy_df[
    (energy_df['priceArea'] == sel_area) & 
    (energy_df[group_col] == sel_group)
].copy()

if energy_subset.empty:
    st.error("No data found for selected area and group.")
    st.stop()

# Resample Energy Data
energy_hourly = energy_subset.set_index('startTime_parsed')['quantityKwh'].resample('h').sum()
# Ensure the index has a frequency set (required for SARIMAX)
energy_hourly = energy_hourly.asfreq('h')

# Filter Training Data
train_mask = (energy_hourly.index.date >= train_start) & (energy_hourly.index.date <= train_end)
train_data_energy = energy_hourly[train_mask].copy()

# Ensure continuous index with frequency for SARIMAX
if not train_data_energy.empty:
    # Create a complete date range and reindex to ensure no gaps
    full_range = pd.date_range(
        start=train_data_energy.index.min(),
        end=train_data_energy.index.max(),
        freq='h',
        tz=train_data_energy.index.tz
    )
    train_data_energy = train_data_energy.reindex(full_range)
    # Fill any missing values (gaps in data)
    train_data_energy = train_data_energy.ffill().bfill()
    # Explicitly set frequency
    train_data_energy.index.freq = 'h'

if train_data_energy.empty:
    st.error("Training set is empty. Adjust dates.")
    st.stop()

# 4. Model Parameters
with st.expander("Model Configuration", expanded=True):
    with st.form("sarimax_params"):
        st.header("SARIMAX Parameters")
        
        # Exogenous Variables Selection
        # Hardcoded list of available weather variables from open-meteo
        AVAILABLE_WEATHER_VARS = [
            'temperature_2m (°C)', 
            'precipitation (mm)', 
            'wind_speed_10m (m/s)', 
            'wind_gusts_10m (m/s)', 
            'wind_direction_10m (°)'
        ]
        
        selected_exog = st.multiselect("Exogenous Variables (Weather)", AVAILABLE_WEATHER_VARS, default=[])
        
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            st.markdown("**Non-seasonal**")
            p = st.number_input("p (AR order)", 0, 5, 1, help="Autoregressive order - number of lag observations")
            d = st.number_input("d (Differencing)", 0, 2, 1, help="Degree of differencing for stationarity")
            q = st.number_input("q (MA order)", 0, 5, 1, help="Moving average order - size of moving average window")
        with col_p2:
            st.markdown("**Seasonal**")
            P = st.number_input("P (Seasonal AR)", 0, 2, 1, help="Seasonal autoregressive order")
            D = st.number_input("D (Seasonal I)", 0, 2, 1, help="Seasonal differencing order")
            Q = st.number_input("Q (Seasonal MA)", 0, 2, 1, help="Seasonal moving average order")
        with col_p3:
            st.markdown("**Season & Trend**")
            s = st.number_input("s (Season length)", 1, 168, 24, help="Number of periods in a season (24 for daily seasonality in hourly data)")
            trend = st.selectbox("Trend", [None, 'c', 't', 'ct'], 
                                format_func=lambda x: {'c': 'Constant', 't': 'Linear', 'ct': 'Constant + Linear', None: 'None'}[x],
                                help="Trend component: None, Constant (c), Linear (t), or both (ct)")
        with col_p4:
            st.markdown("**Forecast Settings**")
            show_training_data = st.checkbox("Show Training Data on Graph", value=False,
                                            help="Display the training data on the forecast graph")
        
        train_btn = st.form_submit_button("Train & Forecast")

# Coordinate Mapping for Price Areas (Approximate Centers)
AREA_COORDINATES = {
    "NO1": (59.91, 10.75), # Oslo
    "NO2": (58.15, 8.0),   # Kristiansand
    "NO3": (63.43, 10.39), # Trondheim
    "NO4": (69.65, 18.96), # Tromsø
    "NO5": (60.39, 5.32)   # Bergen
}

# 5. Train & Forecast Logic
if train_btn:
    with st.spinner("Preparing data and training model..."):
        try:
            # --- Weather Data Handling ---
            weather_hourly = None
            
            if selected_exog:
                # Determine required range
                req_start_year = train_start.year
                # Forecast end time
                last_train_time = train_data_energy.index.max()
                forecast_end_time = last_train_time + timedelta(hours=forecast_horizon)
                req_end_year = forecast_end_time.year
                
                # Check if we need to download
                need_download = True
                stored_area = st.session_state.get('weather_data_area', None)
                
                if 'weather_data' in st.session_state and st.session_state.weather_data is not None:
                    w_df = st.session_state.weather_data
                    w_df['time'] = pd.to_datetime(w_df['time'], utc=True)
                    
                    # Check range coverage AND area match
                    if (w_df['time'].dt.year.min() <= req_start_year and 
                        w_df['time'].dt.year.max() >= req_end_year and
                        stored_area == sel_area):
                        need_download = False
                        weather_hourly = w_df.set_index('time').resample('h').mean()
                
                if need_download:
                    lat, lon = AREA_COORDINATES.get(sel_area, (59.91, 10.75)) # Default to Oslo
                    st.info(f"Downloading weather data for {sel_area} ({req_start_year}-{req_end_year})...")
                    w_df = download_weather_data(lat, lon, req_start_year, req_end_year)
                    st.session_state.weather_data = w_df
                    st.session_state.weather_data_area = sel_area
                    weather_hourly = w_df.set_index('time').resample('h').mean()

            # --- Prepare Training Data with Exog ---
            # Merge Energy and Weather
            
            # Construct Training Set
            y_train = train_data_energy
            exog_train = None
            exog_forecast = None
            
            if selected_exog and weather_hourly is not None:
                # Align weather to training data
                # We use reindex to ensure exact match, filling missing with interpolation if needed
                exog_cols = [col for col in selected_exog if col in weather_hourly.columns]
                if not exog_cols:
                    st.warning("Selected weather variables not found in data.")
                    exog_train = None
                    exog_forecast = None
                else:
                    # Ensure timezone alignment between weather and energy data
                    weather_idx = weather_hourly.index
                    train_idx = y_train.index
                    
                    # Convert both to UTC for alignment if needed
                    if weather_idx.tz is None and train_idx.tz is not None:
                        weather_hourly.index = weather_hourly.index.tz_localize('UTC')
                    elif weather_idx.tz is not None and train_idx.tz is None:
                        weather_hourly.index = weather_hourly.index.tz_localize(None)
                    elif weather_idx.tz is not None and train_idx.tz is not None:
                        weather_hourly.index = weather_hourly.index.tz_convert(train_idx.tz)
                    
                    exog_train = weather_hourly.reindex(y_train.index)[exog_cols]
                    
                    # Handle missing values in exog (if any) - using non-deprecated syntax
                    if exog_train.isnull().any().any():
                        exog_train = exog_train.interpolate(method='linear').bfill().ffill()
                    
                    # Replace any remaining inf values
                    exog_train = exog_train.replace([np.inf, -np.inf], np.nan)
                    exog_train = exog_train.ffill().bfill()
                    
                    # Final check - if still has NaN, fill with column means
                    if exog_train.isnull().any().any():
                        for col in exog_cols:
                            col_mean = exog_train[col].mean()
                            if pd.isna(col_mean):
                                col_mean = 0  # Fallback to 0 if entire column is NaN
                            exog_train[col] = exog_train[col].fillna(col_mean)
                    
                    # Explicitly set frequency for exog_train
                    if hasattr(y_train.index, 'freq'):
                        exog_train.index.freq = y_train.index.freq

                    # Prepare Forecast Exog
                    last_train_time = y_train.index.max()
                    forecast_index = pd.date_range(
                        start=last_train_time + timedelta(hours=1), 
                        periods=forecast_horizon, 
                        freq='h',
                        tz=y_train.index.tz  # Match timezone
                    )
                    
                    exog_forecast = weather_hourly.reindex(forecast_index)[exog_cols]
                    
                    # Check if we have enough future weather data
                    if exog_forecast.isnull().any().any():
                        st.warning("Weather data incomplete for forecast horizon. Filling with last known values.")
                        exog_forecast = exog_forecast.interpolate(method='linear').ffill().bfill()
                    
                    # Replace any inf values and final NaN cleanup
                    exog_forecast = exog_forecast.replace([np.inf, -np.inf], np.nan)
                    exog_forecast = exog_forecast.ffill().bfill()
                    
                    # Use training data means as fallback for forecast
                    if exog_forecast.isnull().any().any():
                        for col in exog_cols:
                            col_mean = exog_train[col].mean()
                            exog_forecast[col] = exog_forecast[col].fillna(col_mean)
                    
                    # Explicitly set frequency for exog_forecast
                    exog_forecast.index.freq = 'h'

            # Fit Model
            model = sm.tsa.statespace.SARIMAX(
                y_train,
                exog=exog_train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=False)

            # Forecast
            forecast = results.get_forecast(steps=forecast_horizon, exog=exog_forecast)
            pred_mean = forecast.predicted_mean
            pred_ci = forecast.conf_int()
            
            try:
                # Get actual energy data for forecast period if available
                last_train_time = y_train.index.max()
                
                forecast_start_time = pd.Timestamp(last_train_time) + pd.Timedelta(hours=1)
                forecast_end_time = pd.Timestamp(last_train_time) + pd.Timedelta(hours=forecast_horizon)
                
                energy_future = energy_hourly[
                    (energy_hourly.index >= forecast_start_time) & 
                    (energy_hourly.index <= forecast_end_time)
                ]
                has_validation = not energy_future.empty

                # Plotting - convert all indices to Python datetime to avoid pandas Timestamp issues
                fig = go.Figure()
                
                # Convert indices to Python datetime for plotly compatibility
                pred_mean_x = pd.to_datetime(pred_mean.index).to_pydatetime().tolist()

                # Training Data (observed) - only if toggle is enabled
                if show_training_data:
                    y_train_x = pd.to_datetime(y_train.index).to_pydatetime().tolist()
                    fig.add_trace(go.Scatter(
                        x=y_train_x, y=y_train.values, 
                        name="Observed (Training)", 
                        mode='lines',
                        line=dict(color='blue', width=1)
                    ))
                
                # Validation Data (if available)
                if has_validation:
                    energy_future_x = pd.to_datetime(energy_future.index).to_pydatetime().tolist()
                    fig.add_trace(go.Scatter(
                        x=energy_future_x, y=energy_future.values, 
                        name="Observed (Test)", 
                        mode='markers',
                        marker=dict(color='green', size=4)
                    ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=pred_mean_x, y=pred_mean.values, 
                    name="Dynamic Forecast", 
                    mode='lines',
                    line=dict(color='red', width=2)
                ))
                
                # Confidence Interval for forecast
                pred_ci_x = pd.to_datetime(pred_ci.index).to_pydatetime().tolist()
                pred_ci_x_combined = pred_ci_x + pred_ci_x[::-1]
                pred_ci_y_combined = list(pred_ci.iloc[:, 0].values) + list(pred_ci.iloc[:, 1].values[::-1])
                
                fig.add_trace(go.Scatter(
                    x=pred_ci_x_combined,
                    y=pred_ci_y_combined,
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.15)',
                    line=dict(color='rgba(255,0,0,0)'),
                    name="95% Confidence Interval",
                    hoverinfo='skip'
                ))
                
                # Add vertical line at forecast start using shapes (avoids datetime arithmetic issue)
                vline_x = pd.Timestamp(y_train.index.max()).to_pydatetime()
                fig.add_shape(
                    type="line",
                    x0=vline_x,
                    x1=vline_x,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="gray", dash="dash", width=1)
                )
                # Add annotation separately
                fig.add_annotation(
                    x=vline_x,
                    y=1,
                    yref="paper",
                    text="Forecast Start",
                    showarrow=False,
                    yshift=10
                )
                
                fig.update_layout(
                    title=f"SARIMAX({p},{d},{q})({P},{D},{Q},{s}) Forecast - {data_type} ({sel_group})",
                    xaxis_title="Time",
                    yaxis_title="Energy (kWh)",
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics (if validation available)
                if has_validation:
                    # Intersect indices to calculate metrics
                    common_indices = pred_mean.index.intersection(energy_future.index)
                    if not common_indices.empty:
                        y_true = energy_future.loc[common_indices]
                        y_pred = pred_mean.loc[common_indices]
                        
                        mae = np.mean(np.abs(y_pred - y_true))
                        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
                        
                        col_m1, col_m2 = st.columns(2)
                        col_m1.metric("MAE", f"{mae:.2f}")
                        col_m2.metric("RMSE", f"{rmse:.2f}")
                    else:
                        st.info("Forecast period does not overlap with available validation data.")
                else:
                    st.info("No actual data available for this period (Future Forecast).")
            
            except Exception as plot_err:
                st.error(f"Plotting failed: {plot_err}")
                import traceback
                st.text(traceback.format_exc())

        except Exception as e:
            st.error(f"Model training failed: {e}")
            import traceback
            st.text(traceback.format_exc())

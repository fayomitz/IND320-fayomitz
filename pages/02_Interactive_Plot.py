import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import render_sidebar_info, check_data_requirements

# Page configuration
st.set_page_config(
    page_title="Interactive Plot & Data",
    layout="wide"
)

# Render sidebar data info
render_sidebar_info()

# Load weather data from session state
def get_weather_data():
    """Get weather data from session state (set by page 2)"""
    if 'weather_data' in st.session_state and st.session_state.weather_data is not None:
        return st.session_state.weather_data
    return None

def get_unique_months(df):
    """Get list of unique year-month combinations from the dataframe"""
    df['year_month'] = df['time'].dt.to_period('M')
    months = df['year_month'].unique()
    return sorted([str(m) for m in months])

def get_first_month_data(df, column):
    """Extract the first month of data for a specific column"""
    first_month = df[df['time'].dt.month == df['time'].dt.month.iloc[0]]
    first_month = first_month[first_month['time'].dt.year == df['time'].dt.year.iloc[0]]
    return first_month[['time', column]].head(744)  # First month (up to 31 days * 24 hours)

# Main page content
st.title("Weather Data Analysis")
st.header("Explore and Visualize Weather Data")

# Check if weather data is available
if not check_data_requirements(require_weather=True):
    st.info("Once you download data on the Map Selector page, it will be available here for interactive plotting.")
    st.stop()

data = get_weather_data()

if data is not None:
    st.success(f"✅ Weather data loaded: {len(data)} records")
    if 'selected_price_area' in st.session_state:
        st.info(f"📍 Data for: **{st.session_state.selected_price_area}**")
    
    tab1, tab2 = st.tabs(["Interactive Plot", "Data Overview"])
    
    with tab1:
        st.subheader("Interactive Visualization")
        st.markdown("Use the controls below to customize your view of the weather data.")
        
        try:
            # Get all columns except 'time' and 'year_month' (if it exists)
            data_columns = [col for col in data.columns if col not in ['time', 'year_month']]
            
            # Create two columns for the controls
            col1, col2 = st.columns(2)
            
            with col1:
                # Multi-select for columns with "Select All" option
                select_all = st.checkbox("Select All Variables", value=False, help="Check to plot all variables")
                
                if select_all:
                    selected_columns = data_columns
                else:
                    selected_columns = st.multiselect(
                        "Select Variables to Plot",
                        options=data_columns,
                        default=[data_columns[0]] if data_columns else [],
                        help="Choose one or more variables to plot"
                    )
            
            with col2:
                # Get available months
                available_months = get_unique_months(data)
                
                # Selection slider for months (range selection)
                if len(available_months) > 1:
                    selected_months = st.select_slider(
                        "Select Month Range",
                        options=available_months,
                        value=(available_months[0], available_months[-1]),
                        help="Select the range of months to display"
                    )
                else:
                    selected_months = (available_months[0], available_months[0])
                    st.info(f"Showing data for: {selected_months[0]}")
            
            st.markdown("---")
            
            # Filter data based on selected months
            data['year_month'] = data['time'].dt.to_period('M')
            
            # Handle month range (tuple from select_slider)
            if isinstance(selected_months, tuple):
                start_period = pd.Period(selected_months[0], freq='M')
                end_period = pd.Period(selected_months[1], freq='M')
                filtered_data = data[
                    (data['year_month'] >= start_period) & 
                    (data['year_month'] <= end_period)
                ].copy()
                month_label = f"{selected_months[0]} to {selected_months[1]}"
            else:
                # Single month selected (fallback)
                selected_period = pd.Period(selected_months, freq='M')
                filtered_data = data[data['year_month'] == selected_period].copy()
                month_label = selected_months
            
            # Drop the year_month column to avoid plotting issues
            filtered_data = filtered_data.drop('year_month', axis=1)
            
            # Check if any columns are selected
            if not selected_columns:
                st.warning("Please select at least one variable to plot.")
                st.stop()
            
            # Create the plot
            if len(selected_columns) > 1:
                # Plot multiple columns
                melted_df = filtered_data.melt(id_vars=['time'], value_vars=selected_columns, 
                                             var_name='Variable', value_name='Value')
                fig = px.line(melted_df, x='time', y='Value', color='Variable',
                              title=f"Weather Variables - {month_label}")
            else:
                # Plot single column
                fig = px.line(filtered_data, x='time', y=selected_columns[0],
                              title=f"{selected_columns[0]} - {month_label}")
            
            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            # Display summary statistics
            st.markdown("---")
            st.subheader("Summary Statistics for Selected Period")
            
            if len(selected_columns) > 1:
                summary_df = filtered_data[selected_columns].describe().T
                summary_df = summary_df[['mean', 'std', 'min', 'max']]
                summary_df.columns = ['Mean', 'Std Dev', 'Min', 'Max']
                st.dataframe(summary_df.round(2), width='stretch')
            else:
                col_stats = filtered_data[selected_columns[0]].describe()
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    st.metric("Mean", f"{col_stats['mean']:.2f}")
                with stats_col2:
                    st.metric("Std Dev", f"{col_stats['std']:.2f}")
                with stats_col3:
                    st.metric("Min", f"{col_stats['min']:.2f}")
                with stats_col4:
                    st.metric("Max", f"{col_stats['max']:.2f}")
            
            st.info(f"Displaying {len(filtered_data):,} data points for the selected period")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

    with tab2:
        st.subheader("Dataset Overview with First Month Trends")
        st.markdown("Each row represents a variable from the dataset, with a line chart showing the first month of data.")
        
        try:
            # Get all numeric columns except 'time'
            data_columns = [col for col in data.columns if col != 'time' and pd.api.types.is_numeric_dtype(data[col])]
            
            # Create a dataframe to display with line charts
            table_data = []
            
            for column in data_columns:
                # Get basic statistics for the column
                col_mean = data[column].mean()
                col_std = data[column].std()
                col_min = data[column].min()
                col_max = data[column].max()
                
                # Get first month data for the chart
                first_month = get_first_month_data(data, column)
                
                table_data.append({
                    'Variable': column,
                    'Mean': f"{col_mean:.2f}",
                    'Std Dev': f"{col_std:.2f}",
                    'Min': f"{col_min:.2f}",
                    'Max': f"{col_max:.2f}",
                    'First Month Data': first_month[column].tolist()
                })
            
            # Create DataFrame for display
            display_df = pd.DataFrame(table_data)
            
            # Display the table with line chart column
            st.dataframe(
                display_df,
                column_config={
                    "Variable": st.column_config.TextColumn("Variable Name", width="medium"),
                    "Mean": st.column_config.TextColumn("Mean", width="small"),
                    "Std Dev": st.column_config.TextColumn("Std Dev", width="small"),
                    "Min": st.column_config.TextColumn("Min", width="small"),
                    "Max": st.column_config.TextColumn("Max", width="small"),
                    "First Month Data": st.column_config.LineChartColumn(
                        "First Month Trend",
                        width="large",
                        help="Visualization of the first month of data"
                    ),
                },
                hide_index=True,
                width='stretch'
            )
            
            st.info(f"Showing statistics and first month trends for {len(data_columns)} variables across {len(data):,} total records")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

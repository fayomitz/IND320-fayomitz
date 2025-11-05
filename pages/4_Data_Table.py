import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Data Table - Weather Data",
    layout="wide"
)

# Load weather data from session state
def get_weather_data():
    """Get weather data from session state (set by page 2)"""
    if 'weather_data' in st.session_state and st.session_state.weather_data is not None:
        return st.session_state.weather_data
    return None

def get_first_month_data(df, column):
    """Extract the first month of data for a specific column"""
    first_month = df[df['time'].dt.month == df['time'].dt.month.iloc[0]]
    first_month = first_month[first_month['time'].dt.year == df['time'].dt.year.iloc[0]]
    return first_month[['time', column]].head(744)  # First month (up to 31 days * 24 hours)

# Main page content
st.title("Data Table View")
st.header("Weather Data Overview with First Month Visualization")

st.markdown("""
This page displays the weather data in a tabular format with a line chart showing 
the first month of data for each variable.
""")

st.markdown("---")

# Check if weather data is available
data = get_weather_data()

if data is None:
    st.warning("⚠️ No weather data loaded. Please visit the **Price Area Selector** page first to download weather data.")
    st.info("Once you download data on that page, it will be available here for viewing.")
else:
    st.success(f"✅ Weather data loaded: {len(data)} records")
    if 'selected_area' in st.session_state:
        st.info(f"📍 Data for: **{st.session_state.selected_area}** ({st.session_state.selected_city})")
    
    try:
        st.subheader("Dataset with First Month Line Charts")
        st.markdown("Each row represents a variable from the dataset, with a line chart showing the first month of data.")
        
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
        
        st.markdown("---")
        st.info(f"Showing statistics and first month trends for {len(data_columns)} variables across {len(data):,} total records")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

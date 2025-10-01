import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Data Table - Weather Data",
    layout="wide"
)

# Cache the data loading function for performance
@st.cache_data
def load_data():
    """Load the weather data from CSV file with caching for app speed"""
    df = pd.read_csv('open-meteo-subset.csv')
    df['time'] = pd.to_datetime(df['time'])
    return df

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

try:
    # Load the data
    data = load_data()
    
    st.subheader("Dataset with First Month Line Charts")
    st.markdown("Each row represents a variable from the dataset, with a line chart showing the first month of data.")
    
    # Get all columns except 'time'
    data_columns = [col for col in data.columns if col != 'time']
    
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
        use_container_width=True
    )
    
    st.markdown("---")
    st.info(f"Showing statistics and first month trends for {len(data_columns)} variables across {len(data):,} total records")
    
except FileNotFoundError:
    st.error("Data file 'open-meteo-subset.csv' not found. Please ensure the file is in the same directory as this app.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

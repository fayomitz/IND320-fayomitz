import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Interactive Plot - Weather Data",
    layout="wide"
)

# Cache the data loading function for performance
@st.cache_data
def load_data():
    """Load the weather data from CSV file with caching for app speed"""
    df = pd.read_csv('open-meteo-subset.csv')
    df['time'] = pd.to_datetime(df['time'])
    return df

def get_unique_months(df):
    """Get list of unique year-month combinations from the dataframe"""
    df['year_month'] = df['time'].dt.to_period('M')
    months = df['year_month'].unique()
    return sorted([str(m) for m in months])

# Main page content
st.title("Interactive Weather Data Plot")
st.header("Explore the Data with Custom Visualizations")

st.markdown("""
Use the controls below to customize your view of the weather data.
""")

st.markdown("---")

try:
    # Load the data
    data = load_data()
    
    # Get all columns except 'time'
    data_columns = [col for col in data.columns if col != 'time']
    
    # Create two columns for the controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Dropdown menu to select columns
        column_options = ['All Columns'] + data_columns
        selected_column = st.selectbox(
            "Select Variable to Plot",
            options=column_options,
            help="Choose a specific variable or plot all variables together"
        )
    
    with col2:
        # Get available months
        available_months = get_unique_months(data)
        
        # Selection slider for months
        if len(available_months) > 1:
            selected_months = st.select_slider(
                "Select Month Range",
                options=available_months,
                value=available_months[0],
                help="Select the range of months to display"
            )
        else:
            selected_months = available_months[0]
            st.info(f"Showing data for: {selected_months}")
    
    st.markdown("---")
    
    # Filter data based on selected months
    data['year_month'] = data['time'].dt.to_period('M')
    
    # Handle single month or range
    if isinstance(selected_months, str):
        # Single month selected
        filtered_data = data[data['year_month'] == selected_months]
    else:
        # Range of months (if using both values from select_slider)
        filtered_data = data[data['year_month'] == selected_months]
    
    # Create the plot
    st.subheader("Weather Data Visualization")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if selected_column == 'All Columns':
        # Plot all columns
        for column in data_columns:
            ax.plot(filtered_data['time'], filtered_data[column], label=column, linewidth=1.5, alpha=0.7)
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        plot_title = f"All Weather Variables - {selected_months}"
    else:
        # Plot selected column
        ax.plot(filtered_data['time'], filtered_data[selected_column], 
                linewidth=2, color='#1f77b4', label=selected_column)
        ax.set_ylabel(selected_column, fontsize=12, fontweight='bold')
        plot_title = f"{selected_column} - {selected_months}"
    
    # Format the plot
    ax.set_title(plot_title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    # Display summary statistics
    st.markdown("---")
    st.subheader("Summary Statistics for Selected Period")
    
    if selected_column == 'All Columns':
        summary_df = filtered_data[data_columns].describe().T
        summary_df = summary_df[['mean', 'std', 'min', 'max']]
        summary_df.columns = ['Mean', 'Std Dev', 'Min', 'Max']
        st.dataframe(summary_df.round(2), use_container_width=True)
    else:
        col_stats = filtered_data[selected_column].describe()
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
    
except FileNotFoundError:
    st.error("Data file 'open-meteo-subset.csv' not found. Please ensure the file is in the same directory as this app.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)

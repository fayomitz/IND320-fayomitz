import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Page configuration
st.set_page_config(
    page_title="Energy Production Data",
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

# Main page content
st.title("Energy Production Data Analysis")
st.header("Norwegian Electricity Production by Price Area and Production Group")

st.markdown("""
Explore the electricity production data across different price areas in Norway for 2021.
Select a price area and production groups to visualize the data.
""")

st.markdown("---")

try:
    # Load the data
    df = load_production_data()
    
    # Get unique price areas and production groups
    price_areas = sorted(df['priceArea'].unique())
    production_groups = sorted(df['productionGroup'].unique())
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Yearly Production Distribution")
        
        # Radio buttons to select price area
        selected_price_area = st.radio(
            "Select Price Area",
            options=price_areas,
            help="Choose a Norwegian price area to analyze"
        )
        
        # Filter data for selected price area
        price_area_data = df[df['priceArea'] == selected_price_area]
        
        # Calculate total production by group for the year
        yearly_production = price_area_data.groupby('productionGroup')['quantityKwh'].sum()
        
        # Create pie chart
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.pie(yearly_production, labels=yearly_production.index, autopct='%1.1f%%', 
                startangle=90, textprops={'fontsize': 10})
        ax1.set_title(f'Total Production by Group for {selected_price_area} in 2021', 
                      fontsize=12, fontweight='bold', pad=20)
        
        st.pyplot(fig1)
        
        # Display total production
        total_production = yearly_production.sum()
        st.metric("Total Production", f"{total_production:,.0f} kWh")
    
    with col2:
        st.subheader("Monthly Production Trends")
        
        # Pills to select production groups
        selected_groups = st.pills(
            "Select Production Group(s)",
            options=production_groups,
            selection_mode="multi",
            default=production_groups[0] if production_groups else None,
            help="Choose one or more production groups to display"
        )
        
        # Selectbox for month
        selected_month = st.selectbox(
            "Select Month",
            options=months,
            index=0,
            help="Choose a month to analyze"
        )
        
        # Convert selected month to month number
        month_num = months.index(selected_month) + 1
        
        # Filter data for selected price area, groups, and month
        if selected_groups:
            # Handle both single and multiple selections
            if isinstance(selected_groups, str):
                selected_groups = [selected_groups]
            
            month_data = df[(df['priceArea'] == selected_price_area) & 
                           (df['productionGroup'].isin(selected_groups)) &
                           (df['startTime_parsed'].dt.month == month_num) & 
                           (df['startTime_parsed'].dt.year == 2021)]
            
            # Create line plot
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            for group in selected_groups:
                group_data = month_data[month_data['productionGroup'] == group]
                group_data_sorted = group_data.sort_values('startTime_parsed')
                ax2.plot(group_data_sorted['startTime_parsed'], 
                        group_data_sorted['quantityKwh'], 
                        label=group, 
                        marker='o', 
                        markersize=2,
                        linewidth=1.5,
                        alpha=0.8)
            
            ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Production (kWh)', fontsize=11, fontweight='bold')
            ax2.set_title(f'Hourly Production for {selected_price_area} - {selected_month} 2021', 
                         fontsize=12, fontweight='bold', pad=15)
            ax2.legend(loc='best', fontsize=9)
            ax2.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig2)
            
            # Display stats
            if len(month_data) > 0:
                avg_production = month_data.groupby('productionGroup')['quantityKwh'].mean()
                st.write("**Average Hourly Production:**")
                for group in selected_groups:
                    if group in avg_production.index:
                        st.write(f"- {group}: {avg_production[group]:,.0f} kWh")
            else:
                st.warning("No data available for the selected combination.")
        else:
            st.info("Please select at least one production group to display the line plot.")
    
    # Expander for data source documentation
    st.markdown("---")
    with st.expander("Data Source Information"):
        st.markdown("""
        ### Data Source
        
        **Source:** Elhub - Norwegian Energy Data Hub  
        **API Endpoint:** `https://api.elhub.no/energy-data/v0/price-areas`  
        **Dataset:** Production per Group MBA (Metering Balance Area) per Hour
        
        **Description:**  
        This dataset contains hourly electricity production data for Norway, broken down by:
        - **Price Areas:** Different geographical pricing zones in Norway (NO1-NO5)
        - **Production Groups:** Various types of electricity generation (e.g., hydro, wind, thermal)
        - **Time Period:** Full year 2021 (January 1 - December 31, 2021)
        
        **Data Fields:**
        - `priceArea`: Norwegian price area identifier
        - `productionGroup`: Type of energy production
        - `startTime` / `endTime`: Time period for the measurement
        - `quantityKwh`: Amount of electricity produced in kilowatt-hours
        - `lastUpdatedTime`: Timestamp of last data update
        
        **Data Collection:**  
        The data was retrieved from the Elhub API and stored in a MongoDB database for analysis and visualization.
        
        **Note:** All production values are measured in kilowatt-hours (kWh) and represent actual generation data 
        reported to the Norwegian energy system.
        """)
    
except Exception as e:
    st.error(f"An error occurred while loading the data: {str(e)}")
    st.info("Make sure your MongoDB connection is properly configured in Streamlit secrets.")
    st.exception(e)

import streamlit as st
import pandas as pd
import json
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
from shapely.geometry import shape, Point
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Map & Data Selector",
    layout="wide"
)

# MongoDB connection
@st.cache_resource
def init_connection():
    """Initialize MongoDB connection using secrets"""
    try:
        uri = st.secrets["URI"]
        client = MongoClient(uri, server_api=ServerApi('1'))
        return client
    except Exception:
        return None

# Fetch energy stats from MongoDB
@st.cache_data(ttl=3600)
def fetch_energy_stats(start_date_str, end_date_str, group):
    """
    Fetch aggregated energy stats for all price areas for the given time range and group.
    Returns a dictionary {area_name: mean_value}
    """
    client = init_connection()
    if not client:
        return {}
    
    try:
        db = client['energy_data']
        collection_name = 'production' if group == 'Production' else 'consumption'
        collection = db[collection_name]
        
        pipeline = [
            {
                "$match": {
                    "startTime": {"$gte": start_date_str, "$lte": end_date_str}
                }
            },
            {
                "$group": {
                    "_id": "$priceArea",
                    "mean_value": {"$avg": "$quantityKwh"}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        stats = {res['_id']: res['mean_value'] for res in results}
        return stats
        
    except Exception as e:
        st.error(f"Database error: {e}")
        return {}

# Initialize session state variables
if 'selected_coordinates' not in st.session_state:
    st.session_state.selected_coordinates = None
if 'selected_price_area' not in st.session_state:
    st.session_state.selected_price_area = None
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 5
if 'map_center' not in st.session_state:
    st.session_state.map_center = [65.0, 13.0]
elif isinstance(st.session_state.map_center, dict):
    st.session_state.map_center = [st.session_state.map_center['lat'], st.session_state.map_center['lon']]
if 'last_clicked_processed' not in st.session_state:
    st.session_state.last_clicked_processed = None

# Load GeoJSON data
@st.cache_data
def load_geojsons():
    price_area_path = 'ElSpot_omraade.geojson'
    kommune_path = 'Basisdata_0000_Norge_4258_Kommune_GeoJSON.geojson'
    
    data = {}
    if os.path.exists(price_area_path):
        with open(price_area_path, 'r', encoding='utf-8-sig') as f:
            data['price_area'] = json.load(f)
            
            # Pre-process: ensure IDs are set
            for feature in data['price_area']['features']:
                props = feature['properties']
                name = props.get('ElSpotOmr') or props.get('omrnavn') or props.get('navn') or props.get('name')
                if name:
                    name = name.replace(" ", "")
                feature['id'] = name
                props['name'] = name
            
    if os.path.exists(kommune_path):
        with open(kommune_path, 'r', encoding='utf-8-sig') as f:
            data['kommune'] = json.load(f)
            
            # Pre-process: Map Municipalities to Price Areas for coloring
            price_polygons = []
            if 'price_area' in data:
                for feature in data['price_area']['features']:
                    poly = shape(feature['geometry'])
                    name = feature.get('id')
                    price_polygons.append((poly, name))
            
            # Map each municipality to its price area
            if 'features' in data['kommune']:
                for i, feature in enumerate(data['kommune']['features']):
                    props = feature.get('properties', {})
                    
                    if 'id' not in feature:
                        feature['id'] = props.get('kommunenummer', str(i))
                    
                    try:
                        geom = shape(feature['geometry'])
                        centroid = geom.centroid
                        for poly, area_name in price_polygons:
                            if poly.contains(centroid):
                                props['price_area'] = area_name
                                break
                    except Exception:
                        pass
            
    return data

# Helper function to find which price area a point is in
def find_price_area_for_point(lat, lon, price_area_data):
    """Find which price area a point is in"""
    point = Point(lon, lat)  # Point takes (x, y) = (lon, lat)
    
    if price_area_data:
        for feature in price_area_data['features']:
            try:
                poly = shape(feature['geometry'])
                if poly.contains(point):
                    return feature.get('id')
            except Exception:
                pass
    return None

from utils import download_weather_data, render_sidebar_info

# Render sidebar data info
render_sidebar_info()

# Main Layout
st.title("📍 Map & Data Selector")
st.markdown("Click anywhere on the map to select a location and its Price Area.")

col_controls, col_map = st.columns([1, 2])

geojson_data = load_geojsons()
price_area_data = geojson_data.get('price_area')
kommune_data = geojson_data.get('kommune')

with col_controls:
    st.subheader("1. Map Configuration")
    st.caption("Configure the energy data displayed on the map.")
    
    # Energy Group Selector
    energy_group = st.selectbox(
        "Energy Group",
        options=["Production", "Consumption"],
        help="Select energy data type to visualize on the map"
    )
    
    # Time Interval Selector
    time_interval = st.number_input(
        "Time Interval (Days)",
        min_value=1,
        max_value=365,
        value=30,
        help="Interval for calculating mean energy values"
    )
    
    # Analysis Date Selector
    default_date = datetime(2024, 1, 1).date()
    analysis_date = st.date_input(
        "Map Data Start Date",
        value=default_date,
        help="Select the start date for the map visualization interval"
    )

with col_map:
    st.subheader("Interactive Map")
    st.caption("Click to select location. Zoom in to see municipalities, zoom out for price areas.")
    
    # Calculate date range for stats
    stats_start_date = analysis_date
    stats_end_date = stats_start_date + timedelta(days=time_interval)
    
    stats_start_str = stats_start_date.strftime("%Y-%m-%dT00:00:00Z")
    stats_end_str = stats_end_date.strftime("%Y-%m-%dT23:59:59Z")
    
    # Fetch stats
    energy_stats = fetch_energy_stats(stats_start_str, stats_end_str, energy_group)
    
    # Get min/max for color scale
    if energy_stats:
        min_val = min(energy_stats.values())
        max_val = max(energy_stats.values())
    else:
        min_val, max_val = 0, 100
    
    # Create colormap (for styling polygons)
    colormap = cm.LinearColormap(
        colors=['#FFEDA0', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026'],
        vmin=min_val,
        vmax=max_val
    )
    
    # Create a custom vertical legend HTML
    def create_vertical_legend(min_v, max_v, group_name):
        """Create a nice vertical color legend"""
        colors = ['#BD0026', '#E31A1C', '#FC4E2A', '#FD8D3C', '#FEB24C', '#FED976', '#FFEDA0']
        n_colors = len(colors)
        
        # Format values nicely
        def fmt(v):
            if v >= 1000000:
                return f'{v/1000000:.1f}M'
            elif v >= 1000:
                return f'{v/1000:.0f}k'
            else:
                return f'{v:.0f}'
        
        gradient_stops = ', '.join([f'{c} {i*100//(n_colors-1)}%' for i, c in enumerate(colors)])
        
        legend_html = f'''
        <div style="
            position: fixed;
            bottom: 50px;
            right: 20px;
            z-index: 1000;
            background: white;
            padding: 12px 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            font-family: Arial, sans-serif;
            font-size: 12px;
        ">
            <div style="font-weight: bold; margin-bottom: 8px; text-align: center; color: #333;">
                Mean {group_name}<br><span style="font-size: 10px; color: #666;">(kWh)</span>
            </div>
            <div style="display: flex; align-items: stretch;">
                <div style="
                    width: 20px;
                    height: 150px;
                    background: linear-gradient(to bottom, {gradient_stops});
                    border-radius: 3px;
                    border: 1px solid #ccc;
                "></div>
                <div style="
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    margin-left: 8px;
                    padding: 2px 0;
                    color: #444;
                ">
                    <span>{fmt(max_v)}</span>
                    <span>{fmt((max_v + min_v) / 2)}</span>
                    <span>{fmt(min_v)}</span>
                </div>
            </div>
        </div>
        '''
        return legend_html
    
    # Create Folium map
    map_center = st.session_state.map_center
    m = folium.Map(
        location=map_center,
        zoom_start=st.session_state.map_zoom,
        tiles='cartodbpositron'
    )
    
    # Get selected price area for highlighting
    selected_area = st.session_state.selected_price_area
    
    # Zoom threshold for switching layers
    ZOOM_THRESHOLD = 7
    current_zoom = st.session_state.map_zoom
    
    # Style function for Price Areas - highlight selected area
    def price_area_style(feature):
        area_id = feature.get('id') or feature['properties'].get('name')
        value = energy_stats.get(area_id, 0)
        is_selected = (area_id == selected_area)
        
        return {
            'fillColor': colormap(value) if value else '#gray',
            'color': '#0000FF' if is_selected else '#000000',  # Blue outline if selected
            'weight': 4 if is_selected else 2,
            'fillOpacity': 0.6 if is_selected else 0.5
        }
    
    # Style function for Municipalities
    def municipality_style(feature):
        props = feature.get('properties', {})
        price_area = props.get('price_area')
        value = energy_stats.get(price_area, 0) if price_area else 0
        is_selected = (price_area == selected_area)
        
        return {
            'fillColor': colormap(value) if value else '#gray',
            'color': '#0000FF' if is_selected else '#333333',  # Blue outline if in selected area
            'weight': 2 if is_selected else 1,
            'fillOpacity': 0.6 if is_selected else 0.5
        }
    
    # Conditionally add layer based on zoom level
    if current_zoom >= ZOOM_THRESHOLD:
        # Zoomed in - show municipalities
        if kommune_data:
            folium.GeoJson(
                kommune_data,
                name='municipalities_layer',
                style_function=municipality_style,
                tooltip=folium.GeoJsonTooltip(
                    fields=['kommunenavn', 'price_area'],
                    aliases=['Municipality:', 'Price Area:'],
                    localize=True
                )
            ).add_to(m)
    else:
        # Zoomed out - show price areas
        if price_area_data:
            folium.GeoJson(
                price_area_data,
                name='price_areas_layer',
                style_function=price_area_style,
                tooltip=folium.GeoJsonTooltip(
                    fields=['name'],
                    aliases=['Price Area:'],
                    localize=True
                )
            ).add_to(m)
    
    # Add custom vertical legend to map
    legend_html = create_vertical_legend(min_val, max_val, energy_group)
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add marker for selected location
    if st.session_state.selected_coordinates:
        lat, lon = st.session_state.selected_coordinates
        popup_text = f"""
        <b>Selected Location</b><br>
        Price Area: {st.session_state.selected_price_area or 'Unknown'}<br>
        Lat: {lat:.4f}<br>
        Lon: {lon:.4f}
        """
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=200),
            icon=folium.Icon(color='red', icon='info-sign'),
            tooltip=f"Price Area: {st.session_state.selected_price_area or 'Unknown'}"
        ).add_to(m)
    
    # Render the map and capture click events
    map_data = st_folium(
        m,
        width=None,
        height=600,
        returned_objects=["last_clicked", "zoom", "bounds"],
        use_container_width=True,
        key="main_map"
    )
    
    # Handle click events - only rerun on actual clicks
    need_rerun = False
    
    if map_data and map_data.get('last_clicked'):
        clicked_lat = map_data['last_clicked']['lat']
        clicked_lon = map_data['last_clicked']['lng']
        click_key = (round(clicked_lat, 6), round(clicked_lon, 6))
        
        # Check if this click has already been processed
        if click_key != st.session_state.last_clicked_processed:
            st.session_state.last_clicked_processed = click_key
            
            # Find which price area the click is in
            price_area = find_price_area_for_point(clicked_lat, clicked_lon, price_area_data)
            
            # Update session state
            st.session_state.selected_coordinates = (clicked_lat, clicked_lon)
            st.session_state.selected_price_area = price_area
            st.session_state.map_center = [clicked_lat, clicked_lon]
            
            # Update zoom if returned
            if map_data.get('zoom'):
                st.session_state.map_zoom = map_data['zoom']
            
            need_rerun = True
    
    # Check if zoom crossed threshold (for layer switching)
    if map_data and map_data.get('zoom'):
        new_zoom = map_data['zoom']
        old_zoom = st.session_state.map_zoom
        
        # Check if zoom crossed the threshold
        crossed_threshold = (old_zoom < ZOOM_THRESHOLD and new_zoom >= ZOOM_THRESHOLD) or \
                           (old_zoom >= ZOOM_THRESHOLD and new_zoom < ZOOM_THRESHOLD)
        
        if crossed_threshold:
            st.session_state.map_zoom = new_zoom
            # Calculate center from bounds to preserve map position
            if map_data.get('bounds'):
                bounds = map_data['bounds']
                center_lat = (bounds['_southWest']['lat'] + bounds['_northEast']['lat']) / 2
                center_lng = (bounds['_southWest']['lng'] + bounds['_northEast']['lng']) / 2
                st.session_state.map_center = [center_lat, center_lng]
            need_rerun = True
    
    if need_rerun:
        st.rerun()

with col_controls:
    st.markdown("---")
    st.subheader("2. Weather Data Download")
    st.caption("Configure and download historical weather data for the selected location.")
    
    # Year Range Selector
    current_year = 2024
    years = list(range(2021, current_year + 1))
    
    col_y1, col_y2 = st.columns(2)
    with col_y1:
        start_year = st.selectbox("Start Year", options=years, index=0)
    with col_y2:
        end_year = st.selectbox("End Year", options=years, index=len(years)-1)

    if start_year > end_year:
        st.error("Start year must be before End year")
    
    st.markdown("---")
    
    # Display current selection info
    if st.session_state.selected_coordinates:
        lat, lon = st.session_state.selected_coordinates
        st.info(f"📍 **Coordinates:** {lat:.4f}, {lon:.4f}")
    else:
        st.warning("Click on the map to select a location.")
        
    if st.session_state.selected_price_area:
        st.success(f"⚡ **Price Area:** {st.session_state.selected_price_area}")
    
    # Download Button
    if st.button("📥 Download Weather Data", type="primary", disabled=not st.session_state.selected_coordinates):
        if st.session_state.selected_coordinates and start_year <= end_year:
            lat, lon = st.session_state.selected_coordinates
            with st.spinner(f"Downloading weather data for {start_year}-{end_year}..."):
                try:
                    weather_df = download_weather_data(lat, lon, start_year, end_year)
                    st.session_state.weather_data = weather_df
                    st.session_state.data_range = (start_year, end_year)
                    st.session_state.weather_data_area = st.session_state.selected_price_area
                    st.success(f"✅ Loaded {len(weather_df)} records ({start_year}-{end_year})!")
                except Exception as e:
                    st.error(f"Error downloading data: {e}")

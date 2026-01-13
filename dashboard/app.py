import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Afghanistan Drought Stress Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

# Cache data loading
@st.cache_data
def load_drought_indicators():
    """Load the main drought indicators dataset."""
    df = pd.read_csv(PROCESSED_DIR / "afg_drought_indicators_2000_2025.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_district_summary():
    """Load the district drought summary dataset."""
    df = pd.read_csv(PROCESSED_DIR / "afg_district_drought_summary.csv")
    df['worst_drought_date'] = pd.to_datetime(df['worst_drought_date'])
    return df

@st.cache_data
def load_geojson():
    """Load the district boundaries GeoJSON."""
    geojson_path = RAW_DIR / "afg_district_lookup.geojson"
    if geojson_path.exists():
        with open(geojson_path, 'r') as f:
            return json.load(f)
    return None

# Helper functions
def get_drought_severity(cdi_value):
    """Classify drought severity based on CDI value."""
    if cdi_value < 20:
        return "Extreme Drought"
    elif cdi_value < 35:
        return "Severe Drought"
    elif cdi_value < 50:
        return "Moderate Drought"
    elif cdi_value < 65:
        return "Mild Drought"
    else:
        return "No Drought"

def get_severity_color(severity):
    """Get color for drought severity."""
    colors = {
        "Extreme Drought": "#8B0000",
        "Severe Drought": "#FF4500",
        "Moderate Drought": "#FFA500",
        "Mild Drought": "#FFD700",
        "No Drought": "#228B22"
    }
    return colors.get(severity, "#808080")

def filter_data(df, provinces=None, districts=None, date_range=None, cdi_column='CDI'):
    """Filter the dataframe based on user selections."""
    filtered = df.copy()
    
    if provinces and len(provinces) > 0:
        filtered = filtered[filtered['ADM1_NAME'].isin(provinces)]
    
    if districts and len(districts) > 0:
        filtered = filtered[filtered['ADM2_NAME'].isin(districts)]
    
    if date_range:
        start_date, end_date = date_range
        filtered = filtered[(filtered['date'] >= pd.to_datetime(start_date)) & 
                           (filtered['date'] <= pd.to_datetime(end_date))]
    
    return filtered

# Load data
@st.cache_data
def get_data():
    indicators = load_drought_indicators()
    summary = load_district_summary()
    geojson = load_geojson()
    return indicators, summary, geojson

# Main app
def main():
    # Header
    st.title("Afghanistan Drought Stress Dashboard")
    st.markdown("Monitoring drought conditions across Afghanistan using satellite-derived indices (2000-2025)")
    
    # Load data
    try:
        indicators_df, summary_df, geojson = get_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure the data files are in the correct location.")
        return
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # CDI toggle
    cdi_column = st.sidebar.radio(
        "Select CDI Measure",
        options=['CDI', 'CDI_alt'],
        help="CDI: Combined Drought Index, CDI_alt: Alternative calculation"
    )
    
    # Date range selector
    st.sidebar.subheader("Date Range")
    min_date = indicators_df['date'].min()
    max_date = indicators_df['date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Province/District selection
    st.sidebar.subheader("Location Filter")
    provinces = sorted(indicators_df['ADM1_NAME'].unique())
    selected_provinces = st.sidebar.multiselect(
        "Select Province(s)",
        options=provinces,
        default=[]
    )
    
    # Filter districts based on selected provinces
    if selected_provinces:
        available_districts = indicators_df[indicators_df['ADM1_NAME'].isin(selected_provinces)]['ADM2_NAME'].unique()
    else:
        available_districts = indicators_df['ADM2_NAME'].unique()
    
    selected_districts = st.sidebar.multiselect(
        "Select District(s)",
        options=sorted(available_districts),
        default=[]
    )
    
    # Drought severity threshold
    st.sidebar.subheader("Alert Threshold")
    drought_threshold = st.sidebar.slider(
        "Drought Alert Threshold (CDI)",
        min_value=0,
        max_value=100,
        value=20,
        help="Districts with CDI below this value are flagged as severe drought"
    )
    
    # Filter data
    filtered_df = filter_data(
        indicators_df,
        provinces=selected_provinces if selected_provinces else None,
        districts=selected_districts if selected_districts else None,
        date_range=date_range if len(date_range) == 2 else None,
        cdi_column=cdi_column
    )
    
    # ==================== CURRENT CONDITIONS PANEL ====================
    st.header("Current Conditions")
    
    # Get latest date data
    latest_date = filtered_df['date'].max()
    latest_data = filtered_df[filtered_df['date'] == latest_date]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_cdi = latest_data[cdi_column].mean()
        severity = get_drought_severity(avg_cdi)
        st.metric(
            label="National Avg CDI",
            value=f"{avg_cdi:.1f}",
            delta=severity
        )
    
    with col2:
        severe_count = len(latest_data[latest_data[cdi_column] < drought_threshold])
        total_districts = len(latest_data)
        st.metric(
            label="Districts in Severe Drought",
            value=f"{severe_count}",
            delta=f"of {total_districts} total"
        )
    
    with col3:
        avg_vci = latest_data['VCI'].mean()
        st.metric(
            label="Avg VCI (Vegetation)",
            value=f"{avg_vci:.1f}",
            delta="Vegetation Condition"
        )
    
    with col4:
        avg_spi = latest_data['SPI'].mean()
        spi_status = "Wet" if avg_spi > 0 else "Dry"
        st.metric(
            label="Avg SPI (Precipitation)",
            value=f"{avg_spi:.2f}",
            delta=spi_status
        )
    
    # Drought Alert Box
    if severe_count > 0:
        st.warning(f"**DROUGHT ALERT**: {severe_count} districts have CDI below {drought_threshold} (severe drought conditions)")
        
        # Show affected districts
        affected_districts = latest_data[latest_data[cdi_column] < drought_threshold][['ADM2_NAME', 'ADM1_NAME', cdi_column]].sort_values(cdi_column)
        with st.expander(f"View {severe_count} Affected Districts"):
            st.dataframe(affected_districts.rename(columns={cdi_column: 'CDI Value'}), use_container_width=True)
    
    # ==================== INTERACTIVE MAP PANEL ====================
    st.header("Interactive Drought Map")
    
    # Date slider for map
    available_dates = sorted(filtered_df['date'].unique())
    if len(available_dates) > 0:
        date_options = [d.strftime('%Y-%m') for d in pd.to_datetime(available_dates)]
        unique_months = list(dict.fromkeys(date_options))
        
        selected_month_idx = st.select_slider(
            "Select Month",
            options=range(len(unique_months)),
            value=len(unique_months) - 1,
            format_func=lambda x: unique_months[x]
        )
        
        selected_month = unique_months[selected_month_idx]
        map_data = filtered_df[filtered_df['date'].dt.strftime('%Y-%m') == selected_month]
        
        # Create choropleth map
        if geojson is not None and not map_data.empty:
            # District-level choropleth map
            fig_map = px.choropleth_mapbox(
                map_data,
                geojson=geojson,
                locations='ADM2_CODE',
                featureidkey="properties.ADM2_CODE",
                color=cdi_column,
                color_continuous_scale=[
                    [0, "#8B0000"],      # Extreme drought (dark red)
                    [0.2, "#FF4500"],    # Severe drought (orange red)
                    [0.35, "#FFA500"],   # Moderate drought (orange)
                    [0.5, "#FFD700"],    # Mild drought (gold)
                    [0.65, "#90EE90"],   # Near normal (light green)
                    [1, "#228B22"]       # Good conditions (forest green)
                ],
                range_color=[0, 100],
                mapbox_style="carto-positron",
                zoom=4.5,
                center={"lat": 33.9391, "lon": 67.7100},
                opacity=0.7,
                hover_name='ADM2_NAME',
                hover_data={cdi_column: ':.1f', 'VCI': ':.1f', 'TCI': ':.1f', 'SPI': ':.2f'}
            )
            fig_map.update_layout(
                margin={"r": 0, "t": 30, "l": 0, "b": 0},
                height=500,
                title=f"Drought Conditions - {selected_month}"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        elif not map_data.empty:
            # Fallback: bar chart if no GeoJSON
            st.info("Creating bar chart visualization (GeoJSON boundaries not available)")
            
            fig_bar = px.bar(
                map_data.sort_values(cdi_column),
                x='ADM2_NAME',
                y=cdi_column,
                color=cdi_column,
                color_continuous_scale='RdYlGn',
                range_color=[0, 100],
                title=f"CDI by District - {selected_month}",
                hover_data={'VCI': ':.1f', 'TCI': ':.1f', 'SPI': ':.2f'}
            )
            fig_bar.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # ==================== TIME SERIES PANEL ====================
    st.header("Time Series Analysis")
    
    # Allow selection of districts for time series
    ts_districts = selected_districts if selected_districts else list(filtered_df['ADM2_NAME'].unique()[:5])
    
    if not ts_districts:
        ts_districts = list(indicators_df['ADM2_NAME'].unique()[:3])
    
    ts_selection = st.multiselect(
        "Select Districts for Time Series",
        options=sorted(filtered_df['ADM2_NAME'].unique()),
        default=ts_districts[:3] if len(ts_districts) >= 3 else ts_districts
    )
    
    if ts_selection:
        ts_data = filtered_df[filtered_df['ADM2_NAME'].isin(ts_selection)]
        
        # Create tabs for different indicators
        tab1, tab2, tab3, tab4 = st.tabs(["CDI", "VCI", "TCI", "SPI"])
        
        with tab1:
            fig_cdi = px.line(
                ts_data,
                x='date',
                y=cdi_column,
                color='ADM2_NAME',
                title=f"Combined Drought Index ({cdi_column}) Over Time",
                labels={cdi_column: 'CDI Value', 'date': 'Date', 'ADM2_NAME': 'District'}
            )
            fig_cdi.add_hline(y=drought_threshold, line_dash="dash", line_color="red", 
                            annotation_text=f"Drought Threshold ({drought_threshold})")
            fig_cdi.update_layout(height=400)
            st.plotly_chart(fig_cdi, use_container_width=True)
        
        with tab2:
            fig_vci = px.line(
                ts_data,
                x='date',
                y='VCI',
                color='ADM2_NAME',
                title="Vegetation Condition Index (VCI) Over Time",
                labels={'VCI': 'VCI Value', 'date': 'Date', 'ADM2_NAME': 'District'}
            )
            fig_vci.add_hline(y=35, line_dash="dash", line_color="orange",
                            annotation_text="Vegetation Stress Threshold")
            fig_vci.update_layout(height=400)
            st.plotly_chart(fig_vci, use_container_width=True)
        
        with tab3:
            fig_tci = px.line(
                ts_data,
                x='date',
                y='TCI',
                color='ADM2_NAME',
                title="Temperature Condition Index (TCI) Over Time",
                labels={'TCI': 'TCI Value', 'date': 'Date', 'ADM2_NAME': 'District'}
            )
            fig_tci.update_layout(height=400)
            st.plotly_chart(fig_tci, use_container_width=True)
        
        with tab4:
            fig_spi = px.line(
                ts_data,
                x='date',
                y='SPI',
                color='ADM2_NAME',
                title="Standardized Precipitation Index (SPI) Over Time",
                labels={'SPI': 'SPI Value', 'date': 'Date', 'ADM2_NAME': 'District'}
            )
            fig_spi.add_hline(y=-1, line_dash="dash", line_color="orange",
                            annotation_text="Moderate Drought")
            fig_spi.add_hline(y=-2, line_dash="dash", line_color="red",
                            annotation_text="Extreme Drought")
            fig_spi.update_layout(height=400)
            st.plotly_chart(fig_spi, use_container_width=True)
    
    # ==================== HISTORICAL COMPARISON PANEL ====================
    st.header("Historical Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select month for comparison
        comparison_month = st.selectbox(
            "Select Month for Comparison",
            options=range(1, 13),
            format_func=lambda x: datetime(2000, x, 1).strftime('%B'),
            index=datetime.now().month - 1
        )
    
    with col2:
        # Select district for comparison
        comparison_district = st.selectbox(
            "Select District for Comparison",
            options=sorted(filtered_df['ADM2_NAME'].unique()),
            index=0
        )
    
    # Get historical data for the selected month
    historical_data = indicators_df[
        (indicators_df['month'] == comparison_month) & 
        (indicators_df['ADM2_NAME'] == comparison_district)
    ].copy()
    
    if not historical_data.empty:
        # Create comparison chart
        fig_historical = go.Figure()
        
        # Add bars for each year
        fig_historical.add_trace(go.Bar(
            x=historical_data['year'],
            y=historical_data[cdi_column],
            marker_color=historical_data[cdi_column].apply(
                lambda x: get_severity_color(get_drought_severity(x))
            ),
            text=historical_data[cdi_column].round(1),
            textposition='outside',
            name='CDI'
        ))
        
        # Add threshold line
        fig_historical.add_hline(y=drought_threshold, line_dash="dash", line_color="red",
                                annotation_text=f"Drought Threshold ({drought_threshold})")
        
        fig_historical.update_layout(
            title=f"Historical CDI Comparison - {datetime(2000, comparison_month, 1).strftime('%B')} - {comparison_district}",
            xaxis_title="Year",
            yaxis_title=f"CDI Value ({cdi_column})",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_historical, use_container_width=True)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean CDI", f"{historical_data[cdi_column].mean():.1f}")
        with col2:
            st.metric("Min CDI", f"{historical_data[cdi_column].min():.1f}")
        with col3:
            st.metric("Max CDI", f"{historical_data[cdi_column].max():.1f}")
    
    # ==================== SUMMARY TABLE ====================
    st.header("District Summary")
    
    # Filter summary based on selection
    if selected_provinces:
        display_summary = summary_df[summary_df['ADM1_NAME'].isin(selected_provinces)]
    else:
        display_summary = summary_df
    
    # Sort by worst drought
    display_summary = display_summary.sort_values('CDI_min')
    
    # Display summary table
    st.dataframe(
        display_summary[[
            'ADM2_NAME', 'ADM1_NAME', 'CDI_mean', 'CDI_min', 'CDI_max', 
            'n_observations', 'worst_drought_date'
        ]].rename(columns={
            'ADM2_NAME': 'District',
            'ADM1_NAME': 'Province',
            'CDI_mean': 'Avg CDI',
            'CDI_min': 'Min CDI',
            'CDI_max': 'Max CDI',
            'n_observations': 'Observations',
            'worst_drought_date': 'Worst Drought Date'
        }),
        use_container_width=True,
        height=400
    )
    
    # ==================== DOWNLOAD SECTION ====================
    st.header("Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare filtered data for download
        download_data = filtered_df.copy()
        download_data['date'] = download_data['date'].dt.strftime('%Y-%m-%d')
        
        csv = download_data.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name=f"drought_data_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download summary
        summary_csv = display_summary.to_csv(index=False)
        st.download_button(
            label="Download Summary Data (CSV)",
            data=summary_csv,
            file_name=f"drought_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Data Sources**: 
    - VCI/TCI derived from MODIS satellite imagery
    - SPI calculated from CHIRPS precipitation data
    - CDI combines VCI, TCI, and SPI indices
    
    **About**: This dashboard monitors drought conditions across Afghanistan's districts using satellite-derived vegetation, 
    temperature, and precipitation indices from 2000-2025.
    """)

if __name__ == "__main__":
    main()

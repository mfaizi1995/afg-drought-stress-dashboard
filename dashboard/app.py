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

# Paths - resolve to absolute path for Streamlit Cloud compatibility
APP_DIR = Path(__file__).parent.resolve()
ROOT_DIR = APP_DIR.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

# Cache data loading
@st.cache_data
def load_drought_indicators():
    """Load the main drought indicators dataset from Parquet."""
    parquet_path = PROCESSED_DIR / "afg_drought_indicators_2000_2025.parquet"
    csv_path = PROCESSED_DIR / "afg_drought_indicators_2000_2025.csv"
    
    # Prefer Parquet if available (faster loading)
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M').astype(str)
    return df

@st.cache_data
def load_district_summary():
    """Load the district drought summary dataset."""
    df = pd.read_csv(PROCESSED_DIR / "afg_district_drought_summary.csv")
    df['worst_drought_date'] = pd.to_datetime(df['worst_drought_date'])
    return df

@st.cache_data
def load_geojson():
    """Load the pre-simplified district boundaries GeoJSON."""
    # Prefer simplified version if available
    simplified_path = RAW_DIR / "afg_district_lookup_simplified.geojson"
    original_path = RAW_DIR / "afg_district_lookup.geojson"
    
    geojson_path = simplified_path if simplified_path.exists() else original_path
    
    if geojson_path.exists():
        with open(geojson_path, 'r') as f:
            return json.load(f)
    return None

# Helper functions
def get_drought_severity(cdi_value):
    """Classify drought severity based on CDI value (matches Notebook 01 interpretation guide)."""
    if cdi_value < 10:
        return "Extreme Drought"
    elif cdi_value < 20:
        return "Severe Drought"
    elif cdi_value < 30:
        return "Moderate Drought"
    elif cdi_value < 40:
        return "Mild Drought"
    elif cdi_value < 50:
        return "Near Normal (Dry)"
    else:
        return "Wet/Favorable"

def get_severity_color(severity):
    """Get color for drought severity."""
    colors = {
        "Extreme Drought": "#8B0000",      # Dark red
        "Severe Drought": "#FF4500",       # Orange red
        "Moderate Drought": "#FFA500",     # Orange
        "Mild Drought": "#FFD700",         # Gold
        "Near Normal (Dry)": "#ADFF2F",    # Green yellow
        "Wet/Favorable": "#228B22"         # Forest green
    }
    return colors.get(severity, "#808080")

def get_vci_descriptor(vci_value):
    """Get VCI condition descriptor."""
    if vci_value < 10:
        return "Extreme vegetation stress"
    elif vci_value < 20:
        return "Severe vegetation stress"
    elif vci_value < 35:
        return "Moderate vegetation stress"
    elif vci_value < 50:
        return "Mild vegetation stress"
    elif vci_value < 65:
        return "Normal conditions"
    elif vci_value < 80:
        return "Good vegetation"
    elif vci_value < 90:
        return "Very good vegetation"
    else:
        return "Excellent vegetation"

def get_tci_descriptor(tci_value):
    """Get TCI condition descriptor."""
    if tci_value < 10:
        return "Extreme heat stress"
    elif tci_value < 20:
        return "Severe heat stress"
    elif tci_value < 35:
        return "Moderate heat stress"
    elif tci_value < 50:
        return "Mild heat stress"
    elif tci_value < 65:
        return "Normal conditions"
    elif tci_value < 80:
        return "Mild cool conditions"
    elif tci_value < 90:
        return "Moderate cool conditions"
    else:
        return "Extreme cool conditions"

def get_spi_descriptor(spi_value):
    """Get SPI condition descriptor."""
    if spi_value <= -2.0:
        return "Extremely dry"
    elif spi_value <= -1.5:
        return "Severely dry"
    elif spi_value <= -1.0:
        return "Moderately dry"
    elif spi_value <= -0.5:
        return "Mildly dry"
    elif spi_value <= 0.5:
        return "Near normal"
    elif spi_value <= 1.0:
        return "Mildly wet"
    elif spi_value <= 1.5:
        return "Moderately wet"
    elif spi_value <= 2.0:
        return "Severely wet"
    else:
        return "Extremely wet"

def format_year_month(year_month_str):
    """Convert '2025-11' to 'Nov 2025'."""
    try:
        dt = datetime.strptime(year_month_str, '%Y-%m')
        return dt.strftime('%b %Y')
    except:
        return year_month_str

@st.cache_data
def filter_data(_df, provinces=None, districts=None, date_range=None):
    """Filter the dataframe based on user selections. Cached for performance."""
    mask = pd.Series(True, index=_df.index)
    
    if provinces and len(provinces) > 0:
        mask &= _df['ADM1_NAME'].isin(provinces)
    
    if districts and len(districts) > 0:
        mask &= _df['ADM2_NAME'].isin(districts)
    
    if date_range and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        mask &= (_df['date'] >= start_date) & (_df['date'] <= end_date)
    
    return _df[mask]

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
        # Ensure year_month column exists (handles cached data)
        if 'year_month' not in indicators_df.columns:
            indicators_df = indicators_df.copy()
            indicators_df['year_month'] = indicators_df['date'].dt.to_period('M').astype(str)
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
        help="CDI: Equal weighting (⅓ VCI + ⅓ TCI + ⅓ SPI). CDI_alt: Alternative weighting (0.4 VCI + 0.4 TCI + 0.2 SPI) used to check robustness of results."
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
        value=30,
        help="Adjust to flag districts below your chosen CDI threshold."
    )
    
    # CDI Interpretation Guide in sidebar
    st.sidebar.markdown("""
    **CDI Interpretation:**
    - 0–10: Extreme Drought
    - 10–20: Severe Drought
    - 20–30: Moderate Drought
    - 30–40: Mild Drought
    - 40–50: Near Normal (Dry)
    - 50+: Wet/Favorable
    """)
    
    # Filter data (convert lists to tuples for caching)
    filtered_df = filter_data(
        indicators_df,
        provinces=tuple(selected_provinces) if selected_provinces else None,
        districts=tuple(selected_districts) if selected_districts else None,
        date_range=tuple(date_range) if len(date_range) == 2 else None
    )
    
    # Ensure year_month column exists in filtered data
    if 'year_month' not in filtered_df.columns:
        filtered_df = filtered_df.copy()
        filtered_df['year_month'] = filtered_df['date'].dt.to_period('M').astype(str)
    
    # Get unique months for slider (used later)
    unique_months = sorted(filtered_df['year_month'].unique())
    
    # ==================== CONDITIONS PANEL ====================
    # Use latest month for conditions panel
    if len(unique_months) > 0:
        latest_month = unique_months[-1]
        latest_month_data = filtered_df[filtered_df['year_month'] == latest_month]
        latest_month_display = format_year_month(latest_month)
    else:
        latest_month_display = "N/A"
        latest_month_data = filtered_df
    
    st.header(f"Latest Conditions — {latest_month_display}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_cdi = latest_month_data[cdi_column].mean()
        severity = get_drought_severity(avg_cdi)
        st.metric(
            label="National Avg CDI",
            value=f"{avg_cdi:.1f}",
            delta=severity
        )
    
    with col2:
        severe_count = len(latest_month_data[latest_month_data[cdi_column] < drought_threshold])
        total_districts = len(latest_month_data)
        st.metric(
            label="Districts Below Threshold",
            value=f"{severe_count}",
            delta=f"of {total_districts} total",
            help=f"Number of districts with CDI below {drought_threshold} (your selected threshold)"
        )
    
    with col3:
        avg_vci = latest_month_data['VCI'].mean()
        vci_descriptor = get_vci_descriptor(avg_vci)
        st.metric(
            label="Avg VCI",
            value=f"{avg_vci:.1f}",
            delta=vci_descriptor
        )
    
    with col4:
        avg_tci = latest_month_data['TCI'].mean()
        tci_descriptor = get_tci_descriptor(avg_tci)
        st.metric(
            label="Avg TCI",
            value=f"{avg_tci:.1f}",
            delta=tci_descriptor
        )
    
    with col5:
        avg_spi = latest_month_data['SPI'].mean()
        spi_descriptor = get_spi_descriptor(avg_spi)
        st.metric(
            label="Avg SPI",
            value=f"{avg_spi:.2f}",
            delta=spi_descriptor
        )
    
    # Drought Alert Box
    if severe_count > 0:
        st.warning(f"**DROUGHT ALERT**: {severe_count} districts have CDI below {drought_threshold}")
        
        # Show affected districts
        affected_districts = latest_month_data[latest_month_data[cdi_column] < drought_threshold][['ADM2_NAME', 'ADM1_NAME', cdi_column]].sort_values(cdi_column)
        with st.expander(f"View {severe_count} Affected Districts"):
            st.dataframe(affected_districts.rename(columns={cdi_column: 'CDI Value'}), use_container_width=True)
    
    # ==================== INTERACTIVE MAP PANEL ====================
    st.header("Interactive Drought Map")
    
    # Month slider for map
    if len(unique_months) > 0:
        selected_month_idx = st.select_slider(
            "Select Month",
            options=range(len(unique_months)),
            value=len(unique_months) - 1,
            format_func=lambda x: format_year_month(unique_months[x])
        )
        selected_month = unique_months[selected_month_idx]
        selected_month_display = format_year_month(selected_month)
        month_data = filtered_df[filtered_df['year_month'] == selected_month]
        # Create choropleth map
        if geojson is not None and not month_data.empty:
            # Only pass necessary columns to reduce data transfer
            map_cols = ['ADM2_CODE', 'ADM2_NAME', cdi_column, 'VCI', 'TCI', 'SPI']
            map_data_slim = month_data[map_cols].copy()
            
            # District-level choropleth map
            fig_map = px.choropleth_mapbox(
                map_data_slim,
                geojson=geojson,
                locations='ADM2_CODE',
                featureidkey="properties.ADM2_CODE",
                color=cdi_column,
                color_continuous_scale="RdYlGn",
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
                title=f"Drought Conditions — {selected_month_display}"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        elif not month_data.empty:
            # Fallback: bar chart if no GeoJSON
            st.info("Creating bar chart visualization (GeoJSON boundaries not available)")
            
            fig_bar = px.bar(
                month_data.sort_values(cdi_column),
                x='ADM2_NAME',
                y=cdi_column,
                color=cdi_column,
                color_continuous_scale='RdYlGn',
                range_color=[0, 100],
                title=f"CDI by District — {selected_month_display}",
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
        # Create comparison chart using continuous colorscale
        fig_historical = px.bar(
            historical_data,
            x='year',
            y=cdi_column,
            color=cdi_column,
            color_continuous_scale='RdYlGn',
            range_color=[0, 100],
            text=historical_data[cdi_column].round(1),
            title=f"Historical CDI Comparison - {datetime(2000, comparison_month, 1).strftime('%B')} - {comparison_district}"
        )
        
        fig_historical.update_traces(textposition='outside')
        
        # Add threshold line
        fig_historical.add_hline(y=drought_threshold, line_dash="dash", line_color="red",
                                annotation_text=f"Drought Threshold ({drought_threshold})")
        
        fig_historical.update_layout(
            xaxis_title="Year",
            yaxis_title=f"CDI Value ({cdi_column})",
            height=400,
            showlegend=True
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
    ### Methodology
    The **Combined Drought Index (CDI)** integrates three satellite-derived indicators:
    - **VCI (Vegetation Condition Index)**: Measures vegetation health relative to historical min/max from MODIS NDVI
    - **TCI (Temperature Condition Index)**: Measures thermal stress relative to historical min/max from MODIS LST
    - **SPI (Standardized Precipitation Index)**: Measures precipitation anomalies using a 3-month window from CHIRPS data
    
    **CDI** uses equal weighting (⅓ VCI + ⅓ TCI + ⅓ SPI). **CDI_alt** applies alternative weights (0.4 VCI + 0.4 TCI + 0.2 SPI) 
    to test robustness. This project does not aim to determine optimal weights, the alternative calculation demonstrates 
    that findings remain consistent across different weighting schemes.
    
    ### Data Sources
    - **Vegetation & Temperature**: MODIS Terra (MOD13A2 NDVI, MOD11A2 LST) via Google Earth Engine
    - **Precipitation**: CHIRPS daily rainfall aggregated to monthly totals
    - **Administrative Boundaries**: OCHA Afghanistan district shapefiles (400+ districts, 34 provinces)
    - **Temporal Coverage**: January 2000 – December 2025 (monthly)
    
    ### About
    This dashboard monitors drought conditions across Afghanistan's 400+ districts using satellite-derived indices. 
    All data processing, indicator construction, and validation workflows are documented in the project's Jupyter notebooks. 
    For methodology details and source code, see the [GitHub repository](https://github.com/mfaizi1995/afg-drought-stress-dashboard).
    """)

if __name__ == "__main__":
    main()

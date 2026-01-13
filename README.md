# Afghanistan Drought Stress Dashboard

District-level drought monitoring for Afghanistan using satellite data (2000–2025).

## Overview

This project combines three satellite-derived indicators into a Composite Drought Index (CDI) for all 399 Afghan districts:

- **VCI** (Vegetation Condition Index) — vegetation health from MODIS NDVI
- **TCI** (Temperature Condition Index) — heat stress from MODIS LST
- **SPI-3** (Standardized Precipitation Index) — 3-month rainfall anomalies from CHIRPS

CDI ranges from 0 (extreme drought) to 100 (wet conditions), with 50 as normal.

## Data Sources

| Variable | Source | Resolution |
|----------|--------|------------|
| NDVI | MODIS MOD13A3 | 1 km, monthly |
| LST | MODIS MOD11A2 | 1 km, 8-day |
| Rainfall | CHIRPS | 5.5 km, daily |
| Boundaries | FAO GAUL Level 2 | 399 districts |

## Notebooks

1. **01_project_overview** - Project docs, methods, limitations
2. **02_data_extraction_and_preprocessing** - GEE extraction and spatial aggregation
3. **03_drought_indicator_construction** - VCI, TCI, SPI computation
4. **04_composite_drought_index** - CDI creation and weight sensitivity
5. **05_validation_and_analysis** - Maps, validation, hotspots

## Quick Start

```bash
pip install -r requirements.txt
```

Notebooks are designed to run in order. Notebook 02 requires Google Earth Engine authentication (exported data is already in `data/raw/` if you want to skip that step).

## Outputs

- `data/processed/afg_drought_indicators_2000_2025.csv` - Full time series with all indices
- `data/processed/afg_district_drought_summary.csv` - District-level statistics
- `dashboard/app.py` - Interactive dashboard (Plotly Dash)

## Limitations

- Satellite proxies, not ground-truth agricultural data
- Validation was qualitative (visual comparison with SPEI, alignment with known droughts)
- Equal weighting is arbitrary, though sensitivity analysis shows robustness (r = 0.975)

This is a screening tool for geographic prioritization, not a predictive model.

## Author

Mastoorah Faizi

## License

MIT

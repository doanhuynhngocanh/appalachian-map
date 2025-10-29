# Appalachian Map Chatbot

An interactive web application that combines an interactive map of Appalachian counties with a natural language chatbot for querying economic data.

## Features

- **Interactive Map**: Visualize Appalachian counties with Leaflet.js
- **Natural Language Chatbot**: Ask questions in plain English about county data
- **Economic Data**: Unemployment rates and GDP per capita data
- **Real-time Queries**: Get instant results with highlighted counties on the map

## Data Structure

The application uses normalized data tables:

- `counties.csv`: Master list of Appalachian counties (fips, county, state, arc_flag)
- `unemployment_normalized.csv`: Unemployment data (fips, year, month, unemployment_rate)
- `gdp_with_per_capita.csv`: GDP data with per capita calculations
- `county_summary.csv`: Summary statistics for each county
- `appalachia_snapshot.geojson`: Geographic boundaries for mapping

## Example Queries

- "Show counties with unemployment > 8%"
- "List counties with GDP per capita > $40,000"
- "Find counties in Alabama"
- "What's the unemployment rate in Calhoun County?"
- "Count counties with unemployment below 5%"

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the data normalization script:
```bash
python normalize_data.py
```

3. Start the Flask application:
```bash
python app.py
```

4. Open your browser to `http://localhost:5000`

## Data Sources

- **ARC Counties**: Appalachian Regional Commission county list
- **BEA GDP**: Bureau of Economic Analysis county GDP data
- **BLS Unemployment**: Bureau of Labor Statistics unemployment data
- **Census Geography**: TIGER/Line shapefiles for county boundaries

## Architecture

- **Frontend**: HTML/CSS/JavaScript with Leaflet.js for mapping
- **Backend**: Flask Python web framework
- **Data Processing**: Pandas for data manipulation
- **Natural Language Processing**: Custom regex-based query parser
- **Geographic Data**: GeoPandas for spatial data handling

## File Structure

```
├── app.py                          # Flask web application
├── normalize_data.py               # Data normalization script
├── clean_data.py                   # Original data cleaning script
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                 # Main web interface
├── data_raw/                      # Raw data files
│   ├── arc_counties.csv
│   ├── bea_gdp_county.csv
│   ├── bls_unemployment.csv
│   └── hud_income_limits.csv
└── data_clean/                    # Cleaned data files
    ├── counties.csv
    ├── unemployment_normalized.csv
    ├── gdp_with_per_capita.csv
    ├── county_summary.csv
    └── appalachia_snapshot.geojson
```

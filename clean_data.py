# scripts/build_normalized_data.py
import pandas as pd
import geopandas as gpd
import json
import os

RAW = "data_raw"
OUT = "data_clean"
os.makedirs(OUT, exist_ok=True)

# --- 1) Load ARC counties (list of counties in Appalachia) ---
# Expect columns: state, county, fips (string 5-digit or numeric)
arc = pd.read_csv(os.path.join(RAW, "arc_counties.csv"), dtype={"FIPS":str})
# Normalize FIPS to 5-digit strings
if "FIPS" in arc.columns:
    arc["fips"] = arc["FIPS"].astype(str).str.zfill(5)
else:
    # try to construct from STATEFP & COUNTYFP
    arc["fips"] = arc["STATEFP"].astype(str).str.zfill(2) + arc["COUNTYFP"].astype(str).str.zfill(3)

arc = arc[["fips","COUNTY","STATE"]].rename(
    columns={"COUNTY":"county", "STATE":"state"}
)
arc["arc_flag"] = True
arc = arc.drop_duplicates(subset=["fips"])
arc.to_csv(os.path.join(OUT, "counties_arc.csv"), index=False)
print("ARC counties:", len(arc))

# --- 2) Load county geometries from Census (shapefile) ---
# Adjust path to your downloaded TIGER/cartographic shapefile
shp = os.path.join(RAW, "tl_2025_us_county.shp")  # update filename/year
gdf = gpd.read_file(shp)
# GEOID is the FIPS code (string)
gdf = gdf.rename(columns={"GEOID":"fips"})
gdf["fips"] = gdf["fips"].astype(str).str.zfill(5)
# Subset to Appalachia by FIPS join with arc list
g_app = gdf.merge(arc[["fips"]], on="fips", how="inner")
print("Appalachian geometries:", len(g_app))
# Export simplified geojson for front-end mapping
g_app = g_app.to_crs(epsg=4326)
g_app[["fips","NAME","STATEFP","geometry"]].to_file(os.path.join(OUT, "appalachia_geom.geojson"), driver="GeoJSON")

# --- 3) Clean BLS unemployment data ---
# This varies by file. We'll attempt to load common formats; adapt as needed.
bls = pd.read_csv(os.path.join(RAW, "bls_unemployment.csv"), dtype=str, encoding='latin-1')
# Convert numeric columns
if 'fips' not in bls.columns:
    # try to create from series id or other fields
    pass

# Normalize column names
if 'Year' in bls.columns:
    bls = bls.rename(columns={'Year': 'year'})

# Ensure numeric
bls['year'] = pd.to_numeric(bls['year'], errors='coerce')
bls['unemployment_rate'] = pd.to_numeric(bls['unemployment_rate'], errors='coerce')

# Ensure FIPS is properly formatted
bls['fips'] = bls['fips'].astype(str).str.zfill(5)

# Keep only Appalachia FIPS
bls = bls[bls['fips'].isin(arc['fips'])]
bls.to_csv(os.path.join(OUT, "unemployment.csv"), index=False)

# --- 4) Clean BEA GDP per capita data ---
print("Cleaning BEA GDP per capita data...")

# Read the messy BEA GDP per capita data
bea_raw = pd.read_csv(os.path.join(RAW, "bea_gdp_county.csv"), dtype=str, encoding='latin-1')

# Skip header rows and get to the actual data
# The data starts after the header rows (rows 0-4 are headers)
bea_data = bea_raw.iloc[5:].copy()  # Skip first 5 rows which are headers

# Reset index and clean up
bea_data = bea_data.reset_index(drop=True)

# The first column contains state/county names, columns 1-3 contain GDP per capita data for 2021-2023
bea_data.columns = ['name', 'gdp_per_capita_2021', 'gdp_per_capita_2022', 'gdp_per_capita_2023', 'rank_2023', 'pct_change_2022', 'pct_change_2023', 'rank_pct_change_2023']

# Remove rows with empty names
bea_data = bea_data[bea_data['name'].notna() & (bea_data['name'] != '')]

# Create a mapping from ARC counties to help identify Appalachian counties
arc_mapping = arc.set_index(['state', 'county']).to_dict()['fips']

# Function to determine if a row is a state or county
def is_state_row(name):
    """Determine if a row represents a state (not a county)"""
    # States typically have much larger GDP values and no rank in the last column
    # Also, we can check against known state names
    state_names = {
        'Alabama', 'Georgia', 'Kentucky', 'Maryland', 'Mississippi', 'New York',
        'North Carolina', 'Ohio', 'Pennsylvania', 'South Carolina', 'Tennessee',
        'Virginia', 'West Virginia', 'United States'
    }
    return name in state_names

# Process the data to separate states from counties
bea_processed = []
current_state = None

for idx, row in bea_data.iterrows():
    name = row['name'].strip()
    
    if is_state_row(name):
        current_state = name
        continue  # Skip state rows, we only want counties
    
    if current_state is None:
        continue  # Skip rows before we encounter a state
    
    # This is a county row
    county_data = {
        'state': current_state,
        'county': name,
        'gdp_per_capita_2021': row['gdp_per_capita_2021'], 
        'gdp_per_capita_2022': row['gdp_per_capita_2022'],
        'gdp_per_capita_2023': row['gdp_per_capita_2023'],
        'rank_2023': row['rank_2023'],
        'pct_change_2022': row['pct_change_2022'],
        'pct_change_2023': row['pct_change_2023']
    }
    bea_processed.append(county_data)

# Convert to DataFrame
bea_counties = pd.DataFrame(bea_processed)

# Clean up county names to match ARC data format
def clean_county_name(county_name):
    """Clean county names to match ARC data format"""
    # Remove common suffixes and clean up
    county_name = county_name.strip()
    # Handle special cases like "De Kalb" -> "De Kalb" (already correct in ARC)
    return county_name

bea_counties['county_clean'] = bea_counties['county'].apply(clean_county_name)

# Create FIPS mapping by matching state and county names
bea_counties['fips'] = None

for idx, row in bea_counties.iterrows():
    state = row['state']
    county = row['county_clean']
    
    # Try exact match first
    if (state, county) in arc_mapping:
        bea_counties.at[idx, 'fips'] = arc_mapping[(state, county)]
    else:
        # Try some common variations
        variations = [
            county.replace(' County', ''),
            county.replace(' Parish', ''),
            county.replace(' Borough', ''),
            county.replace(' Municipality', ''),
            county.replace(' City', ''),
            county.replace(' and', ''),
            county.replace(' &', ''),
        ]
        
        for variation in variations:
            if (state, variation) in arc_mapping:
                bea_counties.at[idx, 'fips'] = arc_mapping[(state, variation)]
                break

# Filter to only Appalachian counties (those with FIPS codes)
bea_appalachian = bea_counties[bea_counties['fips'].notna()].copy()

print(f"Total counties in BEA data: {len(bea_counties)}")
print(f"Appalachian counties found: {len(bea_appalachian)}")

# Convert GDP per capita columns to numeric
for col in ['gdp_per_capita_2021', 'gdp_per_capita_2022', 'gdp_per_capita_2023']:
    bea_appalachian[col] = pd.to_numeric(bea_appalachian[col].str.replace(',', ''), errors='coerce')

# Convert percentage change columns to numeric
for col in ['pct_change_2022', 'pct_change_2023']:
    bea_appalachian[col] = pd.to_numeric(bea_appalachian[col], errors='coerce')

# Convert rank columns to numeric (only if they exist)
for col in ['rank_2023']:
    if col in bea_appalachian.columns:
        bea_appalachian[col] = pd.to_numeric(bea_appalachian[col], errors='coerce')

# Create a long format for easier analysis (optional)
bea_long = []
for year in ['2021', '2022', '2023']:
    year_data = bea_appalachian[['fips', 'state', 'county', f'gdp_per_capita_{year}']].copy()
    year_data['year'] = int(year)
    year_data['gdp_per_capita'] = year_data[f'gdp_per_capita_{year}']
    year_data = year_data[['fips', 'state', 'county', 'year', 'gdp_per_capita']]
    bea_long.append(year_data)

bea_long_df = pd.concat(bea_long, ignore_index=True)

# Save both wide and long formats
bea_appalachian.to_csv(os.path.join(OUT, "gdp_wide.csv"), index=False)
bea_long_df.to_csv(os.path.join(OUT, "gdp.csv"), index=False)

print(f"Saved {len(bea_appalachian)} Appalachian counties GDP per capita data")
print(f"Sample counties: {bea_appalachian[['state', 'county', 'fips']].head().to_string()}")

# --- 5) Clean HUD income limits (these files sometimes have one file per state) ---
income_limits = pd.read_csv(os.path.join(RAW, "hud_income_limits.csv"), dtype=str, encoding='latin-1')
# Expect fips or county/state plus year and income columns (e.g., very_low, median, 60pct)
# Ensure fips exists; otherwise map state+county to FIPS using a crosswalk
if 'fips' not in income_limits.columns:
    # create FIPS by joining to a crosswalk (we'll create below)
    pass

# Rename income limit columns for easier filtering
# l50_X -> very_low_X_person (Very Low 50% Income Limit)
# ELI_X -> extremely_low_X_person (Extremely Low 30% Income Limit)
# l80_X -> low_X_person (Low 80% Income Limit)
column_rename_map = {}
for i in range(1, 9):
    column_rename_map[f'l50_{i}'] = f'very_low_{i}_person'
    column_rename_map[f'ELI_{i}'] = f'extremely_low_{i}_person'
    column_rename_map[f'l80_{i}'] = f'low_{i}_person'

income_limits = income_limits.rename(columns=column_rename_map)

income_limits.to_csv(os.path.join(OUT, "income_limits.csv"), index=False)

# --- 6) Build a small "latest snapshot" merged GeoJSON for the front end ---
# Compute latest (most recent year) unemployment and gdp per capita per FIPS
latest_gdp = bea_long_df.sort_values(['fips','year']).groupby('fips').tail(1).drop_duplicates('fips')

# Handle unemployment data - get latest year for each county
if 'year' in bls.columns:
    latest_unemp = bls.sort_values(['fips','year']).groupby('fips').tail(1).drop_duplicates('fips')
    merged = arc.merge(latest_unemp[['fips','unemployment_rate']], on='fips', how='left') \
                .merge(latest_gdp[['fips','gdp_per_capita']], on='fips', how='left') \
                .merge(g_app[['fips','geometry']], on='fips', how='left')
else:
    # Skip unemployment data if structure is different
    print("Warning: BLS unemployment data structure not as expected, skipping unemployment merge")
    merged = arc.merge(latest_gdp[['fips','gdp_per_capita']], on='fips', how='left') \
                .merge(g_app[['fips','geometry']], on='fips', how='left')

# Convert to GeoDataFrame for export
merged_gdf = gpd.GeoDataFrame(merged, geometry='geometry', crs=g_app.crs)
merged_gdf.to_file(os.path.join(OUT, "appalachia_snapshot.geojson"), driver="GeoJSON")
print("Wrote snapshot geojson and cleaned tables to", OUT)

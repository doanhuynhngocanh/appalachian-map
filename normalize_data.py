#!/usr/bin/env python3
"""
Create normalized data tables for Appalachian Map Chatbot
"""
import pandas as pd
import os

# Set up paths
RAW = "data_raw"
OUT = "data_clean"
os.makedirs(OUT, exist_ok=True)

print("Creating normalized data tables...")

# --- 1) Create counties.csv as master list ---
print("Creating counties master list...")
arc = pd.read_csv(os.path.join(RAW, "arc_counties.csv"), dtype={"FIPS":str})
arc["fips"] = arc["FIPS"].astype(str).str.zfill(5)
arc = arc[["fips","COUNTY","STATE"]].rename(columns={"COUNTY":"county", "STATE":"state"})
arc["arc_flag"] = True
arc = arc.drop_duplicates(subset=["fips"])

# Save as counties.csv
arc.to_csv(os.path.join(OUT, "counties.csv"), index=False)
print(f"Created counties.csv with {len(arc)} Appalachian counties")

# --- 2) Normalize unemployment data ---
print("Normalizing unemployment data...")
bls = pd.read_csv(os.path.join(RAW, "bls_unemployment.csv"), dtype=str, encoding='latin-1')

# Clean and normalize unemployment data
bls_clean = bls.copy()
bls_clean['fips'] = bls_clean['fips'].astype(str).str.zfill(5)
bls_clean['year'] = pd.to_numeric(bls_clean['Year'], errors='coerce')
bls_clean['unemployment_rate'] = pd.to_numeric(bls_clean['unemployment_rate'], errors='coerce')

# Since this appears to be annual data (not monthly), we'll use month=1 for all records
bls_clean['month'] = 1

# Keep only Appalachian counties
bls_appalachian = bls_clean[bls_clean['fips'].isin(arc['fips'])].copy()

# Select and rename columns
unemployment_normalized = bls_appalachian[['fips', 'year', 'month', 'unemployment_rate']].copy()

# Save normalized unemployment data
unemployment_normalized.to_csv(os.path.join(OUT, "unemployment_normalized.csv"), index=False)
print(f"Created unemployment_normalized.csv with {len(unemployment_normalized)} records")

# --- 3) Load GDP per capita data ---
print("Loading GDP per capita data...")
gdp_data = pd.read_csv(os.path.join(OUT, "gdp.csv"))

# Ensure FIPS codes are properly formatted as 5-digit strings
gdp_data['fips'] = gdp_data['fips'].astype(str).str.zfill(5)

# The data already contains GDP per capita values, just rename column for consistency
gdp_with_per_capita = gdp_data.copy()
gdp_with_per_capita = gdp_with_per_capita.rename(columns={'gdp_per_capita': 'gdp_per_capita'})

# Save GDP with per capita
gdp_with_per_capita.to_csv(os.path.join(OUT, "gdp_with_per_capita.csv"), index=False)
print(f"Created gdp_with_per_capita.csv with {len(gdp_with_per_capita)} records")

# --- 4) Load and normalize income limits data ---
print("Loading income limits data...")
try:
    income_data = pd.read_csv(os.path.join(RAW, "hud_income_limits.csv"), dtype=str, encoding='latin-1')
    
    # Extract FIPS code (first 5 digits)
    income_data['fips'] = income_data['fips'].astype(str).str[:5].str.zfill(5)
    
    # Keep only Appalachian counties
    income_appalachian = income_data[income_data['fips'].isin(arc['fips'])].copy()
    
    # Convert median income to numeric
    income_appalachian['median_income'] = pd.to_numeric(income_appalachian['median2025'], errors='coerce')
    
    # Get latest median income per county
    latest_income = income_appalachian.groupby('fips')['median_income'].last().reset_index()
    latest_income['fips'] = latest_income['fips'].astype(str).str.zfill(5)
    
    print(f"Loaded income data for {len(latest_income)} counties")
except Exception as e:
    print(f"Warning: Could not load income data: {e}")
    latest_income = pd.DataFrame({'fips': [], 'median_income': []})

# --- 5) Create a summary statistics table ---
print("Creating summary statistics...")
latest_unemployment = unemployment_normalized.groupby('fips')['unemployment_rate'].last().reset_index()
latest_gdp = gdp_with_per_capita.groupby('fips')['gdp_per_capita'].last().reset_index()

# Ensure fips columns are strings for merging
latest_unemployment['fips'] = latest_unemployment['fips'].astype(str).str.zfill(5)
latest_gdp['fips'] = latest_gdp['fips'].astype(str).str.zfill(5)

# Merge with counties
summary = arc.merge(latest_unemployment, on='fips', how='left') \
             .merge(latest_gdp, on='fips', how='left')

# Merge income data if available
if len(latest_income) > 0:
    summary = summary.merge(latest_income, on='fips', how='left')

summary.to_csv(os.path.join(OUT, "county_summary.csv"), index=False)
print(f"Created county_summary.csv with {len(summary)} counties")

print("\nData normalization complete!")
print("Files created:")
print("- counties.csv: Master list of Appalachian counties")
print("- unemployment_normalized.csv: Normalized unemployment data")
print("- gdp_with_per_capita.csv: GDP data with per capita calculations")
print("- county_summary.csv: Summary statistics for each county")

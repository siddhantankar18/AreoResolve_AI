import pandas as pd
import os

print("Starting Step 05: The Grand Merge (Flight + Weather)...")

# 1. Paths
flight_file = './data/processed/clean_flight_data_600k.csv' 
weather_file = './data/weather/master_weather_2025.csv'
output_file = './data/processed/merged_flight_weather_data_600K.csv'

print("Loading flight and weather data...")
flights_df = pd.read_csv(flight_file, low_memory=False)
weather_df = pd.read_csv(weather_file, low_memory=False)

# Calculate Departure and Arrival hours from HHMM format
# (e.g., 1745 // 100 = 17)
flights_df['DEP_HOUR'] = (flights_df['CRSDepTime'] // 100) % 24
flights_df['ARR_HOUR'] = (flights_df['CRSArrTime'] // 100) % 24

# Prepare Origin Weather: Rename columns to *_origin
weather_origin = weather_df.rename(columns={
    'ORIGIN': 'Origin',
    'YEAR': 'Year',
    'MONTH': 'Month',
    'DAY_OF_MONTH': 'DayofMonth',
    'HOUR': 'DEP_HOUR',
    'temperature_2m': 'temp_origin',
    'precipitation': 'precip_origin',
    'wind_speed_10m': 'wind_origin',
    'visibility': 'visib_origin'
})

# Prepare Destination Weather: Rename columns to *_dest
weather_dest = weather_df.rename(columns={
    'ORIGIN': 'Dest', 
    'YEAR': 'Year',
    'MONTH': 'Month',
    'DAY_OF_MONTH': 'DayofMonth',
    'HOUR': 'ARR_HOUR',
    'temperature_2m': 'temp_dest',
    'precipitation': 'precip_dest',
    'wind_speed_10m': 'wind_dest',
    'visibility': 'visib_dest'
})

print("Merging Origin Weather...")
final_df = pd.merge(
    flights_df, 
    weather_origin, 
    how='left', 
    on=['Origin', 'Year', 'Month', 'DayofMonth', 'DEP_HOUR']
)

print("Merging Destination Weather...")
final_df = pd.merge(
    final_df, 
    weather_dest, 
    how='left', 
    on=['Dest', 'Year', 'Month', 'DayofMonth', 'ARR_HOUR']
)

print(f"\n--- MERGE COMPLETE ---")
print(f"Final dataset has {len(final_df):,} rows.")
print(f"Features: {final_df.columns.tolist()}")

# Save the final file
final_df.to_csv(output_file, index=False)
print(f"\n Success! Ultimate dataset saved to: {output_file}")
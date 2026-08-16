import pandas as pd
import requests
import airportsdata
from tqdm import tqdm
import os

# 1. AIRPORT LIST
file_path = "data/processed/clean_flight_data_600k.csv"

df = pd.read_csv(file_path)
target_airports = set(
    df["Origin"].dropna()
).union(
    df["Dest"].dropna()
)

weather_dir = "./data/weather"
os.makedirs(weather_dir, exist_ok=True)

print(f"Starting Weather Fetch for {len(target_airports)} airports...")

# 2. GET GPS AND DOWNLOAD WEATHER
airports = airportsdata.load('IATA')
master_weather_list = []

print("\nDownloading 2025 Weather Data from Open-Meteo...")
for airport_code in tqdm(target_airports, desc="Fetching Airport Weather"):
    # Safety check to ensure the code exists in the GPS library
    if airport_code not in airports:
        continue
        
    lat = airports[airport_code]['lat']
    lon = airports[airport_code]['lon']
    
    # Hit the Open-Meteo Archive for the ENTIRE year in one single call per airport
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2025-01-01&end_date=2025-12-31&hourly=temperature_2m,precipitation,wind_speed_10m,visibility&timezone=auto"
    
    response = requests.get(url, timeout = 30)
    
    if response.status_code == 200:
        data = response.json()
        if 'hourly' in data:
            df_weather = pd.DataFrame(data['hourly'])
            df_weather['ORIGIN'] = airport_code
            master_weather_list.append(df_weather)
    else:
        print(f" Failed to fetch weather for {airport_code}. Status: {response.status_code}")

# 3. PROCESS AND SAVE MASTER DATABASE
if not master_weather_list:
    print("\n No weather data was downloaded. Exiting.")
    exit()

print("\nProcessing and cleaning weather data...")
master_weather_df = pd.concat(master_weather_list, ignore_index=True)

# Convert the messy time string into clean Pandas columns (Year, Month, Day, Hour)
master_weather_df['time'] = pd.to_datetime(master_weather_df['time'])
master_weather_df['YEAR'] = master_weather_df['time'].dt.year
master_weather_df['MONTH'] = master_weather_df['time'].dt.month
master_weather_df['DAY_OF_MONTH'] = master_weather_df['time'].dt.day
master_weather_df['HOUR'] = master_weather_df['time'].dt.hour

# Drop the raw string time column to save memory
master_weather_df = master_weather_df.drop(columns=['time'])

# Save to a single master file to match your 600k master flight file!
output_path = os.path.join(weather_dir, "master_weather_2025.csv")
master_weather_df.to_csv(output_path, index=False)

print(f"\n Success! Master weather database saved to: {output_path}")
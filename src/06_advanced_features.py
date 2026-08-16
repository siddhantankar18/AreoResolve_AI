import pandas as pd
import numpy as np
import os

print("Starting advanced feature engineering...")

input_file = './data/processed/merged_flight_weather_data_600K.csv'
output_file = './data/processed/advanced_features_600K.csv'

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found.")
    exit()

print("Loading data...")
df = pd.read_csv(input_file, low_memory=False)
print(f"Original shape: {df.shape}")

print("Engineering cyclic time features...")
df['dep_hour_sin'] = np.sin(2 * np.pi * df['DEP_HOUR'] / 24)
df['dep_hour_cos'] = np.cos(2 * np.pi * df['DEP_HOUR'] / 24)
df['arr_hour_sin'] = np.sin(2 * np.pi * df['ARR_HOUR'] / 24)
df['arr_hour_cos'] = np.cos(2 * np.pi * df['ARR_HOUR'] / 24)

df['is_morning_rush'] = ((df['DEP_HOUR'] >= 6) & (df['DEP_HOUR'] <= 9)).astype(int)
df['is_evening_rush'] = ((df['DEP_HOUR'] >= 15) & (df['DEP_HOUR'] <= 19)).astype(int)

print("Engineering weather severity scores...")
df['weather_severity_origin'] = (
    df['precip_origin'].fillna(0) * 2 + 
    df['wind_origin'].fillna(0) * 0.5 + 
    (10 - df['visib_origin'].fillna(10))
)

df['weather_severity_dest'] = (
    df['precip_dest'].fillna(0) * 2 + 
    df['wind_dest'].fillna(0) * 0.5 + 
    (10 - df['visib_dest'].fillna(10))
)

df['storm_flag'] = (
    (df['precip_origin'] > 5) | (df['precip_dest'] > 5) | 
    (df['wind_origin'] > 35) | (df['wind_dest'] > 35)
).astype(int)

print("Engineering traffic volume metrics...")
df['ROUTE'] = df['Origin'].astype(str) + '_' + df['Dest'].astype(str)

df['origin_traffic_volume'] = df.groupby('Origin')['Origin'].transform('count')
df['dest_traffic_volume'] = df.groupby('Dest')['Dest'].transform('count')
df['route_traffic_volume'] = df.groupby('ROUTE')['ROUTE'].transform('count')

print("Handling missing values and saving...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

df.to_csv(output_file, index=False)

print(f"Feature engineering complete. New shape: {df.shape}")
print(f"Saved to: {output_file}")
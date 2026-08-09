import pandas as pd
import os

# 1. Setup Input/Output paths
input_file = './data/processed/master_flight_data_600k.csv'
output_file = './data/processed/clean_flight_data_600k.csv'
os.makedirs('./data/processed', exist_ok=True)

print(" Starting Step 03: Data Cleaning & Feature Selection...")

# 2. Load the Master 600k file
if not os.path.exists(input_file):
    print(f" Error: {input_file} not found. Please run Step 02 first!")
else:
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Original dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # 3. THE WHITELIST: Keeping only predictive features
    # We keep Schedule, Airline, and Geography info + our target.
    columns_to_keep = [
        'Year', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate',
        'Reporting_Airline', 'Origin', 'Dest', 
        'CRSDepTime', 'CRSArrTime', 'Distance',
        'ArrDel15'  # Our Target (The Answer Key)
    ]

    # Verify if these columns exist in our current master file
    actual_columns = [col for col in columns_to_keep if col in df.columns]
    
    # 4. Filter the dataframe to just these columns
    df_clean = df[actual_columns]
    
    # Save the cleaned file
    df_clean.to_csv(output_file, index=False)

    print(f" Cleaned dataset created: {df_clean.shape[0]:,} rows, {df_clean.shape[1]} columns")
    print(f" Saved to: {output_file}")
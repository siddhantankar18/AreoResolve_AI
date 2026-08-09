import pandas as pd
import glob
import os
from tqdm import tqdm

# ==========================================
# 1. SETUP & CONFIGURATION (LOCAL MAC)
# ==========================================
raw_data_dir = "./data/raw"
output_dir = "./data/processed"
os.makedirs(output_dir, exist_ok=True)

# The Magic Numbers: 40k Normal + 10k Delayed = 50,000 rows per month (80/20 split)
# This perfectly preserves the real-world imbalance while saving RAM!
TARGET_NORMAL = 40000    
TARGET_ABNORMAL = 10000  

flight_files = sorted(glob.glob(os.path.join(raw_data_dir, "*.csv")))

if not flight_files:
    print(f"No files found in {raw_data_dir}. Make sure you are in the ML_project folder!")
    exit()

print(f"\nStarting Step 02: Targeted Undersampling...")
print(f"Goal: {TARGET_NORMAL} Normal & {TARGET_ABNORMAL} Delayed flights per month.\n")

# This empty list will hold all 12 of our 50k dataframes
all_sampled_months = []

# ==========================================
# 2. PROCESS EACH MONTH
# ==========================================
for file in tqdm(flight_files, desc="Sampling Months"):
    filename = os.path.basename(file)
    
    # 1. Find the target column safely
    headers = pd.read_csv(file, nrows=0).columns
    potential_targets = [col for col in headers if 'DEL15' in col.upper() or 'DELAY' in col.upper()]
    target_col = next((c for c in potential_targets if 'ARR_DEL15' in c.upper() or 'ARRDEL15' in c.upper()), None)
    
    if not target_col:
        print(f"\nSkipping {filename}: No delay column found.")
        continue

    # 2. Load the CSV
    df = pd.read_csv(file, low_memory=False)
    
    # Clean the target column to ensure it is numeric
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    
    # 3. Separate into buckets
    df_normal = df[df[target_col] == 0.0]
    df_delayed = df[df[target_col] == 1.0]
    
    # 4. Perform the Targeted Sampling!
    try:
        sampled_normal = df_normal.sample(n=TARGET_NORMAL, random_state=42)
        sampled_delayed = df_delayed.sample(n=TARGET_ABNORMAL, random_state=42)
    except ValueError as e:
        print(f"\n Error in {filename}: Not enough rows to meet your target. Details: {e}")
        continue
        
    # 5. Combine and shuffle the month
    final_50k_df = pd.concat([sampled_normal, sampled_delayed])
    final_50k_df = final_50k_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add to our master list
    all_sampled_months.append(final_50k_df)

# ==========================================
# 3. THE GRAND MERGE
# ==========================================
print("\n Merging all 12 months into a single Master Dataset...")

master_df = pd.concat(all_sampled_months, ignore_index=True)

# Save the final giant 600,000 row CSV
output_path = os.path.join(output_dir, "master_flight_data_600k.csv")
master_df.to_csv(output_path, index=False)

print(f"\n Success! You now have a single file containing exactly {len(master_df):,} rows.")
print(f"Saved to: {output_path}")
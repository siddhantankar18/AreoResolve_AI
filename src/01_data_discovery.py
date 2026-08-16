import os
import pandas as pd

def main():
    folder_path = "data/raw"
    
    if not os.path.exists(folder_path):
        print(f"Directory '{folder_path}' does not exist.")
        return

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in '{folder_path}' folder.")
        return
        
    first_file = csv_files[0]
    file_path = os.path.join(folder_path, first_file)
    
    print(f"Reading file: {file_path}")
    
    # Read ONLY the first 1000 rows to save memory
    df = pd.read_csv(file_path, nrows=1000)
    
    print("\n--- Column Names ---")
    print(df.columns.tolist())
    
    print("\n--- info ---")
    print(df.info())
    
    print("\n--- null values ---")
    print(df.isnull().sum())
    
    print("\n--- describe ---")
    print(df.describe())
    
    print("\n--- First 10 Rows ---")
    print(df)

if __name__ == "__main__":
    main()

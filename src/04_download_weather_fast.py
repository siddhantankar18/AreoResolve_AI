import pandas as pd
import requests
import airportsdata
from tqdm import tqdm
import os

# ==========================================
# 1. PASTE YOUR AIRPORT LIST HERE
# ==========================================
# Replace this list with the exact output from your Jupyter Notebook!
target_airports = ['ISP', 'SBP', 'MFE', 'HLN', 'MEI', 'CDC', 'GPT', 'GRR', 'LEX', 'PIT', 'SPN',
 'PSP', 'RHI', 'IDA', 'CRW', 'PIR', 'LAN', 'SGF', 'IAG', 'DAL', 'COS', 'HHH',
 'CLE', 'ABR', 'MSY', 'BPT', 'LAS', 'MRY', 'DTW', 'ABY', 'JMS', 'LGA', 'MIA',
 'LIH', 'EWN', 'ONT', 'CID', 'OAJ', 'FNT', 'GFK', 'BOS', 'SJU', 'ABQ', 'TUL',
 'SCC', 'OAK', 'MOT', 'AMA', 'JNU', 'HYA', 'BTM', 'SAN', 'ERI', 'LAW', 'DEN',
 'DLH', 'BLI', 'PIH', 'MCW', 'MHK', 'MSN', 'OTH', 'ACY', 'PSE', 'PSM', 'STC',
 'ABE', 'ALB', 'BGM', 'LNK', 'SFO', 'ROW', 'BTV', 'LAX', 'SAV', 'OKC', 'PRC',
 'BIH', 'DDC', 'SJC', 'DSM', 'CLD', 'JAX', 'MAF', 'BRD', 'MEM', 'DVL', 'SMX',
 'PSG', 'GST', 'FCA', 'YAK', 'PAE', 'GEG', 'RFD', 'PQI', 'LBE', 'CRP', 'OMA',
 'EUG', 'FSM', 'IMT', 'MLB', 'MBS', 'VCT', 'ADQ', 'STX', 'GSO', 'CHS', 'KTN',
 'DEC', 'FAR', 'HTS', 'LGB', 'TRI', 'MFR', 'DFW', 'APN', 'MGM', 'CAK', 'LBB',
 'TWF', 'AGS', 'AEX', 'PGD', 'PVU', 'AUS', 'BTR', 'EAR', 'CPR', 'MLU', 'DHN',
 'JAC', 'BIL', 'GNV', 'BGR', 'VRB', 'YUM', 'LFT', 'HYS', 'HSV', 'TXK', 'BHM',
 'ILM', 'ESC', 'PIE', 'SNA', 'RIW', 'JST', 'MOB', 'CMI', 'CIU', 'KOA', 'CMH',
 'COU', 'CLT', 'RKS', 'BJI', 'RDM', 'HPN', 'MLI', 'FWA', 'INL', 'LSE', 'LCK',
 'AKN', 'SDF', 'GUF', 'AZA', 'ITO', 'SCE', 'BMI', 'GRB', 'RNO', 'SEA', 'LRD',
 'GGG', 'ATL', 'STL', 'PNS', 'GSP', 'AVL', 'OGG', 'ABI', 'MDW', 'DAY', 'GRI',
 'MVY', 'BFL', 'RIC', 'FLL', 'SLN', 'BZN', 'SYR', 'STT', 'CVG', 'FSD', 'SHR',
 'EWR', 'CLL', 'BET', 'CHO', 'SGU', 'JFK', 'RAP', 'BLV', 'FLG', 'ELP', 'SBN',
 'ATW', 'HIB', 'USA', 'PPG', 'HDN', 'ADK', 'BWI', 'AVP', 'MDT', 'TYR', 'BDL',
 'DCA', 'RDU', 'SAT', 'BIS', 'MTJ', 'ORD', 'PDX', 'HRL', 'GTF', 'XWA', 'PSC',
 'OME', 'SFB', 'LCH', 'SPI', 'SUN', 'BUF', 'HGR', 'LBL', 'RSW', 'GUC', 'TYS',
 'HNL', 'COD', 'TUS', 'IAD', 'BFF', 'EVV', 'CAE', 'SMF', 'STS', 'FOD', 'SUX',
 'HOB', 'PIA', 'FAT', 'LIT', 'RDD', 'LAR', 'SCK', 'MCO', 'LBF', 'JAN', 'BNA',
 'SRQ', 'CDV', 'BUR', 'PHX', 'GUM', 'GTR', 'BOI', 'ICT', 'XNA', 'MHT', 'WYS',
 'PIB', 'ROA', 'MYR', 'GCC', 'CYS', 'TPA', 'FMN', 'ACT', 'MGW', 'ELM', 'DAB',
 'TLH', 'FAY', 'AZO', 'PWM', 'CWA', 'DIK', 'MCI', 'BRO', 'BRW', 'PBG', 'SWF',
 'JLN', 'CKB', 'BQN', 'PHL', 'EKO', 'OTZ', 'DLG', 'LAF', 'HOU', 'ITH', 'SWO',
 'ROC', 'IAH', 'ECP', 'PHF', 'WRG', 'ACV', 'SLC', 'ASE', 'SJT', 'RST', 'CMX',
 'SAF', 'MSP', 'EAU', 'PVD', 'FAI', 'GJT', 'SBA', 'SIT', 'IND', 'VPS', 'GCK',
 'EGE', 'TVC', 'ANC', 'EYW', 'ORH', 'ALO', 'SPS', 'MKE', 'PLN', 'SHV', 'GRK',
 'ORF', 'TTN', 'ACK', 'TOL', 'DRO', 'CHA', 'LWS', 'PBI', 'MQT', 'ATY', 'MSO']

weather_dir = "./data/weather"
os.makedirs(weather_dir, exist_ok=True)

print(f"🚀 Starting Weather Fetch for {len(target_airports)} airports...")

# ==========================================
# 2. GET GPS AND DOWNLOAD WEATHER
# ==========================================
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
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if 'hourly' in data:
            df_weather = pd.DataFrame(data['hourly'])
            df_weather['ORIGIN'] = airport_code
            master_weather_list.append(df_weather)
    else:
        print(f" Failed to fetch weather for {airport_code}. Status: {response.status_code}")

# ==========================================
# 3. PROCESS AND SAVE MASTER DATABASE
# ==========================================
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
"""
Module Name: process_data.py
Description: used to join together ISO-NE spreadsheets and convert data to EST
Usage: scripts/python process_dara.py
"""

import pandas as pd
from pathlib import Path

#Get path
file_path = Path.cwd().parent / "data" 
files = list((file_path / "raw").glob("*.xlsx"))
df_list = []

#Loop through excel files
for file in files:
    df = pd.read_excel(file, sheet_name='NEMA')
    df = df[['Date', 'Hr_End', 'RT_LMP']]
    df_list.append(df)

combo = pd.concat(df_list, ignore_index=True)


combo['Hr_End'] = combo['Hr_End'].astype(str)
combo['Hr_Start'] = combo['Hr_End'].str.replace(r'\D', '', regex=True).astype(int) - 1 #remove coding for DST
combo['Temp_Timestamp'] = combo['Date'] + pd.to_timedelta(combo['Hr_Start'], unit = 'h')
combo = combo.sort_values(by=['Date', 'Temp_Timestamp']).reset_index(drop=True)

#Split the data based on convention of daylight savings in ISO-NE code
df_old = combo[combo['Date'].dt.year < 2024].copy()
df_new = combo[combo['Date'].dt.year >= 2024].copy()


df_new['Timestamp'] = (
    df_new['Temp_Timestamp']
    .dt.tz_localize('America/New_York', ambiguous='infer', nonexistent=pd.Timedelta(hours=-1)) #handle DST, duplicate 
    .dt.tz_convert('EST')
)

df_old['Timestamp'] = (
    df_old['Temp_Timestamp']
    .dt.tz_localize('EST')
)

final_df = pd.concat([df_old, df_new], ignore_index=True)
final_df = final_df.sort_values('Timestamp').reset_index(drop=True)
final_df = final_df[['Timestamp', 'RT_LMP']]

final_df.to_parquet(file_path / "processed" / "NEMA_LMP.parquet", engine='pyarrow', index=False)
import pandas as pd
import torch
from torch.utils.data import Dataset

class NEMADataset(Dataset):
    def __init__(self, file, start_date=None, end_date=None, input_len=512, output_len=24, include_time=False):
        
        self.input_len = input_len
        self.output_len = output_len
        self.include_time = include_time
        
        #Load the data
        df = pd.read_parquet(file, engine='pyarrow')
        if start_date:
            start_ts = pd.to_datetime(start_date).tz_localize('EST')
        else:
            start_ts = df['Timestamp'].min()

        if end_date:
            #Go to the end of the day 
            end_ts = pd.to_datetime(end_date).replace(hour=23, minute=59).tz_localize('EST')
        else:
            end_ts = df['Timestamp'].max()

        #Find indices for dates
        start_idx = df[df['Timestamp'] >= start_ts].index[0]
        end_idx = df[df['Timestamp'] <= end_ts].index[-1]
        
        #Reach back input_len to be able to predict on start_date (if there is enough data)
        pad_start_idx = max(0, start_idx - self.input_len)
        
        # Store timestamps if needed for temporal features
        if self.include_time:
            self.timestamps = df.loc[pad_start_idx : end_idx, 'Timestamp'].values
        
        #Select relevant data
        self.data = df.loc[pad_start_idx : end_idx, 'RT_LMP'].values.reshape(-1,1) #Add training dimension: shape (timestamps, 1)
        
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return (len(self.data) - self.input_len - self.output_len) // self.output_len + 1

    def __getitem__(self, idx):
        x = self.data[idx * self.output_len : idx * self.output_len + self.input_len]
        y = self.data[idx * self.output_len + self.input_len : idx * self.output_len + self.input_len + self.output_len]
        
        x = x.transpose(0, 1) #transpose to MOMENT input shape (channel (1), timestamps)
        y = y.squeeze(-1) #flatten back to shape (timestamps)
        
        # Extract temporal features if requested
        if self.include_time:
            # Get the timestamps for the output window
            output_start_idx = idx * self.output_len + self.input_len
            output_timestamps = self.timestamps[output_start_idx : output_start_idx + self.output_len]
            
            # Extract hour of day and day of week
            hours = torch.tensor([pd.Timestamp(ts).hour for ts in output_timestamps], dtype=torch.float32)
            days_of_week = torch.tensor([pd.Timestamp(ts).dayofweek for ts in output_timestamps], dtype=torch.float32)
            
            return x, y, hours, days_of_week
        
        return x, y
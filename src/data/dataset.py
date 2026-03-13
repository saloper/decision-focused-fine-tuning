import pandas as pd
import torch
from torch.utils.data import Dataset

class NEMADataset(Dataset):
    def __init__(self, file, start_date, end_date, input_len=512, output_len=24):
        
        self.input_len = input_len
        self.output_len = output_len
        
        #Load the data
        df = pd.read_parquet(file, engine='pyarrow')
        
        #Find indices for dates
        start_idx = df[df['Timestamp'] >= start_date].index[0]
        end_idx = df[df['Timestamp'] <= end_date].index[-1]
        
        #Reach back input_len to be able to predict on start_date (if there is enough data)
        pad_start_idx = max(0, start_idx - self.input_len)
        
        #Select relevant data
        self.data = df.loc[pad_start_idx : end_idx, 'RT_LMP'].values.reshape(-1,1)
        
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.input_len - self.output_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_len]
        y = self.data[idx + self.input_len : idx + self.input_len + self.output_len]
        
        x = x.transpose(0, 1)
        y = y.squeeze(-1)
        return x, y
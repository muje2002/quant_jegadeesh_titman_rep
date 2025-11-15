import pandas as pd
import numpy as np
import zipfile
import os
import time

data_dir='data'
file_name='CRSPm19652024'
start_date='1980-01-01'
end_date='1989-12-31'

file_path = os.path.join(data_dir, file_name)
df = pd.read_csv(file_path, low_memory=False)

df['date'] = pd.to_datetime(df['date'])
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

df.to_excel('df.xlsx', index=False)
import pandas as pd
import pickle
import numpy as np
filename="C:\\mlops_boot_camp\\mlops-zoomcamp\\homework_5\\data\\green_tripdata_2024-03.parquet"
march_2024 = pd.read_parquet(filename)
print(f'Shape of the DataFrame: {march_2024.shape}')


march_2024["date"] = march_2024["lpep_pickup_datetime"].dt.date

daily_medians = march_2024.groupby("date")["fare_amount"].median()

max_value = daily_medians.max()
print(f'Maximum median fare amount: {max_value}')
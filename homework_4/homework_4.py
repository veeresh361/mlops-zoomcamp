# import sys
import pandas as pd
import pickle
import numpy as np

model_path="C:\\mlops_boot_camp\\mlops-zoomcamp\\homework_3_orchestration\\models\\linear_regression_model.pkl"
vectorizer_path="C:\\mlops_boot_camp\\mlops-zoomcamp\\homework_3_orchestration\\models\\dv.pkl"
# # year = int(sys.argv[1])
# # month = int(sys.argv[2])

input_file = f'C:\\mlops_boot_camp\\mlops-zoomcamp\\homework_3_orchestration\\data\\yellow_tripdata_2023-03.parquet'

with open(model_path, 'rb') as f_in:
    model = pickle.load(f_in)

with open(vectorizer_path, 'rb') as f_in:
    vectorizer = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data(input_file)
dicts = df[categorical].to_dict(orient='records')
X_val = vectorizer.transform(dicts)
y_pred = model.predict(X_val)

std_pred = np.std(y_pred)
print(f'Standard deviation of the predicted duration: {std_pred}')


# import os

# file_path = r"C:\mlops_boot_camp\mlops-zoomcamp\homework_4\data\homework_4_results.parquet"

# size_bytes = os.path.getsize(file_path)

# print(f"File size in bytes: {size_bytes}")
# print(f"File size in KB: {size_bytes / 1024:.2f}")
# print(f"File size in MB: {size_bytes / (1024 * 1024):.2f}")

# json_path="C:\\mlops_boot_camp\\mlops-zoomcamp\\Pipfile.lock"
# import json

# with open(json_path, "r") as f:
#     lock_data = json.load(f)

# print('scikit-learn Hash:',lock_data["default"]["scikit-learn"]["hashes"][0])

import sklearn
print(sklearn.__version__)
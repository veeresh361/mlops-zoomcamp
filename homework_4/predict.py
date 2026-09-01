import sys
import pickle
import pandas as pd

model_path = "models/linear_regression_model.pkl"
vectorizer_path = "models/dv.pkl"

categorical = ['PULocationID', 'DOLocationID']

with open(model_path, 'rb') as f_in:
    model = pickle.load(f_in)

with open(vectorizer_path, 'rb') as f_in:
    vectorizer = pickle.load(f_in)

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def main():
    filename = sys.argv[1]
    df = read_data(filename)

    dicts = df[categorical].to_dict(orient='records')
    X_val = vectorizer.transform(dicts)
    y_pred = model.predict(X_val)

    print(y_pred.mean())

if __name__ == "__main__":
    main()
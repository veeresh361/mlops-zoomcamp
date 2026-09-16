import os
import sys
import pickle

import numpy as np
import pandas as pd


def get_input_path(year, month):
    default_input_pattern = (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/"
        "yellow_tripdata_{year:04d}-{month:02d}.parquet"
    )

    input_pattern = os.getenv(
        "INPUT_FILE_PATTERN",
        default_input_pattern
    )

    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = (
        "s3://nyc-duration-prediction/"
        "yellow_tripdata_{year:04d}-{month:02d}.parquet"
    )

    output_pattern = os.getenv(
        "OUTPUT_FILE_PATTERN",
        default_output_pattern
    )

    return output_pattern.format(year=year, month=month)


def get_s3_options():
    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")

    if s3_endpoint_url:
        options = {
            "key": "test",
            "secret": "test",
            "client_kwargs": {
                "endpoint_url": s3_endpoint_url
            }
        }

        return options

    return None


def prepare_data(df, categorical):
    df["duration"] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    )
    df["duration"] = df["duration"].dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = (
        df[categorical]
        .fillna(-1)
        .astype("int")
        .astype("str")
    )

    return df


def read_data(filename, categorical):
    options = get_s3_options()

    if options:
        df = pd.read_parquet(
            filename,
            storage_options=options
        )
    else:
        df = pd.read_parquet(filename)

    df = prepare_data(df, categorical)

    return df


def main(year, month):
    model_path = (
        r"C:\mlops_boot_camp\mlops-zoomcamp"
        r"\homework_3_orchestration\models\linear_regression_model.pkl"
    )

    vectorizer_path = (
        r"C:\mlops_boot_camp\mlops-zoomcamp"
        r"\homework_3_orchestration\models\dv.pkl"
    )

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    categorical = ["PULocationID", "DOLocationID"]

    with open(model_path, "rb") as f_in:
        model = pickle.load(f_in)

    with open(vectorizer_path, "rb") as f_in:
        vectorizer = pickle.load(f_in)

    df = read_data(input_file, categorical)

    dicts = df[categorical].to_dict(orient="records")
    X_val = vectorizer.transform(dicts)

    y_pred = model.predict(X_val)

    print(f"Input file: {input_file}")
    print(f"Number of input records: {len(df)}")
    print(f"Mean predicted duration: {np.mean(y_pred):.2f}")
    print(f"Standard deviation: {np.std(y_pred):.2f}")

    df_result = pd.DataFrame()
    df_result["predicted_duration"] = y_pred

    options = get_s3_options()

    if options:
        df_result.to_parquet(
            output_file,
            engine="pyarrow",
            compression=None,
            index=False,
            storage_options=options
        )
    else:
        df_result.to_parquet(
            output_file,
            engine="pyarrow",
            compression=None,
            index=False
        )

    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    main(year, month)
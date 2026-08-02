import pickle
from pathlib import Path

import pandas as pd
from prefect import flow, task, get_run_logger
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import mlflow


# -----------------------------
# Configuration
# -----------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-prefect-linear-regression_2")

MODELS_FOLDER = Path("models")
MODELS_FOLDER.mkdir(exist_ok=True)


# -----------------------------
# Tasks
# -----------------------------
@task(name="Load Raw Data")
def load_raw_data(file_path: str) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Loading raw data from: {file_path}")

    df = pd.read_parquet(file_path)

    print(f"Q1: How many records did we load? {len(df)}")
    logger.info(f"Raw records loaded: {len(df)}")

    return df


@task(name="Prepare Data")
def read_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Starting dataframe preprocessing")

    df = df.copy()

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    print(f"Q2: After applying read_dataframe, number of rows = {len(df)}")
    logger.info(f"Records after preprocessing: {len(df)}")
    return df


@task(name="Create Features")
def create_X(df: pd.DataFrame, dv: DictVectorizer = None):
    logger = get_run_logger()
    logger.info("Creating feature matrix")

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dicts = df[categorical + numerical].to_dict(orient="records")

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
        logger.info("Fitted a new DictVectorizer")
    else:
        X = dv.transform(dicts)
        logger.info("Used existing DictVectorizer to transform data")

    return X, dv


@task(name="Train Linear Regression Model")
def train_model(X_train, y_train):
    logger = get_run_logger()
    logger.info("Training Linear Regression model with default parameters")

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"Q3: Intercept of the model = {model.intercept_}")
    logger.info(f"Model intercept: {model.intercept_}")

    return model


@task(name="Evaluate Model")
def evaluate_model(model, X_val, y_val) -> float:
    logger = get_run_logger()
    logger.info("Evaluating model")

    y_pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)

    print(f"Validation RMSE: {rmse:.4f}")
    logger.info(f"Validation RMSE: {rmse:.4f}")

    return rmse


@task(name="Save Artifacts")
def save_artifacts(model, dv) -> tuple[str, str]:
    logger = get_run_logger()
    logger.info("Saving model and preprocessor artifacts")

    dv_path = MODELS_FOLDER / "dv.pkl"
    model_path = MODELS_FOLDER / "linear_regression_model.pkl"

    with open(dv_path, "wb") as f_out:
        pickle.dump(dv, f_out)

    with open(model_path, "wb") as f_out:
        pickle.dump(model, f_out)

    logger.info(f"Saved DictVectorizer to {dv_path}")
    logger.info(f"Saved model to {model_path}")

    return str(dv_path), str(model_path)


@task(name="Log to MLflow")
def log_to_mlflow(model, dv_path: str, model_path: str) -> str:
    logger = get_run_logger()
    logger.info("Logging experiment to MLflow")

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features_categorical", ['PULocationID', 'DOLocationID'])
        mlflow.log_param("features_numerical", ["trip_distance"])
        mlflow.log_metric("intercept", float(model.intercept_))

        mlflow.log_artifact(dv_path, artifact_path="preprocessor")
        mlflow.log_artifact(model_path, artifact_path="model")

        run_id = run.info.run_id
        logger.info(f"MLflow run logged successfully with run_id: {run_id}")

    print(f"MLflow run_id: {run_id}")
    return run_id


# -----------------------------
# Flow
# -----------------------------
@flow(name="NYC Taxi Linear Regression Pipeline")
def taxi_duration_pipeline(train_path: str) -> str:
    logger = get_run_logger()
    logger.info("Starting NYC Taxi Linear Regression Prefect pipeline")

    # Load raw data
    df_train_raw = load_raw_data(train_path)
   

    # Prepare data
    df_train = read_dataframe(df_train_raw)

    # Create target
    target = "duration"
    y_train = df_train[target].values

    # Create feature matrices
    X_train, dv = create_X(df_train)

    # Train model
    model = train_model(X_train, y_train)

    # Save artifacts
    dv_path, model_path = save_artifacts(model, dv)

    # Log to MLflow
    run_id = log_to_mlflow(model, dv_path, model_path)

    logger.info("Pipeline completed successfully")
    return run_id


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a Linear Regression model to predict taxi trip duration using Prefect."
    )
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Local path to the training parquet file"
    )

    args = parser.parse_args()

    run_id = taxi_duration_pipeline(
        train_path=args.train_path
    )

    with open("run_id.txt", "w") as f:
        f.write(run_id)

    print("Pipeline finished successfully.")
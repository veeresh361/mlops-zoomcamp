from pathlib import Path

model_path = Path("models/linear_regression_model.pkl")
dv_path = Path("models/dv.pkl")

print("Model size in bytes:", model_path.stat().st_size)
print("Model size in KB:", model_path.stat().st_size / 1024)

print("Vectorizer size in bytes:", dv_path.stat().st_size)
print("Vectorizer size in KB:", dv_path.stat().st_size / 1024)
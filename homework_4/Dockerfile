FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .
COPY data ./data
COPY models ./models

ENTRYPOINT ["python", "predict.py"]
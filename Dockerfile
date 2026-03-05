FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    optuna \
    psycopg2-binary

COPY src/journal_2026/trial.py .

CMD ["python", "trial.py"]
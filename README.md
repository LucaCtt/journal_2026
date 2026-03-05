# Optuna + AWS Batch: Autoencoder HPO

## Prerequisites

- PostgreSQL RDS instance (must be reachable from both your machine and AWS Batch VPC)
- AWS Batch job queue + compute environment
- ECR repository

## Setup

### 1. Create the Optuna DB

```sql
CREATE DATABASE optuna;
```

### 2. Build & push the Docker image

```bash
# Replace with your ECR URI
ECR_URI=123456789.dkr.ecr.us-east-1.amazonaws.com/optuna-trial

aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
docker build -t $ECR_URI:latest .
docker push $ECR_URI:latest
```

### 3. Create a Batch Job Definition

Key settings:
- **Image**: your ECR URI
- **vCPUs**: 2–4  |  **Memory**: 4096–8192 MB
- **Job role**: IAM role with at minimum `AmazonECSTaskExecutionRolePolicy`
- **Network**: same VPC/subnets as your RDS instance (or use a security group that allows port 5432)

The container receives these env vars from the launcher (no need to bake them in):
```
OPTUNA_STORAGE
OPTUNA_STUDY
OPTUNA_TRIAL_ID
```

> **Tip**: Store the DB password in AWS Secrets Manager and inject it via the job definition's `secrets` field instead of passing it in plain text.

### 4. Run the launcher

```bash
pip install optuna boto3 psycopg2-binary

python launcher.py \
    --study my_ae_study \
    --storage "postgresql+psycopg2://user:pass@your-rds-host:5432/optuna" \
    --n-trials 20 \
    --job-queue  my-batch-queue \
    --job-def    optuna-trial:1 \
    --region     us-east-1
```

## How it works

```
launcher.py
  └─ study.ask()           ← Optuna samples params, assigns trial_id
  └─ batch.submit_job()    ← passes trial_id + storage URL as env vars
       └─ trial.py (container)
            └─ study.optimize(objective, n_trials=1)
                 └─ trains AE + classifier, returns accuracy
                 └─ Optuna writes result back to RDS
```

## Tune what's searched

Edit the `objective()` function in `trial.py` — all `trial.suggest_*` calls
define the search space.  The launcher and Batch wiring don't need to change.
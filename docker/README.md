# Containerized Batch Scoring

This directory contains the Docker configuration for the `claims-fraud` batch scoring pipeline, providing a secure, reproducible runtime environment.

## Architecture & Data Handling

To keep the container lightweight and secure, **trained models and CSV datasets are not baked into the image**. Instead, they are decoupled and passed dynamically at runtime using Docker volume mounting. 

## 1. Building the Image

Run the build command from the **root of the project** (not inside the `docker/` folder) so that Docker can correctly copy the source code and configuration files.

```bash
sudo docker build -f docker/Dockerfile -t claims-fraud-scoring .
```

## 2. Running the Batch Scoring Script

The container is configured to automatically use the score-claims CLI tool. Execute the container by mounting your local project's data directory (data/) to /app/data inside the container:

```bash
sudo docker run --rm \
  -v $(pwd)/data:/app/data \
  claims-fraud-scoring \
  --input /app/data/input_claims.csv \
  --model /app/data/model.joblib \
  --output /app/data/output_scored.csv
```

### Argument Breakdown:
- --rm: Automatically cleans up the container after it finishes running.

- -v $(pwd)/data:/app/data: Maps your local data folder to the /app/data folder inside the container.

- --input, --model, --output: Arguments passed directly to the score-claims script.
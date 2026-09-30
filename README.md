# Vehicle Insurance Response Prediction — MLOps Project

An end-to-end machine-learning project that predicts whether a customer will be interested in vehicle insurance (`Response`). It uses a modular Python pipeline, **SQLite** for source data, **scikit-learn** for modeling, **FastAPI** for serving, **Docker** for packaging, and **GitHub Actions** for CI and container publishing.

> **Current status:** The application and prediction form have been tested in a Docker container, and the GitHub Actions workflow has passed. Training from inside Docker requires mounting the local SQLite database (see [Run with Docker](#run-with-docker)). Model storage is currently **local**, not AWS/GCP. MLflow tracking and cloud deployment are potential future additions, not implemented features documented here.

## Features implemented

- Reads vehicle-insurance data from a local SQLite database instead of MongoDB Atlas.
- Saves an ingestion snapshot and creates reproducible training/testing CSV splits.
- Validates required dataset columns against `config/schema.yaml` and produces a validation report.
- Encodes categorical features, scales selected numerical columns, and handles training-class imbalance using SMOTEENN.
- Trains a `RandomForestClassifier` and calculates classification metrics including F1, precision, and recall.
- Compares a newly trained model with an existing locally registered model, applying a configurable improvement threshold.
- Saves a combined preprocessing/model object and promotes accepted models to a local registry.
- Serves an HTML prediction form and a training endpoint through FastAPI.
- Builds and tests the application with GitHub Actions; publishes a Docker image to GitHub Container Registry (GHCR).
- Uses reusable configurations, artifacts, logging, and custom exception handling across pipeline stages.

## Tech stack

| Area | Technology |
| --- | --- |
| Programming and analysis | Python, pandas, NumPy |
| Source database | SQLite (`sqlite3`) |
| Preprocessing | scikit-learn, imbalanced-learn (SMOTEENN) |
| Model | Random Forest classifier |
| API and interface | FastAPI, Uvicorn, Jinja2, HTML/CSS |
| Packaging | Docker |
| CI/CD | GitHub Actions, GitHub Container Registry |
| Model registry | Local directory (`model_registry/`) |

## Architecture

```text
SQLite (notebook/vehicle.db)
          |
          v
   Data ingestion
   |         |
   v         v
 train.csv  test.csv      + feature_store/data.csv
   \         /
    v       v
   Data validation ------> validation report
          |
          v
 Data transformation
 (encoding + scaling)
 (SMOTEENN: train only)
          |
          v
   Model training
   Random Forest
          |
          v
   Model evaluation
 Compare F1 with existing
 local registered model
          |
      if accepted
          v
    Model pusher
 model_registry/model.pkl
          |
          v
 FastAPI prediction form
```

**Important:** The test split should remain untouched by resampling. Fit preprocessing on training data, then apply the same fitted preprocessing to testing and inference inputs. The API's input feature names and ordering must match those used during training.

## Project structure

Key directories/files in the project (some generated directories appear only after running the pipeline):

```text
.
├── .github/
│   └── workflows/
│       └── cicd.yml
├── config/
│   └── schema.yaml
├── notebook/
│   ├── data.csv
│   └── vehicle.db              # local; do not commit by default
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   ├── configuration/
│   │   └── sqlite_db_connection.py
│   ├── constants/
│   ├── data_access/
│   │   └── vehicle_data.py
│   ├── entity/
│   │   ├── config_entity.py
│   │   ├── artifact_entity.py
│   │   └── estimator.py
│   ├── exception/
│   ├── logger/
│   ├── pipline/                # directory name used in current project
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   └── utils/
├── static/
├── templates/
│   └── vehicledata.html
├── tests/
│   └── test_app.py
├── artifact/                  # generated per pipeline run
├── model_registry/            # locally registered model
├── app.py
├── demo.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

## Dataset

The project uses a vehicle-insurance response dataset. The local database table is named `vehicle_data`; the ingestion logs recorded **381,109 rows** before preprocessing. The target is `Response`.

Input columns include gender, age, driving licence, region code, previous insurance status, vehicle age/damage, annual premium, policy sales channel, and vintage. The `id` field, if present, is an identifier and should be removed **during transformation** if the schema expects it during validation.

The dataset and SQLite database are local inputs and may be excluded from Git. Provide your own dataset at the expected location to retrain the model.

## Local setup (Windows + WSL Ubuntu)

Run these commands in your **WSL Ubuntu terminal**, from the repository root (the directory containing `app.py` and `requirements.txt`):

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If test dependencies are not included in your requirements:

```bash
python -m pip install pytest httpx
```

Confirm that the notebook-created database exists and has the expected table:

```bash
ls -lh notebook/vehicle.db
python - <<'PY'
import sqlite3
with sqlite3.connect('notebook/vehicle.db') as conn:
    print(conn.execute('SELECT COUNT(*) FROM vehicle_data').fetchone())
PY
```

If you have only `notebook/data.csv`, create the database before training:

```python
import pandas as pd
import sqlite3

frame = pd.read_csv('notebook/data.csv')
with sqlite3.connect('notebook/vehicle.db') as connection:
    frame.to_sql('vehicle_data', connection, if_exists='replace', index=False)
```

The constants/configuration should point to `notebook/vehicle.db` and the `vehicle_data` table. Paths are relative to the project root; run commands from there.

## Train and run locally

To execute the project training pipeline (including model evaluation and local promotion, if enabled):

```bash
python demo.py
```

A completed run saves timestamped files under `artifact/`, including the ingestion CSVs, a validation report, transformed `.npy` arrays, a preprocessing object, and the trained model. An accepted model is copied to `model_registry/model.pkl` when the model-pusher stage runs.

Start the web application:

```bash
python app.py
```

Open **http://localhost:5000/** for the prediction form. The application also exposes `GET /train` to trigger the training pipeline and **http://localhost:5000/docs** for API documentation. The training route currently triggers substantial work; it should be authenticated and changed to a protected `POST` endpoint before public deployment.

## Run with Docker

Docker Desktop must be running with **WSL Integration enabled for Ubuntu** if using Docker from a WSL terminal. Check:

```bash
docker info
docker run --rm hello-world
```

The CI/CD workflow publishes the image (verify package access in GitHub):

```bash
docker pull ghcr.io/lakshya-cyber/vehicle-mlops:latest
```

From the repository root, ensure the locally registered model exists, then run:

```bash
ls -lh model_registry/model.pkl

docker run -d \
  --name vehicle-mlops \
  -p 5000:5000 \
  -v "$(pwd)/model_registry:/app/model_registry" \
  -v "$(pwd)/notebook:/app/notebook" \
  ghcr.io/lakshya-cyber/vehicle-mlops:latest
```

The mounts are important:

- `model_registry/` provides the saved model for prediction and lets an accepted retrained model be promoted locally.
- `notebook/` provides `vehicle.db` to the training pipeline. The database is generally excluded from the Docker image by `.dockerignore`.

If you only want to **serve predictions**, the model mount can be read-only (`:/app/model_registry:ro`), and you may omit the notebook mount as long as you do not call `/train`.

Check the application:

```bash
docker ps
docker logs -f vehicle-mlops
```

Visit **http://localhost:5000/**. Stop any other process using port `5000` before starting the container. To recreate it after changing mounts:

```bash
docker stop vehicle-mlops
docker rm vehicle-mlops
```

> **Known Docker training issue addressed:** The image can serve predictions while `/train` fails with `sqlite3.OperationalError: unable to open database file` if `notebook/vehicle.db` is not mounted at `/app/notebook/vehicle.db` or if the configured path does not match. The mount above addresses the missing-file case; verify file permissions if errors remain. Training artifacts written only inside a container are not persisted after container removal unless their directories are mounted too.

## Tests and CI/CD

Run local application tests:

```bash
python -m pytest tests/ -v
```

The GitHub Actions workflow is defined in `.github/workflows/cicd.yml`. It performs:

1. **Continuous integration:** checkout, install Python dependencies, check syntax, and run tests.
2. **Docker validation:** build the container image.
3. **Continuous delivery:** after successful checks on `main`, build and publish the image to GHCR.

The GitHub Actions workflow has passed, and the published container has been run successfully for homepage rendering and prediction. This workflow **publishes an image**; it does not automatically deploy a live cloud service or retrain the model on each push.

## Configuration and generated outputs

- `src/constants/`: shared filenames, database path/table, training hyperparameters, metric threshold, and application host/port.
- `src/entity/config_entity.py`: configuration objects for individual pipeline stages.
- `src/entity/artifact_entity.py`: structured outputs handed from one stage to the next.
- `config/schema.yaml`: expected columns and feature groups for validation/transformation.
- `artifact/<timestamp>/`: outputs for a given run.
- `model_registry/model.pkl`: model currently promoted for local inference.

A new model is compared with the locally registered model using F1 score. The configured improvement threshold is `0.02`; the project can accept the first model when no existing registered model is present.

## Limitations and next steps

- **Cloud storage:** an AWS S3 workflow was adapted to a local registry. GCP Cloud Storage was investigated but not used because the cloud billing account was unavailable.
- **Production preprocessing:** all custom feature engineering (including gender/category encoding and dummy-column alignment) should eventually be incorporated into the saved preprocessing pipeline, so raw inference requests and training have identical transformations.
- **Operational readiness:** secure the training endpoint, add comprehensive tests for training and predictions, and persist training artifacts if training inside Docker.
- **Future MLOps additions:** MLflow experiment tracking, model/version metadata, automated scheduled retraining, monitoring, and cloud deployment.

## Interview summary

> Built a modular vehicle-insurance response prediction pipeline using SQLite, pandas, scikit-learn, and FastAPI. Implemented data ingestion, schema validation, feature transformation, imbalanced-class handling, Random Forest training, model evaluation against a locally registered model, and containerized inference. Added GitHub Actions to test, build, and publish the application image to GitHub Container Registry.

# DeepSafe User Manual

DeepSafe is a multimodal deepfake detection platform for images and videos. It provides a React web dashboard, a FastAPI gateway, isolated Docker model services, and ensemble prediction methods that combine model outputs into a single verdict.

This manual is for users and operators who want to run DeepSafe locally, submit media for analysis, inspect results, and manage the deployed detection models.

## 1. System Overview

DeepSafe runs as a set of Docker services:

| Component | Default URL | Purpose |
| --- | --- | --- |
| Web dashboard | `http://localhost:8888` | Browser UI for uploading media and viewing results |
| API gateway | `http://localhost:8001` | Main FastAPI service for detection, auth, health, and history |
| API docs | `http://localhost:8001/docs` | Swagger documentation for all API routes |
| NPR image model | `http://localhost:5001` | Image deepfake detector |
| UniversalFakeDetect image model | `http://localhost:5004` | Image deepfake detector |
| Cross-Efficient ViT video model | `http://localhost:7001` | Video deepfake detector |

The gateway reads active models from `config/deepsafe_config.json`, sends each request to the configured model services, and combines the responses using one of three ensemble methods:

| Method | Description |
| --- | --- |
| `voting` | Uses the majority class predicted by active models |
| `average` | Averages model fake-probability scores |
| `stacking` | Uses a trained meta-learner from `api/meta_model_artifacts/` when available |

If `stacking` is requested but the required artifacts are unavailable for a media type, the API falls back to `voting`.

## 2. Requirements

Install these before running DeepSafe:

- Docker and Docker Compose
- GNU Make
- Python 3.9 or newer for local utility scripts
- Node.js only if you plan to run the frontend outside Docker

The normal workflow uses Docker, so you do not need to install every model dependency directly on your host machine.

## 3. Quick Start

From the repository root:

```bash
make start
```

This builds and starts the API, frontend, and configured model services.

Open the dashboard:

```text
http://localhost:8888
```

Open the API documentation:

```text
http://localhost:8001/docs
```

Stop the system:

```bash
make stop
```

## 4. Common Commands

| Command | Purpose |
| --- | --- |
| `make help` | Show available Makefile commands |
| `make install` | Build all Docker images |
| `make start` | Start all services in detached mode |
| `make stop` | Stop all services |
| `make health` | Check model service health from the host |
| `make test` | Run system tests inside the API container |
| `make lint` | Run Python lint checks for API files |
| `make clean` | Stop services, remove volumes, and clear caches |
| `make add-model NAME=my_model MEDIA_TYPE=image PORT=5008` | Scaffold and register a model |
| `make retrain MEDIA_TYPE=image` | Generate meta-features and retrain the ensemble |
| `make eval MEDIA_TYPE=image` | Retrain from existing meta-feature CSV data |

## 5. Using the Web Dashboard

1. Start DeepSafe with `make start`.
2. Go to `http://localhost:8888`.
3. Upload a supported image or video file.
4. Choose the ensemble method if the UI exposes that control.
5. Submit the analysis.
6. Review the verdict, fake probability, model votes, model-level results, and response time.

The dashboard sends uploads to the API gateway through `POST /detect`. The API infers the media type from the file content type or file extension.

## 6. Supported Media

The active configuration supports:

| Media type | Supported extensions | Active models |
| --- | --- | --- |
| Image | `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`, `.tiff`, `.tif` | `npr_deepfakedetection`, `universalfakedetect` |
| Video | `.mp4`, `.avi`, `.mov`, `.mkv` | `cross_efficient_vit` |

The API code also contains payload mappings for audio, but the current default config does not register an active audio media type.

## 7. API Usage

Use `http://localhost:8001` as the base URL when calling the API from your host machine.

### Health Check

```bash
curl http://localhost:8001/health
```

The response includes overall API status, model health by media type, stacking availability, request ID, and processing mode.

### JSON Prediction

Use `POST /predict` when you already have base64-encoded media.

Image example:

```bash
IMAGE_B64="$(base64 -i test_samples/sample_image.jpg)"

curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d "{
    \"media_type\": \"image\",
    \"image_data\": \"$IMAGE_B64\",
    \"threshold\": 0.5,
    \"ensemble_method\": \"voting\"
  }"
```

Video example:

```bash
VIDEO_B64="$(base64 -i test_samples/sample_video.mp4)"

curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d "{
    \"media_type\": \"video\",
    \"video_data\": \"$VIDEO_B64\",
    \"threshold\": 0.5,
    \"ensemble_method\": \"voting\"
  }"
```

To run only selected models, include a `models` array:

```json
{
  "media_type": "image",
  "image_data": "<base64>",
  "models": ["npr_deepfakedetection"],
  "threshold": 0.5,
  "ensemble_method": "average"
}
```

### File Upload Detection

Use `POST /detect` for multipart file uploads.

```bash
curl -X POST http://localhost:8001/detect \
  -F "file=@test_samples/sample_image.jpg" \
  -F "threshold=0.5" \
  -F "ensemble_method=voting"
```

To select specific models with `/detect`, pass a comma-separated list:

```bash
curl -X POST http://localhost:8001/detect \
  -F "file=@test_samples/sample_image.jpg" \
  -F "models=npr_deepfakedetection,universalfakedetect" \
  -F "ensemble_method=average"
```

### Response Fields

Typical `POST /predict` responses include:

| Field | Meaning |
| --- | --- |
| `request_id` | Unique ID for tracing logs and history |
| `media_type_processed` | Media type handled by the API |
| `verdict` | Final `real`, `fake`, or `undetermined` verdict |
| `confidence_in_verdict` | Confidence in the returned verdict |
| `ensemble_score_is_fake` | Final probability-like fake score |
| `base_model_fake_votes` | Number of valid model results classified as fake |
| `base_model_real_votes` | Number of valid model results classified as real |
| `ensemble_method_requested` | Method requested by the client |
| `ensemble_method_used` | Method actually used after fallbacks |
| `model_results` | Per-model probabilities, classes, timings, or errors |
| `total_inference_time_seconds` | End-to-end processing time |

## 8. Accounts and Analysis History

DeepSafe includes simple token-based auth for protected history endpoints. User accounts are stored in memory by the API process, so registered users are reset when the API restarts.

Register a user:

```bash
curl -X POST http://localhost:8001/register \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=demo&password=demo123"
```

Log in:

```bash
curl -X POST http://localhost:8001/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=demo&password=demo123"
```

Use the returned bearer token:

```bash
curl http://localhost:8001/history \
  -H "Authorization: Bearer <token>"
```

Fetch a specific record:

```bash
curl http://localhost:8001/history/<request_id> \
  -H "Authorization: Bearer <token>"
```

History is stored in SQLite by default through `api/database.py`.

## 9. Configuration

The main runtime config is `config/deepsafe_config.json`.

Important settings:

| Key | Purpose |
| --- | --- |
| `api_url` | API base URL used by local utilities |
| `media_types` | Registered media types, model endpoints, health endpoints, and supported extensions |
| `default_threshold` | Default classification threshold |
| `default_ensemble_method` | Default ensemble method |
| `default_api_timeout_seconds` | Gateway timeout for model calls |
| `default_model_timeout_seconds_video` | Longer timeout for video models |
| `default_max_workers_batch` | Batch worker default for utility scripts |

Inside Docker, model endpoints use service names such as `http://npr_deepfakedetection:5001/predict`. Host-side utilities convert those service names to localhost URLs when needed.

## 10. Checking Service Health

Run:

```bash
make health
```

The health script checks each configured model service on localhost and prints the model name, media type, port, status, and whether the model is currently loaded.

You can also check services directly:

```bash
curl http://localhost:8001/health
curl http://localhost:5001/health
curl http://localhost:5004/health
curl http://localhost:7001/health
```

Model services use lazy loading by default. A healthy service may report `model_loaded: false` until the first prediction request loads the model.

## 11. Adding a Model

Use the scaffold command:

```bash
make add-model NAME=my_detector MEDIA_TYPE=image PORT=5008
```

Then implement the detector logic in:

```text
models/image/my_detector/detector.py
```

A model directory normally contains:

```text
models/<media_type>/<model_name>/
|-- model.yaml
|-- detector.py
|-- Dockerfile
`-- requirements.txt
```

Each model service must expose:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Report service status |
| `POST /predict` | Run detection on base64 media |
| `POST /unload` | Unload the model from memory |

After adding a model:

```bash
make start
make health
make retrain MEDIA_TYPE=image
```

For full model integration details, see `docs/adding-a-model.md`.

## 12. Retraining the Ensemble

Retrain the stacking meta-learner after adding, removing, or changing models.

Full image retraining:

```bash
make retrain MEDIA_TYPE=image
```

Retrain with a custom dataset directory:

```bash
make retrain MEDIA_TYPE=image DATASET_DIR=./my_dataset/images
```

Retrain with Optuna hyperparameter search:

```bash
make retrain MEDIA_TYPE=image OPTIMIZER=optuna TRIALS=100
```

Retrain from an existing meta-feature CSV:

```bash
make eval MEDIA_TYPE=image META_CSV=meta_learning_data/meta_features_dataset.csv
```

Generated and selected artifacts are expected under `api/meta_model_artifacts/<media_type>/` for runtime use.

## 13. Running Tests

Run the system test suite:

```bash
make test
```

Run API unit and integration tests directly from the host if dependencies are installed:

```bash
pytest
```

Run SDK tests:

```bash
pytest sdk/tests
```

Run lint checks:

```bash
make lint
```

## 14. Troubleshooting

### Dashboard does not load

Check that the frontend container is running:

```bash
docker compose ps
```

Restart services:

```bash
make stop
make start
```

### API health is degraded

Run:

```bash
make health
docker compose logs api
```

Common causes are model containers still starting, missing model weights, unavailable stacking artifacts, or a model service timing out.

### First request is slow

Most model services use lazy loading. The first request may download or load weights and take longer than later requests.

### `stacking` falls back to `voting`

Confirm the following files exist for the media type:

```text
api/meta_model_artifacts/<media_type>/deepsafe_meta_learner.joblib
api/meta_model_artifacts/<media_type>/deepsafe_meta_scaler.joblib
api/meta_model_artifacts/<media_type>/deepsafe_meta_imputer.joblib
api/meta_model_artifacts/<media_type>/deepsafe_meta_feature_columns.json
```

Then restart the API:

```bash
docker compose restart api
```

### Upload is rejected

Check that the file extension and MIME type are supported, the file is not empty, and image dimensions are at least 32x32 pixels.

### A model is unreachable

Check its container and logs:

```bash
docker compose ps
docker compose logs npr_deepfakedetection
docker compose logs universalfakedetect
docker compose logs cross_efficient_vit
```

Rebuild if needed:

```bash
docker compose build <service_name>
docker compose up -d <service_name>
```

## 15. Safety Notes

Deepfake detection scores are model outputs, not proof of authenticity. Treat results as decision support and review high-impact cases with additional evidence, provenance checks, and human judgment.

For production deployments, change the default `SECRET_KEY`, use persistent user storage, restrict CORS, add authentication to detection endpoints if required, and monitor model/service logs.

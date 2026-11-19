# AI255TBD-Battery-Prediction-Team26

**AI255TBD Module 1 – Doorbell Battery Life Prediction using Neural Networks (Team 26)**

## 1. Project Overview

This project develops a machine learning-based system to predict the remaining battery life of a smart doorbell using sensor readings.

The system uses a trained Neural Network model and deploys it using ONNX Runtime for efficient CPU-based inference.

It also includes a FastAPI prediction service, a Streamlit monitoring dashboard, offline inference testing, benchmarking, and Docker containerization.

## 2. Project Objectives

- Predict the remaining battery life of a smart doorbell.
- Analyze battery and environmental sensor readings.
- Train and evaluate machine learning models.
- Convert the trained Neural Network into ONNX format.
- Perform CPU-based inference using ONNX Runtime.
- Measure inference latency, throughput, and model size.
- Build an interactive monitoring dashboard.
- Package the prediction API using Docker.

## 3. Dataset Description

The project uses the Team 26 training dataset.

**Dataset file:** `data/team_26_train.csv`

The dataset contains 2,000 records and five input features.

| Feature | Description |
|---|---|
| battery_v | Battery voltage |
| temp_c | Temperature |
| rings_per_day | Daily doorbell usage |
| wifi_rssi_dbm | WiFi signal strength |
| age_months | Battery age in months |
| days_left | Target: remaining battery life |

## 4. Technologies Used

- Python
- NumPy and Pandas
- Scikit-learn
- Neural Networks (MLPRegressor)
- ONNX
- ONNX Runtime
- FastAPI
- Streamlit
- Matplotlib
- Docker
- GitHub Codespaces

## 5. Machine Learning Models

The following models were evaluated:

- Dummy Regressor
- Linear Regression
- Neural Network
- ONNX FP32
- ONNX INT8

### Neural Network Architecture

- Input features: 5
- Hidden layer 1: 32 neurons
- Hidden layer 2: 16 neurons
- Output: 1 predicted battery-life value
- Activation: ReLU

### Model Evaluation Results

| Model | MAE (days) | RMSE (days) | R² |
|---|---:|---:|---:|
| Dummy Regressor | 51.297 | — | — |
| Linear Regression | 22.629 | — | — |
| Neural Network | 6.669 | 11.005 | 0.968864 |
| ONNX FP32 | 6.669 | 11.005 | 0.968864 |
| ONNX INT8 | 6.716 | 10.995 | 0.968919 |

The ONNX FP32 model was selected for inference because it maintained the original model's prediction performance and performed better than INT8 in the measured benchmark.

## 6. ONNX Conversion and Optimization

The trained Neural Network was converted into ONNX format.

| Model | File size |
|---|---:|
| Original PKL model | 40.74 KB |
| ONNX FP32 | 4.00 KB |
| ONNX INT8 | 4.18 KB |

The ONNX FP32 model achieved approximately 90.18% file-size reduction compared with the original PKL file.

## 7. Offline Edge AI Benchmark

A benchmark experiment was conducted using 1,000 simulated sensor samples.

| Metric | Result |
|---|---:|
| Test samples | 1,000 |
| Average latency | 0.02999 ms |
| Median latency | 0.02732 ms |
| P95 latency | 0.04102 ms |
| Throughput | 33,342.23 predictions/sec |
| Mean prediction difference | 0.000004 days |

Inference was performed using ONNX Runtime with CPUExecutionProvider, without an external prediction API.

**Note:** These measurements were obtained on the cloud-hosted GitHub Codespaces CPU environment and are not physical doorbell hardware measurements.

## 8. Application Components

### Streamlit Dashboard

The dashboard provides:

- Sensor input controls.
- Battery-life prediction.
- Battery condition indicators.
- Prediction history.
- Model performance comparison.
- CPU and memory monitoring.

### FastAPI

The REST API provides:

- `GET /` – API status.
- `POST /predict` – Battery-life prediction.

Interactive API documentation is available at `/docs`.

### Offline Inference

The standalone inference program uses ONNX Runtime directly to generate predictions without calling an external prediction API.

### Docker Deployment

The application is containerized using Docker.

**Docker image:** `smart-doorbell-edge-ai:1.0`

**Container:** `smart-doorbell-container`

**API container port:** 8000

The containerized API was tested through Swagger, including invalid sensor input validation.

## 9. Docker Resource Monitoring

Observed container resource usage:

| Metric | Result |
|---|---:|
| CPU usage | 0.08% |
| Memory usage | 206.4 MiB |
| Memory utilization | 2.60% |
| Running processes | 8 |

These are observed idle-time measurements in Codespaces.

## 10. Project Visualizations

Generated performance graphs are available in the `results/` directory:

- `model_accuracy_comparison.png`
- `onnx_quantization_comparison.png`
- `inference_latency.png`
- `edge_ai_performance_summary.png`

Experimental data files:

- `offline_benchmark_results.csv`
- `model_comparison.csv`
- `performance_summary.json`

## 11. Project Execution

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI255TBD-Battery-Prediction-Team26
```

### Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit dashboard

```bash
streamlit run dashboard.py
```

### Run the FastAPI application

```bash
uvicorn app:app --reload
```

### Run standalone offline inference

```bash
python offline_inference.py
```

### Run the offline benchmark

```bash
python offline_benchmark.py
```

### Build the Docker image

```bash
docker build -t smart-doorbell-edge-ai:1.0 .
```

### Run the Docker container

```bash
docker run -d --name smart-doorbell-container -p 8001:8000 smart-doorbell-edge-ai:1.0
```

Access the API documentation at:

`http://localhost:8001/docs`

## 12. Conclusion

The project demonstrates a complete software-based machine learning workflow for smart doorbell battery-life prediction.

It combines Neural Network training, ONNX model conversion, CPU-based inference, performance benchmarking, interactive visualization, API development, and Docker containerization.

The experiments demonstrate that the compact ONNX FP32 model can produce predictions with performance close to the original trained model.

**Team 26 – AI255TBD Module 1**
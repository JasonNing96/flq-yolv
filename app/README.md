# FLQ-Fed Framework

This directory contains the distributed implementation of the FLQ-YOLOv8 framework.

## Core Components

*   **`client.py`**: The client-side logic.
    *   Implements `ManualClientTrainer` based on `flq_yolov_v8.py`.
    *   Features **FreezeBN** logic to handle non-IID data statistics.
    *   Uses `FLQCompressor` for gradient compression and quantization.
    *   Communicates with the server via HTTP.

*   **`server.py`**: The server-side logic (FastAPI).
    *   Implements federated aggregation (FedAvg / FLQ-Fed).
    *   Uses **Global LR = 1.0** (matching `flq_yolov_v8.py`).
    *   Handles quantization decoding.
    *   Saves checkpoints and global model metrics.

*   **`model_utils.py`**: Shared utility functions.
    *   `FLQCompressor`: Handles quantization (1-bit, 8-bit) and error feedback.
    *   Model vectorization and de-vectorization tools.

*   **`runner.py`**: A CLI tool to easily start the server and clients.
    *   Usage: `python -m app.runner train` (Starts everything)

## Usage

### 1. Automated Training (Recommended)

Run the full training loop (server + clients) on a single machine:

```bash
python -m app.runner train
```

### 2. Manual Execution

**Step 1: Start Server**

```bash
python -m app.runner server
# or
python -m app.server
```

**Step 2: Start Clients**

```bash
# Terminal 1
python -m app.client 1

# Terminal 2
python -m app.client 2
```

## Configuration

Configuration is loaded from `configs/flq_config.yaml` (or similar). Key parameters:

*   `aggregation_mode`: `flq-fed` (compressed) or `fedavg`.
*   `quant_bits`: 8 (or 1, 4, 32).
*   `clients_per_round`: Number of clients to wait for before aggregating.
*   `model_name`: Path to the base YOLO model (e.g., `yolov8s.pt`).


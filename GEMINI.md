# Project Overview: FLQ-YOLOv - Federated Learning for Oil Spill Detection

This project implements Federated Learning with Quantization (FLQ) for object detection, specifically focusing on oil spill detection using YOLO (You Only Look Once) models. It also includes a general PyTorch-based FLQ framework for classification tasks (e.g., MNIST/FashionMNIST).

## Key Features:

*   **FLQ-YOLOv:** A single-machine simulation of federated learning for YOLOv8/v11 models, incorporating downlink model quantization and uplink gradient quantization. It tracks metrics like mAP, loss, and communication bits.
*   **FLQ-Fed (PyTorch):** A PyTorch implementation of federated learning with various quantization strategies (bbit, binary, LAQ8, FedAvg) and client selection mechanisms. It supports both IID and non-IID data partitioning.
*   **Distributed Training Orchestration:** The `app/runner.py` script provides a unified entry point to orchestrate distributed federated learning by launching separate server and client processes.
*   **Data Management:** Includes a script (`data/oil_dataset_split.py`) to prepare and partition the oil spill detection dataset for federated training.
*   **Configuration:** Uses YAML files (`flq_config.yaml`, `data/oil.yaml`) for flexible configuration of training parameters, quantization settings, model paths, and server/client details.

## Technologies Used:

*   **Python:** Primary programming language.
*   **PyTorch:** Deep learning framework for model training and FLQ implementation.
*   **Ultralytics YOLO:** For object detection models (YOLOv8, YOLOv11).
*   **NumPy, Pandas, Matplotlib:** For data manipulation, analysis, and visualization of training metrics.
*   **YAML:** For configuration files.
*   **`argparse`:** For command-line argument parsing.
*   **`subprocess`:** For managing parallel processes (server and clients).

## Project Structure:

*   `flq_yolov.py`: Main script for single-machine FLQ-YOLOv training simulation.
*   `flq_fed_v4.py`: PyTorch implementation of general FLQ for classification tasks.
*   `flq_config.yaml`: Global configuration for the federated learning setup.
*   `yolo.sh`: Example shell script to run `flq_yolov.py`.
*   `app/`: Contains the distributed federated learning framework components.
    *   `app/config.py`: Configuration loader.
    *   `app/runner.py`: Unified entry point for distributed training (server + clients).
    *   `app/server.py`: (Not fully reviewed, but implied by `runner.py`) Server-side logic for federated aggregation.
    *   `app/client.py`: (Not fully reviewed, but implied by `runner.py`) Client-side logic for local training and communication.
*   `data/`: Dataset related files.
    *   `data/oil_dataset_split.py`: Script to split the oil detection dataset.
    *   `data/oil.yaml`: Sample YOLO dataset configuration.
    *   `data/oil_detection_dataset/`: Contains the oil spill detection dataset, partitioned for clients.
    *   `data/MNIST/`, `data/FashionMNIST/`: Datasets for `flq_fed_v4.py`.
*   `models/`: Stores pre-trained YOLO models.
*   `outputs/`: Directory for training outputs (logs, checkpoints).
*   `results/`: Directory for training results (metrics CSVs, plots).

## Building and Running:

### 1. Dataset Preparation:

To prepare the oil spill detection dataset for federated training, run the `oil_dataset_split.py` script:

```bash
python data/oil_dataset_split.py
```
This will create client-specific dataset directories and `oil.yaml` files under `data/oil_detection_dataset/`.

### 2. Running FLQ-YOLOv (Single-Machine Simulation):

Use the provided `yolo.sh` script or run `flq_yolov.py` directly:

```bash
# Using the shell script
./yolo.sh

# Or directly with python (example arguments)
python flq_yolov.py \
  --data ./data/oil_detection_dataset/client1/oil.yaml \
  --model ./models/yolov8n.pt \
  --out-dir ./results/yolov8n \
  --rounds 5 \
  --local-epochs 1 \
  --quant-bits 1 \
  --down-bits 0
```

### 3. Running FLQ-Fed (Distributed Mode):

The `app/runner.py` script is the main entry point for distributed federated learning.

*   **Full Training (Recommended - automatically starts server and clients):**

    ```bash
    python -m app.runner train
    ```
    You can specify a custom configuration file:
    ```bash
    python -m app.runner train --config flq_config.yaml
    ```

*   **Starting Server Separately:**

    ```bash
    python -m app.runner server
    ```

*   **Starting Client Separately:**

    ```bash
    python -m app.runner client --id 1 --server http://localhost:8087
    ```

### 4. Running FLQ-Fed (PyTorch - Classification Simulation):

To run the FLQ simulation for classification tasks (e.g., MNIST), execute `flq_fed_v4.py`:

```bash
python flq_fed_v4.py --dataset mnist --mode bbit --iters 800 --M 10 --b 8 --b_down 8
```

### 5. Model Weights:

Pre-trained YOLO models (`yolov8n.pt`, `yolov8s.pt`, `yolo11n.pt`) are expected in the `models/` directory. If they are missing, you might need to download them from Ultralytics.

## Development Conventions:

*   **Language:** Python.
*   **Configuration:** YAML files are used for project configuration.
*   **Logging:** Standard Python `logging` module is used for output.
*   **Results:** Training metrics are saved to CSV/Excel files, and plots are generated for visualization.
*   **Code Style:** Follows general Python best practices; type hints are used in some modules.

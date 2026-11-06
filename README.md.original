
# Intelligent Camera-Based IV Fluid Monitoring and Alert System

A computer vision and deep learning research prototype for classifying visible IV saline bottle fill levels and demonstrating low-level alerts.

## Features

- Four-class saline bottle classification: 100%, 80%, 50%, and empty.
- HOG feature extraction with a Linear SVM baseline.
- MobileNetV2 transfer-learning experiments.
- Streamlit dashboard for image uploads and predictions.
- Dataset inspection, duplicate analysis, split auditing, and error visualization.

## Technology Stack

Python, OpenCV, NumPy, pandas, scikit-learn, TensorFlow/Keras, Streamlit, Pillow, and Joblib.

## Project Structure

```text
IV-Drip-Detection/
├── src/
│   ├── dashboard.py
│   ├── predict.py
│   ├── monitor_iv.py
│   ├── train_model.py
│   ├── train_mobilenet.py
│   └── ...
├── tests/
├── docs/
├── datasets/       # Local only; excluded from Git
├── models/         # Local model artifacts; excluded from Git
├── results/        # Generated outputs; excluded from Git
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Use Python 3.11 or 3.12 for a fresh environment if TensorFlow compatibility with your platform is uncertain. Existing environments may use another supported version.

Create and activate a virtual environment on Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

The experiments use a public saline bottle image dataset with four labelled categories.

- [Mendeley Data — Dataset 1](https://data.mendeley.com/datasets/9mcj3rvvxb/1)
- [Mendeley Data — Dataset 2](https://data.mendeley.com/datasets/n8k2zfr6xm/1)

Download and extract the datasets locally into the expected folders. Preserve the original class labels and follow the dataset licence.

Dataset images are not included in this repository.

## Running the Dashboard

The current dashboard uses the HOG + Linear SVM model.

1. Ensure `models/hog_linear_svc.joblib` exists locally.
2. Install the dependencies.
3. From the repository root, run:

```cmd
streamlit run src/dashboard.py
```

If the model file is missing, train it using the project's training script and verify the output path before starting the dashboard.

## Training

Run the relevant scripts from the repository root, after preparing the dataset:

```cmd
python src/train_model.py
python src/train_mobilenet.py
```

Check each script's expected input paths and output filenames before running it. Training can take time on CPU.

## Experimental Results and Limitations

An earlier random image split reported 95.42% accuracy for MobileNetV2. A chronological evaluation reported 14.90%.

These results are not directly interchangeable. The large difference raises concerns about generalization and possible similarity or capture-condition effects. The current dataset does not establish performance on independently captured bottles.

Future evaluation should hold out independent bottles and capture sessions, preserve class balance where feasible, and report per-class precision, recall, F1-score, and confusion matrices.

## Safety Disclaimer

This is an academic research prototype, not a clinically validated medical device. Predicted categories are not direct measurements of fluid volume. Do not use its outputs to guide patient care, control infusion flow, or replace direct observation by trained healthcare professionals.

# MLOps Assignment 3 - End-to-End MLOps Pipeline

## Overview
This project implements a complete MLOps pipeline with:
- Linear Regression model training on California Housing dataset
- Docker containerization
- CI/CD pipeline with GitHub Actions
- Manual quantization optimization

## Repository Structure
```
├── README.md
├── .gitignore
├── Dockerfile
├── requirements.txt
├── src/
│   ├── train.py
│   ├── predict.py
│   └── quantize.py
└── .github/
    └── workflows/
        └── ci.yml
```

## Branches
- `main`: Initial setup
- `dev`: Model development
- `docker_ci`: Docker and CI/CD implementation
- `quantization`: Model quantization and optimization

## Setup Instructions
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python src/train.py`
4. Run prediction: `python src/predict.py`
5. Run quantization: `python src/quantize.py`

## Docker Usage
```bash
docker build -t mlops-assignment .
docker run mlops-assignment python src/predict.py
```

## Results
| Metric | Original Sklearn Model | Quantized Model |
|--------|------------------------|-----------------|
| R² Score | 0.5758 | -0.1799 |
| Model Size | 0.33 KB | 0.37 KB |

## Analysis
- **Model Performance**: The original scikit-learn model achieves an R² score of 0.5758 on the California Housing dataset
- **Quantization Impact**: 8-bit quantization introduces significant quantization error, reducing R² to -0.1799
- **Size Trade-off**: The quantized model shows minimal size reduction due to metadata overhead in this small model
- **Quantization Error**: Maximum coefficient difference after quantization: 0.00070421

## Branch Structure
- `main`: Initial setup with README and .gitignore
- `dev`: Model development with train.py
- `docker_ci`: Docker containerization and CI/CD pipeline
- `quantization`: Manual quantization implementation
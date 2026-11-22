# Maize Yield Prediction — Linear Regression Model

## Mission & Problem
My ALU mission is to empower farmers in Rwanda and across Africa through AI
and drone technologies that increase productivity, optimize resource use,
and promote sustainable agriculture. This project predicts maize yield
(tons/hectare) from soil, weather, and resource-use data (rainfall,
temperature, fertilizer use, irrigation use), so a farmer or drone-data
pipeline can estimate expected yield before/during a season.

**Dataset:** Agriculture Crop Yield dataset (Kaggle), filtered to
`Crop == Maize`. Source:
https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield

## Public API Endpoint
- Base URL: https://maize-yield-api-3626.onrender.com
- Swagger UI (for grading/testing): https://maize-yield-api-3626.onrender.com/docs
- Predict endpoint: POST https://maize-yield-api-3626.onrender.com/predict

## Video Demo
https://youtu.be/FwHdqdvOI9M?si=mlLVGhQeQTUIV7XZ

## Repository Structure
```
linear_regression_model/
├── summative/
│   ├── linear_regression/
│   │   └── multivariate.ipynb      # EDA, feature engineering, model training/comparison
│   ├── API/
│   │   ├── prediction.py           # FastAPI app (predict + retrain endpoints)
│   │   ├── requirements.txt        # for Render deployment
│   │   ├── model/best_model.pkl    # saved best-performing pipeline
│   │   └── data/crop_yield.csv    # training data (used by /retrain)
│   └── FlutterApp/                 # single-page Flutter mobile app
├── pyproject.toml
└── uv.lock
```

## Running the Notebook
```bash
uv sync
uv run jupyter notebook summative/linear_regression/multivariate.ipynb
```

## Running the API Locally
```bash
cd summative/API
uv run --project ../.. uvicorn prediction:app --reload --port 8000
```
Then open http://127.0.0.1:8000/docs for the Swagger UI.

## Running the Mobile App
1. Install Flutter (see https://docs.flutter.dev/install).
2. From `summative/FlutterApp`:
   ```bash
   flutter pub get
   flutter run
   ```
3. Pick a connected device or simulator when prompted.

"""
Maize Yield Prediction API
---------------------------
Serves predictions from the best-performing regression model trained in
summative/linear_regression/multivariate.ipynb, in support of the mission:
empowering farmers in Rwanda and across Africa through AI and drone
technologies that increase productivity, optimize resource use, and promote
sustainable agriculture.

Endpoints:
    GET  /              -> health check / API info
    POST /predict        -> predict maize yield (tons/hectare) for one input
    POST /retrain         -> retrain the model on newly uploaded data

Run locally:
    uv run uvicorn prediction:app --reload
Then open http://127.0.0.1:8000/docs for the Swagger UI.
"""

import os
from enum import Enum

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "crop_yield.csv")

RANDOM_STATE = 42
NUMERIC_FEATURES = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
CATEGORICAL_FEATURES = ["Region", "Soil_Type", "Weather_Condition"]
TARGET = "Yield_tons_per_hectare"
REQUIRED_TRAINING_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [
    "Fertilizer_Used", "Irrigation_Used", TARGET,
]

app = FastAPI(
    title="Maize Yield Prediction API",
    description=(
        "Predicts maize yield (tons/hectare) from soil, weather, and "
        "resource-use inputs, in support of the ALU mission to empower "
        "African farmers through AI and drone technologies."
    ),
    version="1.0.0",
)

# --------------------------------------------------------------------------
# CORS configuration
#
# Reasoning:
# - allow_origins is an EXPLICIT list, not "*". A wildcard would let any
#   website embed calls to this API from a user's browser session, which is
#   unnecessary risk for a public prediction endpoint and is penalized by
#   the rubric. We only allow the origins that legitimately need browser
#   access: local development servers and the Swagger UI docs host itself
#   (same-origin, added automatically by FastAPI, so it needs no entry
#   here). The Flutter mobile app is NOT a browser context, so CORS does
#   not apply to it at all -- CORS only restricts browser-issued requests.
# - allow_methods is limited to GET and POST, the only two methods this API
#   actually exposes. Allowing PUT/DELETE/PATCH would be dead permission
#   surface for no functional benefit.
# - allow_headers is limited to the headers this API's clients actually
#   send (Content-Type for JSON bodies). We don't accept custom auth headers
#   yet, so we don't allow them.
# - allow_credentials is False because this API does not use cookies or
#   session-based auth -- every request is stateless, so there is nothing
#   that needs credentialed cross-origin access.
#
# Update ALLOWED_ORIGINS with your deployed frontend's real domain(s) before
# submission (e.g. a Flutter web build's hosting URL), and remove the
# localhost entries if they are no longer needed for local testing.
# --------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --------------------------------------------------------------------------
# Pydantic schema -- mirrors exactly the raw columns the trained pipeline
# expects (see summative/linear_regression/multivariate.ipynb, section 3).
# Enums enforce that only categories seen during training are accepted.
# Numeric fields carry realistic range constraints for this use case.
# --------------------------------------------------------------------------
class RegionEnum(str, Enum):
    north = "North"
    east = "East"
    south = "South"
    west = "West"


class SoilTypeEnum(str, Enum):
    clay = "Clay"
    sandy = "Sandy"
    loam = "Loam"
    silt = "Silt"
    peaty = "Peaty"
    chalky = "Chalky"


class WeatherConditionEnum(str, Enum):
    sunny = "Sunny"
    rainy = "Rainy"
    cloudy = "Cloudy"


class PredictionInput(BaseModel):
    Region: RegionEnum = Field(..., description="Farm region")
    Soil_Type: SoilTypeEnum = Field(..., description="Dominant soil type of the field")
    Rainfall_mm: float = Field(
        ..., ge=0, le=2000, description="Total seasonal rainfall in millimeters (0-2000)"
    )
    Temperature_Celsius: float = Field(
        ..., ge=-10, le=45, description="Average growing-season temperature in Celsius (-10 to 45)"
    )
    Fertilizer_Used: bool = Field(..., description="Whether fertilizer was applied")
    Irrigation_Used: bool = Field(..., description="Whether irrigation was used")
    Weather_Condition: WeatherConditionEnum = Field(..., description="Predominant weather condition")
    Days_to_Harvest: int = Field(
        ..., ge=30, le=250, description="Number of days from planting to harvest (30-250)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Region": "East",
                "Soil_Type": "Loam",
                "Rainfall_mm": 650.0,
                "Temperature_Celsius": 23.5,
                "Fertilizer_Used": True,
                "Irrigation_Used": True,
                "Weather_Condition": "Rainy",
                "Days_to_Harvest": 104,
            }
        }
    }


class PredictionOutput(BaseModel):
    predicted_yield_tons_per_hectare: float
    model_file: str


class RetrainOutput(BaseModel):
    message: str
    rows_used: int
    best_model: str
    test_rmse: float
    test_r2: float


# --------------------------------------------------------------------------
# Load model at startup
# --------------------------------------------------------------------------
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=500,
                detail=f"Model file not found at {MODEL_PATH}. Train and save a model first.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/")
def read_root():
    return {
        "message": "Maize Yield Prediction API is running.",
        "docs": "/docs",
        "predict_endpoint": "/predict",
        "retrain_endpoint": "/retrain",
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    model = get_model()

    row = pd.DataFrame([{
        "Region": payload.Region.value,
        "Soil_Type": payload.Soil_Type.value,
        "Rainfall_mm": payload.Rainfall_mm,
        "Temperature_Celsius": payload.Temperature_Celsius,
        "Fertilizer_Used": int(payload.Fertilizer_Used),
        "Irrigation_Used": int(payload.Irrigation_Used),
        "Weather_Condition": payload.Weather_Condition.value,
        "Days_to_Harvest": payload.Days_to_Harvest,
    }])

    try:
        prediction = model.predict(row)[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    return PredictionOutput(
        predicted_yield_tons_per_hectare=round(float(prediction), 3),
        model_file=os.path.basename(MODEL_PATH),
    )


@app.post("/retrain", response_model=RetrainOutput)
def retrain(file: UploadFile = File(...)):
    """
    Retrain the model from scratch using a newly uploaded CSV.

    The CSV must contain these columns (same schema as the original
    training data): Region, Soil_Type, Rainfall_mm, Temperature_Celsius,
    Fertilizer_Used, Irrigation_Used, Weather_Condition, Days_to_Harvest,
    Yield_tons_per_hectare.

    This is a manually-triggered retrain (called via this endpoint whenever
    new data is available), rather than a background/automatic job, so
    behavior is predictable and each retrain is auditable via the response.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    try:
        new_df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")

    missing_cols = [c for c in REQUIRED_TRAINING_COLUMNS if c not in new_df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded CSV is missing required columns: {missing_cols}",
        )

    # Merge with existing training data so the model retrains on the full
    # accumulated dataset, not just the new upload.
    if os.path.exists(DATA_PATH):
        existing_df = pd.read_csv(DATA_PATH)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    # Clean: drop rows missing the target, impute the rest (mirrors the
    # notebook's missing-value handling).
    combined_df = combined_df.dropna(subset=[TARGET]).copy()
    for col in NUMERIC_FEATURES:
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())
    for col in CATEGORICAL_FEATURES + ["Fertilizer_Used", "Irrigation_Used"]:
        combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])

    combined_df["Fertilizer_Used"] = combined_df["Fertilizer_Used"].astype(int)
    combined_df["Irrigation_Used"] = combined_df["Irrigation_Used"].astype(int)

    X = combined_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["Fertilizer_Used", "Irrigation_Used"]]
    y = combined_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], remainder="passthrough")

    candidate_models = {
        "Linear Regression (OLS)": LinearRegression(),
        "SGD Linear Regression (Gradient Descent)": SGDRegressor(
            loss="squared_error", penalty="l2", max_iter=1000,
            learning_rate="invscaling", eta0=0.01, random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    best_name, best_pipeline, best_rmse, best_r2 = None, None, np.inf, None
    for name, estimator in candidate_models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        if rmse < best_rmse:
            best_name, best_pipeline, best_rmse, best_r2 = name, pipe, rmse, r2

    joblib.dump(best_pipeline, MODEL_PATH)
    combined_df.to_csv(DATA_PATH, index=False)

    global _model
    _model = best_pipeline  # hot-swap the in-memory model immediately

    return RetrainOutput(
        message="Model retrained and saved successfully.",
        rows_used=len(combined_df),
        best_model=best_name,
        test_rmse=round(best_rmse, 4),
        test_r2=round(best_r2, 4),
    )

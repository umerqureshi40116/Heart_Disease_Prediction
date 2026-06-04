"""
Heart Disease Prediction API - FastAPI
Backend for React Frontend Integration
Model: K-Nearest Neighbors (KNN) - scikit-learn 1.6.1
Python: 3.11
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import logging
from typing import List, Optional
import sys
from pathlib import Path

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# APP INITIALIZATION
# ============================================================================
app = FastAPI(
    title="Heart Disease Prediction API",
    description="KNN-based heart disease prediction service",
    version="1.0.0"
)

# ============================================================================
# CORS CONFIGURATION
# ============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODEL LOADING
# ============================================================================
try:
    # Check if model files exist
    model_path = Path("knn_heart_disease_model.pkl")
    scaler_path = Path("feature_scaler.pkl")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    # Load model and scaler
    knn_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    logger.info("✓ Model and scaler loaded successfully")
    logger.info(f"  Model type: {type(knn_model).__name__}")
    logger.info(f"  Model: {knn_model}")
    
except Exception as e:
    logger.error(f"✗ Failed to load model: {e}")
    sys.exit(1)

# ============================================================================
# CONSTANTS
# ============================================================================
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

FEATURE_RANGES = {
    'age': (29, 77),
    'sex': (0, 1),
    'cp': (0, 3),
    'trestbps': (94, 200),
    'chol': (126, 564),
    'fbs': (0, 1),
    'restecg': (0, 2),
    'thalach': (60, 202),
    'exang': (0, 1),
    'oldpeak': (0, 6.2),
    'slope': (0, 2),
    'ca': (0, 3),
    'thal': (0, 3)
}

FEATURE_DESCRIPTIONS = {
    'age': 'Age in years',
    'sex': 'Sex (0=Female, 1=Male)',
    'cp': 'Chest pain type (0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic)',
    'trestbps': 'Resting blood pressure in mmHg',
    'chol': 'Serum cholesterol in mg/dl',
    'fbs': 'Fasting blood sugar > 120 mg/dl (0=No, 1=Yes)',
    'restecg': 'Resting ECG results (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (0=No, 1=Yes)',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'Slope of ST segment (0=upsloping, 1=flat, 2=downsloping)',
    'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
    'thal': 'Thalassemia type (0=normal, 1=fixed defect, 2=reversible defect, 3=detected)'
}

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class PatientData(BaseModel):
    """Single patient data model"""
    age: int = Field(..., ge=18, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=Female, 1=Male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type")
    trestbps: int = Field(..., ge=80, le=220, description="Resting blood pressure")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results")
    thalach: int = Field(..., ge=50, le=210, description="Max heart rate")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="ST segment slope")
    ca: int = Field(..., ge=0, le=3, description="Number of vessels")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia type")
    
    def validate_binary(cls, v, info):
        if v not in (0, 1):
            raise ValueError('Must be 0 or 1')
        return v
    
    @field_validator('cp', 'restecg', 'slope')
    @classmethod
    def validate_range(cls, v, info):
        field = info.field_name  # replacement for old "field"

        if field == 'cp' and v not in (0, 1, 2, 3):
            raise ValueError('CP must be 0-3')

        if field == 'restecg' and v not in (0, 1, 2):
            raise ValueError('RESTECG must be 0-2')

        if field == 'slope' and v not in (0, 1, 2):
            raise ValueError('SLOPE must be 0-2')

        return v

    class Config:
        schema_extra = {
            "example": {
                "age": 43,
                "sex": 1,
                "cp": 0,
                "trestbps": 120,
                "chol": 198,
                "fbs": 0,
                "restecg": 0,
                "thalach": 149,
                "exang": 0,
                "oldpeak": 1.8,
                "slope": 1,
                "ca": 0,
                "thal": 3
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    records: List[PatientData] = Field(..., description="List of patient records")


class PredictionResponse(BaseModel):
    """Single prediction response"""
    success: bool
    prediction: int
    disease_present: bool
    prediction_label: str
    confidence: float
    probability_no_disease: float
    probability_disease: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    success: bool
    total_records: int
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: str
    features_count: int


class APIInfo(BaseModel):
    """API information"""
    service: str
    version: str
    model: str
    sklearn_version: str
    features: List[str]
    feature_descriptions: dict


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_risk_level(probability: float) -> str:
    """Determine risk level from probability"""
    if probability >= 0.8:
        return "Very High Risk"
    elif probability >= 0.6:
        return "High Risk"
    elif probability >= 0.4:
        return "Moderate Risk"
    else:
        return "Low Risk"


def make_prediction(features: np.ndarray) -> dict:
    """Make prediction using the model"""
    try:
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = knn_model.predict(scaled_features)[0]
        probabilities = knn_model.predict_proba(scaled_features)[0]
        
        prob_disease = float(probabilities[1])
        
        return {
            'prediction': int(prediction),
            'disease_present': bool(prediction),
            'prediction_label': 'Disease Present' if prediction == 1 else 'No Disease',
            'confidence': float(max(probabilities)),
            'probability_no_disease': float(probabilities[0]),
            'probability_disease': prob_disease,
            'risk_level': get_risk_level(prob_disease)
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint - API overview"""
    return {
        "service": "Heart Disease Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "GET /": "This endpoint",
            "GET /health": "Health check",
            "GET /info": "API information and features",
            "POST /predict": "Single prediction",
            "POST /batch-predict": "Batch predictions",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Test the model
        test_data = np.array([[43, 1, 0, 120, 198, 0, 0, 149, 0, 1.8, 1, 0, 3]])
        scaler.transform(test_data)
        knn_model.predict(test_data)
        
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_type="KNeighborsClassifier",
            features_count=len(FEATURE_NAMES)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Model health check failed")


@app.get("/info", response_model=APIInfo, tags=["Info"])
async def get_info():
    """Get API information"""
    import sklearn
    return APIInfo(
        service="Heart Disease Prediction API",
        version="1.0.0",
        model="K-Nearest Neighbors (KNN)",
        sklearn_version=sklearn.__version__,
        features=FEATURE_NAMES,
        feature_descriptions=FEATURE_DESCRIPTIONS
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientData):
    """
    Make a single prediction
    
    ### Input:
    - All 13 patient health metrics
    
    ### Output:
    - Prediction (0 or 1)
    - Disease probability
    - Confidence score
    - Risk level assessment
    """
    try:
        # Convert patient data to array
        features = np.array([[
            patient.age,
            patient.sex,
            patient.cp,
            patient.trestbps,
            patient.chol,
            patient.fbs,
            patient.restecg,
            patient.thalach,
            patient.exang,
            patient.oldpeak,
            patient.slope,
            patient.ca,
            patient.thal
        ]], dtype=float)
        
        # Make prediction
        result = make_prediction(features)
        
        return PredictionResponse(
            success=True,
            **result
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Make predictions for multiple patients
    
    ### Input:
    - List of patient records
    
    ### Output:
    - Predictions for all patients
    """
    try:
        results = []
        
        for patient in request.records:
            features = np.array([[
                patient.age,
                patient.sex,
                patient.cp,
                patient.trestbps,
                patient.chol,
                patient.fbs,
                patient.restecg,
                patient.thalach,
                patient.exang,
                patient.oldpeak,
                patient.slope,
                patient.ca,
                patient.thal
            ]], dtype=float)
            
            result = make_prediction(features)
            results.append(PredictionResponse(success=True, **result))
        
        return BatchPredictionResponse(
            success=True,
            total_records=len(results),
            results=results
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return {
        "success": False,
        "error": exc.detail,
        "status_code": exc.status_code
    }


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle value errors"""
    return {
        "success": False,
        "error": str(exc),
        "status_code": 422
    }


# ============================================================================
# STARTUP AND SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("="*60)
    logger.info("🏥 Heart Disease Prediction API - Starting")
    logger.info("="*60)
    logger.info(f"✓ Model loaded: {type(knn_model).__name__}")
    logger.info(f"✓ Features: {len(FEATURE_NAMES)}")
    logger.info(f"✓ API docs: http://localhost:8000/docs")
    logger.info("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    logger.info("🏥 Heart Disease Prediction API - Shutting down")


# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
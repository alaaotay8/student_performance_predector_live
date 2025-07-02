from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.preprocessing import LabelEncoder

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the application
app = FastAPI(
    title="Student Performance AI Predictor", 
    description="AI-powered student academic performance prediction using machine learning",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get absolute paths for deployment compatibility
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mount static files and templates with absolute paths
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load model with error handling for deployment
try:
    model_path = os.path.join(BASE_DIR, "models", "enhanced_student_perf_model.pkl")
    model_data = joblib.load(model_path)
    logger.info(f"✅ Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model_data = None

@app.get("/", response_class=HTMLResponse)
@app.head("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main index.html page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint for deployment"""
    return {
        "status": "healthy",
        "model_loaded": model_data is not None,
        "version": "1.0.0"
    }

@app.get("/predict")
async def predict(
    Age: int = Query(..., description="Student age (15 to 18 years)"),
    Gender: int = Query(..., description="Student gender (0: Male, 1: Female)"),
    Ethnicity: int = Query(..., description="Ethnicity (0: Caucasian, 1: African American, 2: Asian, 3: Other)"),
    ParentalEducation: int = Query(..., description="Parental education level (0: None, 1: High School, 2: Some College, 3: Bachelor's, 4: Higher)"),
    StudyTimeWeekly: float = Query(..., description="Weekly study time (in hours, 0 to 20)"),
    Absences: int = Query(..., description="Number of absences during academic year (0 to 30)"),
    Tutoring: int = Query(..., description="Receiving tutoring (0: No, 1: Yes)"),
    ParentalSupport: int = Query(..., description="Parental support level (0: None, 1: Low, 2: Moderate, 3: High, 4: Very High)"),
    Extracurricular: int = Query(..., description="Participation in extracurricular activities (0: No, 1: Yes)"),
    Sports: int = Query(..., description="Participation in sports activities (0: No, 1: Yes)"),
    Music: int = Query(..., description="Participation in music activities (0: No, 1: Yes)"),
    Volunteering: int = Query(..., description="Participation in volunteering (0: No, 1: Yes)"),
):
    try:
        if model_data is None:
            logger.error("Model not loaded - cannot make prediction")
            raise HTTPException(status_code=503, detail="Model not available")
        
        logger.info(f"Making prediction for student data: Age={Age}, Gender={Gender}, etc.")
        
        # Feature engineering function (same as in training)
        def create_additional_features(df):
            df_enhanced = df.copy()
            
            # Interaction features
            if 'StudyTimeWeekly' in df.columns and 'Absences' in df.columns:
                df_enhanced['StudyTime_per_Absence'] = df_enhanced['StudyTimeWeekly'] / (df_enhanced['Absences'] + 1)
            
            # Age grouping
            if 'Age' in df.columns:
                df_enhanced['Age_Group'] = pd.cut(df_enhanced['Age'], bins=3, labels=['Young', 'Medium', 'Old'])
                df_enhanced['Age_Group'] = LabelEncoder().fit_transform(df_enhanced['Age_Group'])
            
            return df_enhanced
        
        # Collect data in dictionary
        data = {
            "Age": Age,
            "Gender": Gender,
            "Ethnicity": Ethnicity,
            "ParentalEducation": ParentalEducation,
            "StudyTimeWeekly": StudyTimeWeekly,
            "Absences": Absences,
            "Tutoring": Tutoring,
            "ParentalSupport": ParentalSupport,
            "Extracurricular": Extracurricular,
            "Sports": Sports,
            "Music": Music,
            "Volunteering": Volunteering,
        }
        
        # Convert data to DataFrame
        df = pd.DataFrame([data])
        
        # Apply feature engineering
        df_enhanced = create_additional_features(df)
        
        # Ensure columns are in the same order as training
        feature_columns = model_data['feature_names']
        df_enhanced = df_enhanced[feature_columns]
        
        # Scale the data if required
        if model_data['requires_scaling']:
            df_scaled = model_data['scaler'].transform(df_enhanced)
        else:
            df_scaled = df_enhanced.values
        
        # Make prediction
        model = model_data['model']
        prediction = model.predict(df_scaled)[0]
        prediction_proba = model.predict_proba(df_scaled)[0]
        
        # Grade mapping
        grade_map = {
            0: 'Excellent (90-100%)',
            1: 'Very Good (80-89%)',
            2: 'Good (70-79%)',
            3: 'Acceptable (60-69%)',
            4: 'Fail (Below 60%)'
        }
        
        grade = grade_map.get(prediction, 'Unknown')
        confidence = float(max(prediction_proba))
        
        result = {
            "predicted_grade": grade,
            "confidence": f"{confidence:.2%}",
            "prediction_probabilities": {
                grade_map[i]: f"{prob:.2%}" for i, prob in enumerate(prediction_proba)
            },
            "model_used": model_data['model_name']
        }
        
        logger.info(f"Prediction successful: {grade} with {confidence:.2%} confidence")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check API status"""
    return {
        "status": "API is running",
        "model_loaded": model_data is not None,
        "timestamp": pd.Timestamp.now().isoformat(),
        "base_dir": BASE_DIR,
        "model_path": os.path.join(BASE_DIR, "models", "enhanced_student_perf_model.pkl") if model_data else "Not loaded"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)

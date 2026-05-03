from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load full pipeline (model + preprocessing inside)
pipeline = joblib.load("adult_model_pipeline.pkl")

@app.get("/")
def home():
    return {"message": "ML API is running with Pipeline"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data])
        
        # Direct prediction using pipeline
        prediction = pipeline.predict(df)
        
        return {"prediction": prediction[0]}
    
    except Exception as e:
        return {"error": str(e)}
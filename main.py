import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(title="Liver Disease Prediction Using Machine Learning Techniques")

model = joblib.load("pipeline.pkl")

# Creating the Input Schema for the Swagger UI
class PatientInput(BaseModel):
    Age: int = Field(..., ge=0)
    Gender: int = Field(...)
    BMI: float = Field(...)
    AlcoholConsumption: float = Field(...)
    Smoking: int = Field(...)
    GeneticRisk: int = Field(...)
    PhysicalActivity: float = Field(...)
    Diabetes: int = Field(...)
    Hypertension: int = Field(...)
    LiverFunctionTest: float = Field(...)


@app.get("/status")
def get_status():
    return {"Status": "The Server is Running Fine"}

@app.get("/model-name")
def get_model_name():
    return {"Model Name": "The Model is Gradient Boosting Classifier"}


@app.post("/predict")
def predictLiverDisease(data: PatientInput):
    df = pd.DataFrame(
        [data.dict()]
    )
    prediction = model.predict(df)
    result = "Liver Disease" if prediction[0]==1 else "No Liver Disease"
    return {
        "Prediction": f"The Patient is having {result}"
    }
    
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# =========================
# 1. Load ONE Model
#    (Change filename when needed)
# =========================
MODEL_PATH = r"C:\Users\Leezi Asghar\Downloads\Riaya\API\BF.pkl"  # <-- fixed path
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ========================= 
# 2. Define Input Schema
# =========================
class InputData(BaseModel):
    height: float  # in cm
    weight: float  # in kg
    age: int
    sex: int  

# =========================
# 3. Create API
# =========================
app = FastAPI(title="Body Composition Prediction API")

@app.get("/")
def root():
    return {"status": "API running 🚀", "using_model": MODEL_PATH}

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.height, data.weight ,data.age, data.sex]])
    pred = model.predict(X)[0]
    return {"BF prediction": float(pred)}
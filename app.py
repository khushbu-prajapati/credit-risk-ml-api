from fastapi import FastAPI
from pydantic import BaseModel
from config import MODEL_PATH
import pandas as pd
import pickle
import logging

logging.basicConfig(level=logging.INFO)
# ---------- LOAD MODEL ----------
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")
app = FastAPI()

# ⚠️ IMPORTANT: apne dataset ke columns yaha likho
class InputData(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: float

# ---------- HOME ----------
@app.get("/")
def home():
    return {"message": "Credit Risk API Running 🚀"}

# ---------- PREDICT ----------
@app.post("/predict")
def predict(data: InputData):
    try:
        logging.info(f"Input received: {data}")

        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)

        logging.info(f"Prediction: {prediction[0]}")
        risk = "High Risk" if prediction[0] == 1 else "Low Risk"

        return {
            "prediction": int(prediction[0]),
            "risk_level": risk
        }

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return {"error": str(e)}

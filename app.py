from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# ---------- LOAD MODEL ----------
model = pickle.load(open("model.pkl", "rb"))

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
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
# Credit Risk Prediction API

## Overview
This project predicts whether a person will default on a loan using Machine Learning.

## Features
- Data preprocessing using pipeline
- Random Forest model
- FastAPI deployment
- Real-time prediction API

## Tech Stack
- Python
- Scikit-learn
- FastAPI

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Train the model:
python train.py

3. Run API:
uvicorn app:app --reload

## API Endpoint

POST /predict

Example Input:
{
  "person_age": 30,
  "person_income": 50000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5,
  "loan_intent": "PERSONAL",
  "loan_grade": "A",
  "loan_amnt": 20000,
  "loan_int_rate": 10.5,
  "loan_percent_income": 0.4,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 8
}

## Note
Run train.py before starting the API to generate model.pkl

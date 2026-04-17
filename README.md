# Credit Risk Prediction API 🚀

## 📌 Overview

This project builds an end-to-end Machine Learning system to predict whether a loan applicant is likely to default.
The model is trained using financial and demographic data and deployed using FastAPI for real-time predictions.

---

## 🎯 Business Problem

Financial institutions need to identify high-risk loan applicants to minimize financial losses.
This project helps in predicting loan default risk based on user attributes.

---

## 🧠 Solution

* Built a machine learning pipeline for data preprocessing and modeling
* Used Random Forest Classifier for prediction
* Extracted feature importance to understand key risk drivers
* Deployed the model using FastAPI for real-time inference

---

## ⚙️ Features

* End-to-end ML pipeline (preprocessing + model)
* Feature importance analysis
* FastAPI-based REST API
* Input validation using Pydantic
* Logging and error handling
* Real-time prediction system

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* FastAPI
* SQLAlchemy
* PyODBC

---

## 📊 Key Insights (Feature Importance)

Top factors affecting loan default:

* loan_percent_income (most important)
* person_income
* loan_int_rate
* loan_amnt

👉 Model focuses more on financial burden rather than demographic features.

---

## 🚀 How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train the model

```
python train.py
```

### 3. Run the API

```
uvicorn app:app --reload
```

### 4. Open Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## 🔌 API Endpoint

### POST `/predict`

#### Example Input:

```json
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
```

#### Example Output:

```json
{
  "prediction": 1,
  "risk_level": "High Risk"
}
```

---

## 📁 Project Structure

```
credit-risk-project/
│
├── app.py
├── train.py
├── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚠️ Important Notes

* Run `train.py` before starting the API to generate `model.pkl`
* `model.pkl` is not included in the repository (to keep repo lightweight)

---

## 💡 Future Improvements

* Add SHAP for explainable AI
* Deploy on cloud (Render / AWS)
* Build frontend UI (Streamlit)

---

## 👩‍💻 Author

Khushbu Prajapati

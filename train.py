import pandas as pd
import pickle
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from config import DATABASE_URL, MODEL_PATH

# ---------- DATABASE CONNECTION ----------
engine = create_engine(DATABASE_URL)

# ---------- LOAD DATA ----------
df = pd.read_sql("SELECT * FROM credit_risk_dataset", engine)

# ---------- BASIC CLEANING ----------
df = df.dropna()

# ---------- FEATURES ----------
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- PIPELINE ----------
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

model = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

# ---------- TRAIN ----------
model.fit(X_train, y_train)

# ---------- FEATURE IMPORTANCE ----------
ohe = model.named_steps["preprocess"].named_transformers_["cat"]
cat_features = ohe.get_feature_names_out(cat_cols)

all_features = num_cols + list(cat_features)

importances = model.named_steps["model"].feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": all_features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n🔥 Top 10 Important Features:\n")
print(feature_importance_df.head(10))

# ---------- SAVE MODEL ----------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model trained and saved as model.pkl")

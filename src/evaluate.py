# src/evaluate.py
import pandas as pd
import yaml
import json
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["evaluate"]

# Загрузка модели и данных
model = joblib.load(params["input_path"])
df = pd.read_csv(params["test_data"])

# Целевая переменная - Survived (0 = не выжил, 1 = выжил)
y_true = df["Survived"]

# Признаки - все числовые столбцы, кроме Survived
numeric_cols = df.select_dtypes(include=['number']).columns.drop("Survived")
X = df[numeric_cols]

# Предсказания
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]  # Вероятность класса 1 (выжил)

# Вычисление метрик
metrics = {}
if "accuracy" in params["params"]["metrics"]:
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
if "f1" in params["params"]["metrics"]:
    metrics["f1"] = f1_score(y_true, y_pred)
if "roc_auc" in params["params"]["metrics"]:
    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

# Сохранение метрик
with open(params["output_path"], "w") as f:
    json.dump(metrics, f, indent=4)

import pandas as pd
import yaml
import json
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["evaluate"]

model = joblib.load(params["input_path"])
df = pd.read_csv(params["test_data"])

y_true = df["Survived"]

numeric_cols = df.select_dtypes(include=['number']).columns.drop("Survived")
X = df[numeric_cols]

y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]  

metrics = {}
if "accuracy" in params["params"]["metrics"]:
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
if "f1" in params["params"]["metrics"]:
    metrics["f1"] = f1_score(y_true, y_pred)
if "roc_auc" in params["params"]["metrics"]:
    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

with open(params["output_path"], "w") as f:
    json.dump(metrics, f, indent=4)

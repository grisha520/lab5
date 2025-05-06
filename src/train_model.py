import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

df = pd.read_csv(params["input_path"])

y = df["Survived"]

numeric_cols = df.select_dtypes(include=['number']).columns.drop("Survived")
X = df[numeric_cols]

model = RandomForestClassifier(
    n_estimators=params["params"]["n_estimators"],
    random_state=params["params"]["random_state"]
)
model.fit(X, y)

joblib.dump(model, params["output_path"])

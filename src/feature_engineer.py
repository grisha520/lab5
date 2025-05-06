import pandas as pd
import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["feature_engineering"]

df = pd.read_csv(params["input_path"])

if "ratio_feature" in params["params"]["new_features"]:
    df["age_to_fare_ratio"] = df["Age"] / (df["Fare"] + 1e-6)

if "interaction_feature" in params["params"]["new_features"]:
    df["class_family_interaction"] = df["Pclass"] * (df["Siblings/Spouses Aboard"] + df["Parents/Children Aboard"])

df.to_csv(params["output_path"], index=False)

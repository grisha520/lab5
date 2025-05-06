# src/feature_engineer.py
import pandas as pd
import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["feature_engineering"]

df = pd.read_csv(params["input_path"])

# Генерация новых признаков
if "ratio_feature" in params["params"]["new_features"]:
    # Отношение возраста к цене билета (защита от деления на 0)
    df["age_to_fare_ratio"] = df["Age"] / (df["Fare"] + 1e-6)

if "interaction_feature" in params["params"]["new_features"]:
    # Взаимодействие класса и количества родственников на борту
    df["class_family_interaction"] = df["Pclass"] * (df["Siblings/Spouses Aboard"] + df["Parents/Children Aboard"])

# Сохранение
df.to_csv(params["output_path"], index=False)

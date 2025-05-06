# src/train_model.py
import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

# Загрузка данных
df = pd.read_csv(params["input_path"])

# Целевая переменная - Survived (0 = не выжил, 1 = выжил)
y = df["Survived"]

# Признаки - все числовые столбцы, кроме Survived
numeric_cols = df.select_dtypes(include=['number']).columns.drop("Survived")
X = df[numeric_cols]

# Обучение модели
model = RandomForestClassifier(
    n_estimators=params["params"]["n_estimators"],
    random_state=params["params"]["random_state"]
)
model.fit(X, y)

# Сохранение модели
joblib.dump(model, params["output_path"])

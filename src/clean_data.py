# src/clean_data.py
import pandas as pd
import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["clean_data"]

df = pd.read_csv(params["input_path"])

if params["params"]["drop_na"]:
    df = df.dropna()
else:
    df = df.fillna(params["params"]["fill_value"])

df.to_csv(params["output_path"], index=False)
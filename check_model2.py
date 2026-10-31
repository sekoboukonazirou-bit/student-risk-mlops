import joblib
import pandas as pd

m = joblib.load("model.pkl")
print("n_features_in_ :", m.n_features_in_)
print("Paramètres     :", m.get_params())

df = pd.read_csv("data/data.csv", sep=";")
print("\nCible :\n", df["Target"].value_counts())
print("\nTypes de colonnes :\n", df.dtypes.value_counts())
print("\nValeurs manquantes :", int(df.isna().sum().sum()))
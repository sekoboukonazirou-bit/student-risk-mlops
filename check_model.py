import sys

import joblib
import pandas as pd
import sklearn

print("Python :", sys.version.split()[0])
print("scikit-learn (ici) :", sklearn.__version__)
print("pandas (ici)       :", pd.__version__)

model = joblib.load("model.pkl")
print("\nType du modèle :", type(model))
if hasattr(model, "steps"):
    print("Étapes du pipeline :", [name for name, _ in model.steps])

print("\nColonnes attendues :", list(getattr(model, "feature_names_in_", [])))
print("Classes prédites   :", list(getattr(model, "classes_", [])))

df = pd.read_csv("data/data.csv", sep=";")
print("\nColonnes du CSV :", df.columns.tolist())
print("Nombre de lignes :", len(df))
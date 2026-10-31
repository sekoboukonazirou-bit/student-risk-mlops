import json

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("scikit-learn installé :", sklearn.__version__, "(le modèle a été entraîné avec 1.6.1)")

# 1. Mêmes étapes que le notebook
df = pd.read_csv("data/data.csv", sep=";")
X = df.drop(columns=["Target"])
y = df["Target"]

le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Correspondance de la cible :", {int(i): c for i, c in enumerate(le.classes_)})

X = pd.get_dummies(X, columns=["Course"], drop_first=True)
print("Nombre de variables après encodage :", X.shape[1], "(le modèle en attend 51)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Preuve : on retrouve la précision du notebook avec model.pkl
model = joblib.load("model.pkl")
_, X_test, _, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)
acc = accuracy_score(y_test, model.predict(X_test))
print(f"\nPrécision retrouvée : {acc:.4f}   (notebook : 0.7571)")

# 3. Sauvegarde du prétraitement
raw_columns = [c for c in df.columns if c != "Target"]
artifact = {
    "scaler": scaler,
    "model_columns": list(X.columns),
    "raw_columns": raw_columns,
    "course_values": sorted(int(c) for c in df["Course"].unique()),
    "labels": [str(c) for c in le.classes_],
}
joblib.dump(artifact, "preprocessor.pkl")

# 4. Un exemple d'étudiant prêt à coller dans Swagger (noms de colonnes nettoyés)
sample = json.loads(
    df.drop(columns=["Target"]).iloc[[0]].rename(columns=str.strip).to_json(orient="records")
)[0]
with open("exemple_etudiant.json", "w", encoding="utf-8") as fh:
    json.dump(sample, fh, indent=2, ensure_ascii=False)
print("\nFichiers créés : preprocessor.pkl, exemple_etudiant.json")
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


class StudentRiskPredictor:
    def __init__(self, model_path=BASE_DIR / "model.pkl", prep_path=BASE_DIR / "preprocessor.pkl"):
        self.model = joblib.load(model_path)
        prep = joblib.load(prep_path)
        self.scaler = prep["scaler"]
        self.model_columns = prep["model_columns"]    # les 51 colonnes, dans l'ordre du modèle
        self.raw_columns = prep["raw_columns"]        # les 36 colonnes d'origine du CSV
        self.course_values = prep["course_values"]    # cours connus
        self.labels = prep["labels"]                  # ['Dropout', 'Enrolled', 'Graduate']
        # Noms « propres » pour l'API (le CSV contient une tabulation dans un nom de colonne)
        self.api_names = [c.strip() for c in self.raw_columns]

    def build_features(self, payload: dict) -> pd.DataFrame:
        missing = [n for n in self.api_names if n not in payload]
        if missing:
            raise ValueError(f"Variables manquantes : {missing}")

        row = {raw: payload[api] for raw, api in zip(self.raw_columns, self.api_names)}
        df = pd.DataFrame([row])
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                raise ValueError(f"La variable '{col.strip()}' doit être un nombre (reçu : {row[col]!r})")

        course = int(df.at[0, "Course"])
        if course not in self.course_values:
            raise ValueError(f"Course inconnu : {course}. Valeurs possibles : {self.course_values}")

        # Équivalent de get_dummies(columns=['Course'], drop_first=True) pour UN étudiant
        df = df.drop(columns=["Course"])
        for col in self.model_columns:
            if col.startswith("Course_"):
                df[col] = int(col == f"Course_{course}")
        return df[self.model_columns]

    def predict(self, payload: dict) -> dict:
        features = self.build_features(payload)
        scaled = self.scaler.transform(features)
        class_id = int(self.model.predict(scaled)[0])
        probas = self.model.predict_proba(scaled)[0]
        return {
            "prediction": self.labels[class_id],
            "class_id": class_id,
            "probabilities": {lab: round(float(p), 4) for lab, p in zip(self.labels, probas)},
        }
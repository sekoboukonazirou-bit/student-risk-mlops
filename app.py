import logging

import uvicorn
from fastapi import FastAPI, HTTPException

from inference import StudentRiskPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("student-risk-api")

app = FastAPI(title="Student Risk Prediction API")

# Chargé une seule fois, au démarrage (pas à chaque requête)
predictor = StudentRiskPredictor()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    return {
        "features": predictor.api_names,
        "courses_valides": predictor.course_values,
        "classes": predictor.labels,
    }


@app.post("/predict")
def predict(data: dict):
    try:
        result = predictor.predict(data)
    except ValueError as e:          # erreur de l'utilisateur : donnée absente ou invalide
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:                # erreur inattendue : on la journalise sans l'exposer
        logger.exception("Erreur d'inférence")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction")
    logger.info("prediction=%s", result["prediction"])
    return result


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
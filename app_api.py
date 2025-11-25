from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np

# Modèles Pydantic
class CommentRequest(BaseModel):
    comments: List[str] = Field(..., min_items=1, max_items=1000)

class SentimentResult(BaseModel):
    text: str
    sentiment: int
    sentiment_label: str
    confidence: float

class PredictionResponse(BaseModel):
    results: List[SentimentResult]
    statistics: dict

# Initialiser l'application
app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="API pour analyser le sentiment des commentaires YouTube",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger les modèles au démarrage
try:
    model = joblib.load("sentiment_model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    print("✓ Modèles chargés avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement des modèles: {e}")
    model = None
    vectorizer = None

@app.get("/")
async def root():
    """Page d'accueil"""
    return {
        "message": "YouTube Sentiment Analysis API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Vérifier l'état de l'API",
            "/predict_batch": "Analyser un batch de commentaires"
        }
    }

@app.get("/health")
async def health_check():
    """Vérifie l'état de l'API et du modèle"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Modèles non chargés")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "vectorizer_loaded": True
    }

@app.post("/predict_batch", response_model=PredictionResponse)
async def predict_batch(request: CommentRequest):
    """Analyse un batch de commentaires"""
    
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Modèles non chargés")
    
    try:
        # Vectoriser
        X = vectorizer.transform(request.comments)
        
        # Prédire
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Mapper les labels
        sentiment_map = {
            -1: "Négatif",
            0: "Neutre",
            1: "Positif"
        }
        
        # Créer les résultats
        results = []
        for text, pred, proba in zip(request.comments, predictions, probabilities):
            max_proba = float(np.max(proba))
            results.append(SentimentResult(
                text=text,
                sentiment=int(pred),
                sentiment_label=sentiment_map[int(pred)],
                confidence=round(max_proba, 3)
            ))
        
        # Calculer les statistiques
        total = len(predictions)
        stats = {
            "total_comments": total,
            "positive": int(np.sum(predictions == 1)),
            "neutral": int(np.sum(predictions == 0)),
            "negative": int(np.sum(predictions == -1)),
            "positive_percentage": round((np.sum(predictions == 1) / total) * 100, 2),
            "neutral_percentage": round((np.sum(predictions == 0) / total) * 100, 2),
            "negative_percentage": round((np.sum(predictions == -1) / total) * 100, 2),
            "average_confidence": round(float(np.mean([np.max(p) for p in probabilities])), 3)
        }
        
        return PredictionResponse(results=results, statistics=stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")
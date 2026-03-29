from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
from datetime import datetime

# Modèles Pydantic
class Comment(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class CommentBatch(BaseModel):
    comments: List[str] = Field(..., min_items=1, max_items=100)

class PredictionResult(BaseModel):
    text: str
    sentiment: str
    sentiment_score: int
    confidence: float

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResult]
    statistics: dict
    processing_time: float

# Initialiser FastAPI
app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="API pour l'analyse de sentiment de commentaires YouTube",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les origines exactes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle au démarrage
try:
    vectorizer = joblib.load('models/vectorizer.joblib')
    model = joblib.load('models/model.joblib')
    print("✅ Modèle chargé avec succès!")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    vectorizer = None
    model = None

# Mapping des sentiments
SENTIMENT_MAP = {
    -1: "négatif",
    0: "neutre",
    1: "positif"
}

@app.get("/")
def read_root():
    """Page d'accueil de l'API"""
    return {
        "message": "YouTube Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Vérifier l'état de l'API",
            "/predict_batch": "Analyser un batch de commentaires"
        }
    }

@app.get("/health")
def health_check():
    """Endpoint de santé pour vérifier l'état de l'API"""
    
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(batch: CommentBatch):
    """
    Analyse un batch de commentaires et retourne les sentiments
    """
    
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        import time
        start_time = time.time()
        
        # Vectoriser les commentaires
        X = vectorizer.transform(batch.comments)
        
        # Prédire
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Préparer les résultats
        results = []
        for text, pred, probs in zip(batch.comments, predictions, probabilities):
            confidence = float(np.max(probs))
            
            results.append(PredictionResult(
                text=text,
                sentiment=SENTIMENT_MAP[pred],
                sentiment_score=int(pred),
                confidence=round(confidence, 4)
            ))
        
        # Calculer les statistiques
        sentiment_counts = {
            "positif": int(np.sum(predictions == 1)),
            "neutre": int(np.sum(predictions == 0)),
            "négatif": int(np.sum(predictions == -1))
        }
        
        total = len(predictions)
        sentiment_percentages = {
            k: round((v / total) * 100, 2)
            for k, v in sentiment_counts.items()
        }
        
        statistics = {
            "total_comments": total,
            "counts": sentiment_counts,
            "percentages": sentiment_percentages,
            "average_confidence": round(float(np.mean([r.confidence for r in results])), 4)
        }
        
        processing_time = round((time.time() - start_time) * 1000, 2)  # en ms
        
        return BatchPredictionResponse(
            results=results,
            statistics=statistics,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
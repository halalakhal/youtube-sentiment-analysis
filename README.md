---
title: YouTube Sentiment Analysis API
emoji: 🎭
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# YouTube Sentiment Analysis API

API REST pour analyser le sentiment des commentaires YouTube en temps réel.

## Endpoints

- `GET /` - Page d'accueil
- `GET /health` - Vérification de l'état de l'API
- `POST /predict_batch` - Analyse de sentiment pour un batch de commentaires

## Utilisation
```bash
curl -X POST https://your-space.hf.space/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"comments": ["This is amazing!", "I hate this", "It is okay"]}'
```

## Modèle

- Vectorisation: TF-IDF
- Algorithme: Logistic Regression
- Classes: Positif (1), Neutre (0), Négatif (-1)
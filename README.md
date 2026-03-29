# YouTube Sentiment Analysis

ML pipeline for analyzing sentiment in YouTube comments with real-time Chrome extension and cloud-deployed API.

## 🎯 Project Overview

This project implements a complete MLOps pipeline for sentiment analysis:
- **Dataset:** Reddit Sentiment Analysis (36,450 samples)
- **Model:** TF-IDF + Logistic Regression
- **Performance:** 87.9% accuracy, F1-macro 0.866
- **Deployment:** FastAPI on HuggingFace Spaces
- **Interface:** Chrome extension for YouTube

## 🚀 Quick Start

### Installation
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Training
```bash
python src/models/train_model.py
```

### API Server
```bash
uvicorn app_api:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
docker build -t sentiment-api .
docker run -p 7860:7860 sentiment-api
```

## 📊 Model Details

- **Architecture:** TF-IDF (5000 features, bigrams) + Logistic Regression
- **Hyperparameters:** C=10, L2 penalty, lbfgs solver
- **Training time:** 45 seconds (CPU)
- **Inference:** <100ms for 50 comments

## 🌐 Deployment

**Live API:** https://halalakhal-youtube-sentiment-api.hf.space

**Endpoints:**
- `GET /health` - Health check
- `POST /predict_batch` - Batch sentiment prediction

## 📁 Project Structure
```
├── src/               # Source code
│   ├── data/         # Data processing
│   └── models/       # Model training
├── models/           # Saved models
├── chrome-extension/ # Chrome extension
├── tests/            # Unit tests
└── app_api.py       # FastAPI application
```

## 🧪 Testing
```bash
python tests/test_model.py
python tests/test_api.py
```

## 📝 License

MIT License

## 👤 Author

**Hala Lakhal**
- GitHub: [@halalakhal](https://github.com/halalakhal)
- HuggingFace: [halalakhal](https://huggingface.co/halalakhal)
```
# Core ML
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0

# API
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Utilities
joblib==1.3.2
```

#### `requirements_prod.txt`
```
# Production dependencies (minimal)
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
joblib==1.3.2
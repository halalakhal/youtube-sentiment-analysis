import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import time

def test_model_performance():
    """Test les performances du modèle"""
    
    print("🧪 TEST 1: Performance sur le test set")
    print("=" * 50)
    
    # Charger le modèle et le test set
    model = joblib.load("models/model.joblib")
    vectorizer = joblib.load("models/vectorizer.joblib")
    test_df = pd.read_csv("./data/processed/clean_reddit.csv")
    
    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['label']
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Négatif', 'Neutre', 'Positif']))
    
    assert accuracy >= 0.80, "❌ Accuracy < 80%"
    print("✅ Test réussi: Accuracy >= 80%")

def test_edge_cases():
    """Test avec des cas limites"""
    
    print("\n🧪 TEST 2: Cas limites")
    print("=" * 50)
    
    model = joblib.load("models/model.joblib")
    vectorizer = joblib.load("models/vectorizer.joblib")
    
    test_cases = [
        ("", "Texte vide"),
        ("a", "Texte très court"),
        ("This is a very long comment " * 50, "Texte très long"),
        ("😊😊😊", "Emojis seulement"),
        ("AWESOME!!!", "Majuscules et ponctuation"),
        ("Good bad good bad", "Mélange positif/négatif")
    ]
    
    for text, description in test_cases:
        try:
            X = vectorizer.transform([text])
            pred = model.predict(X)[0]
            print(f"✓ {description}: {pred}")
        except Exception as e:
            print(f"❌ {description}: Erreur - {e}")

def test_inference_time():
    """Test le temps d'inférence"""
    
    print("\n🧪 TEST 3: Temps d'inférence")
    print("=" * 50)
    
    model = joblib.load("models/model.joblib")
    vectorizer = joblib.load("models/vectorizer.joblib")
    
    # Créer un batch de 50 commentaires
    test_comments = ["This is a test comment"] * 50
    
    X = vectorizer.transform(test_comments)
    
    start_time = time.time()
    predictions = model.predict(X)
    end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000
    
    print(f"Temps pour 50 commentaires: {inference_time:.2f}ms")
    
    assert inference_time < 100, "❌ Temps d'inférence > 100ms"
    print("✅ Test réussi: Temps d'inférence < 100ms")

if __name__ == "__main__":
    test_model_performance()
    test_edge_cases()
    test_inference_time()
    print("\n" + "=" * 50)
    print("🎉 Tous les tests sont terminés!")
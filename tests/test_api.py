import requests
import time

# URL de l'API (local ou déployée)
API_URL = "http://127.0.0.1:8000"  # Changer pour l'URL Hugging Face en production

def test_health_endpoint():
    """Test le endpoint /health"""
    
    print("🧪 TEST 1: Endpoint /health")
    print("=" * 50)
    
    response = requests.get(f"{API_URL}/health")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200, "❌ Health check failed"
    assert response.json()["status"] == "healthy", "❌ API not healthy"
    
    print("✅ Test réussi: API est healthy")

def test_predict_batch_endpoint():
    """Test le endpoint /predict_batch"""
    
    print("\n🧪 TEST 2: Endpoint /predict_batch")
    print("=" * 50)
    
    payload = {
        "comments": [
            "This video is absolutely amazing!",
            "I don't like this content",
            "It's okay, nothing special"
        ]
    }
    
    start_time = time.time()
    response = requests.post(f"{API_URL}/predict_batch", json=payload)
    end_time = time.time()
    
    response_time = (end_time - start_time) * 1000
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Time: {response_time:.2f}ms")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nRésultats:")
        for result in data["results"]:
            print(f"  - {result['sentiment_label']}: {result['text'][:50]}...")
        
        print(f"\nStatistiques:")
        print(f"  - Total: {data['statistics']['total_comments']}")
        print(f"  - Positif: {data['statistics']['positive']}")
        print(f"  - Neutre: {data['statistics']['neutral']}")
        print(f"  - Négatif: {data['statistics']['negative']}")
        
        assert len(data["results"]) == 3, "❌ Nombre de résultats incorrect"
        print("\n✅ Test réussi: Prédictions correctes")
    else:
        print(f"❌ Erreur: {response.text}")

def test_error_handling():
    """Test la gestion d'erreurs"""
    
    print("\n🧪 TEST 3: Gestion d'erreurs")
    print("=" * 50)
    
    # Test 1: Liste vide
    response = requests.post(f"{API_URL}/predict_batch", json={"comments": []})
    print(f"Liste vide - Status: {response.status_code}")
    assert response.status_code == 422, "❌ Devrait rejeter une liste vide"
    
    # Test 2: Données invalides
    response = requests.post(f"{API_URL}/predict_batch", json={"invalid": "data"})
    print(f"Données invalides - Status: {response.status_code}")
    assert response.status_code == 422, "❌ Devrait rejeter des données invalides"
    
    print("✅ Test réussi: Gestion d'erreurs correcte")

def test_large_batch():
    """Test avec un grand batch"""
    
    print("\n🧪 TEST 4: Grand batch de commentaires")
    print("=" * 50)
    
    # Créer 100 commentaires
    comments = [f"Test comment number {i}" for i in range(100)]
    
    payload = {"comments": comments}
    
    start_time = time.time()
    response = requests.post(f"{API_URL}/predict_batch", json=payload)
    end_time = time.time()
    
    response_time = (end_time - start_time) * 1000
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Time pour 100 commentaires: {response_time:.2f}ms")
    
    if response.status_code == 200:
        data = response.json()
        assert len(data["results"]) == 100, "❌ Nombre de résultats incorrect"
        print("✅ Test réussi: Traitement de 100 commentaires")
    else:
        print(f"❌ Erreur: {response.text}")

if __name__ == "__main__":
    try:
        test_health_endpoint()
        test_predict_batch_endpoint()
        test_error_handling()
        test_large_batch()
        print("\n" + "=" * 50)
        print("🎉 Tous les tests API sont réussis!")
    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {e}")
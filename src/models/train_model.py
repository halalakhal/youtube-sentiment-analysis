import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time

class SentimentModel:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        
    def prepare_data(self, data_path, test_size=0.2, random_state=42):
        """Prépare les données pour l'entraînement"""
        
        print("📂 Chargement des données...")
        df = pd.read_csv(data_path)
        
        X = df['text']
        y = df['label']
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✅ Données préparées:")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Test: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_vectorizer(self, X_train):
        """Entraîne le vectoriseur TF-IDF"""
        
        print("\n🔤 Entraînement du vectoriseur TF-IDF...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            strip_accents='unicode',
            lowercase=True
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        print(f"✅ Vocabulaire: {len(self.vectorizer.vocabulary_)} mots")
        print(f"✅ Shape des features: {X_train_vec.shape}")
        
        return X_train_vec
    
    def train_model(self, X_train_vec, y_train, optimize=True):
        """Entraîne le modèle de classification"""
        
        print("\n🎯 Entraînement du modèle...")
        
        if optimize:
            print("   Optimisation des hyperparamètres avec GridSearch...")
            
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs'],
                'max_iter': [1000]
            }
            
            lr = LogisticRegression(random_state=42)
            
            grid_search = GridSearchCV(
                lr, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train_vec, y_train)
            
            self.model = grid_search.best_estimator_
            print(f"\n✅ Meilleurs paramètres: {grid_search.best_params_}")
            print(f"✅ Meilleur score F1: {grid_search.best_score_:.4f}")
            
        else:
            self.model = LogisticRegression(
                C=1, max_iter=1000, random_state=42
            )
            self.model.fit(X_train_vec, y_train)
        
        print("✅ Modèle entraîné!")
        
    def evaluate(self, X_test, y_test):
        """Évalue le modèle"""
        
        print("\n📊 ÉVALUATION DU MODÈLE")
        print("=" * 60)
        
        # Vectoriser les données de test
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Prédictions
        y_pred = self.model.predict(X_test_vec)
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"\n🎯 Accuracy: {accuracy:.4f}")
        print(f"🎯 F1-Score (macro): {f1:.4f}")
        
        print("\n📋 Rapport de classification:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Négatif', 'Neutre', 'Positif']))
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Négatif', 'Neutre', 'Positif'],
                    yticklabels=['Négatif', 'Neutre', 'Positif'])
        plt.title('Matrice de Confusion')
        plt.ylabel('Vraie classe')
        plt.xlabel('Classe prédite')
        plt.tight_layout()
        
        os.makedirs('logs', exist_ok=True)
        plt.savefig('logs/confusion_matrix.png')
        print("\n✅ Matrice de confusion sauvegardée dans logs/confusion_matrix.png")
        
        # Test de temps d'inférence
        print("\n⏱️ Test de performance:")
        batch_size = 50
        test_batch = X_test.head(batch_size)
        test_batch_vec = self.vectorizer.transform(test_batch)
        
        start_time = time.time()
        _ = self.model.predict(test_batch_vec)
        inference_time = (time.time() - start_time) * 1000  # en ms
        
        print(f"   Temps d'inférence pour {batch_size} commentaires: {inference_time:.2f}ms")
        print(f"   Temps moyen par commentaire: {inference_time/batch_size:.2f}ms")
        
        return accuracy, f1
    
    def save_model(self, model_dir='models'):
        """Sauvegarde le modèle et le vectoriseur"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.vectorizer, f'{model_dir}/vectorizer.joblib')
        joblib.dump(self.model, f'{model_dir}/model.joblib')
        
        print(f"\n✅ Modèle sauvegardé dans {model_dir}/")
    
    def load_model(self, model_dir='models'):
        """Charge le modèle et le vectoriseur"""
        
        self.vectorizer = joblib.load(f'{model_dir}/vectorizer.joblib')
        self.model = joblib.load(f'{model_dir}/model.joblib')
        
        print("✅ Modèle chargé!")

def main():
    # Initialiser le modèle
    sentiment_model = SentimentModel()
    
    # Préparer les données
    X_train, X_test, y_train, y_test = sentiment_model.prepare_data(
        'data/processed/clean_reddit.csv'
    )
    
    # Entraîner le vectoriseur
    X_train_vec = sentiment_model.train_vectorizer(X_train)
    
    # Entraîner le modèle
    sentiment_model.train_model(X_train_vec, y_train, optimize=True)
    
    # Évaluer
    sentiment_model.evaluate(X_test, y_test)
    
    # Sauvegarder
    sentiment_model.save_model()
    
    print("\n" + "=" * 60)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("=" * 60)

if __name__ == "__main__":
    main()
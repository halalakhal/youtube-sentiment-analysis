import pandas as pd
import requests
import os

def download_dataset():
    """Télécharge le dataset Reddit depuis GitHub"""
    
    url = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
    
    print("📥 Téléchargement du dataset...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Créer le dossier si nécessaire
        os.makedirs('data/raw', exist_ok=True)
        
        # Sauvegarder le fichier
        with open('data/raw/reddit.csv', 'wb') as f:
            f.write(response.content)
        
        print("✅ Dataset téléchargé avec succès!")
        
        # Charger et afficher les statistiques
        df = pd.read_csv('data/raw/reddit.csv')
        
        print("\n📊 Statistiques du dataset:")
        print(f"Nombre total de commentaires: {len(df)}")
        print(f"\nDistribution des labels:")
        print(df['category'].value_counts())
        print(f"\nNombre de colonnes: {len(df.columns)}")
        print(f"Colonnes: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement: {e}")
        return None

if __name__ == "__main__":
    download_dataset()
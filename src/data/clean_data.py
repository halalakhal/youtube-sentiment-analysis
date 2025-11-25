import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Télécharger les stopwords
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class DataCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Nettoie un texte"""
        if pd.isna(text):
            return ""
        
        # Convertir en minuscules
        text = text.lower()
        
        # Supprimer les URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Supprimer les mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Supprimer les hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Supprimer les caractères spéciaux mais garder les espaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def process_dataset(self, input_path, output_path):
        """Nettoie le dataset complet"""
        
        print("🧹 Nettoyage du dataset...")
        
        # Charger les données
        df = pd.read_csv(input_path)
        
        # Renommer les colonnes si nécessaire
        if 'clean_comment' in df.columns:
            df = df.rename(columns={'clean_comment': 'text', 'category': 'label'})
        
        print(f"Données brutes: {len(df)} lignes")
        
        # Nettoyer les textes
        df['text'] = df['text'].apply(self.clean_text)
        
        # Supprimer les lignes vides
        df = df[df['text'].str.len() > 0]
        
        # Mapper les labels (-1, 0, 1)
        # Le dataset Reddit utilise déjà -1, 0, 1
        
        # Supprimer les doublons
        df = df.drop_duplicates(subset=['text'])
        
        print(f"Données nettoyées: {len(df)} lignes")
        print(f"\nDistribution des labels:")
        print(df['label'].value_counts())
        
        # Sauvegarder
        df.to_csv(output_path, index=False)
        print(f"\n✅ Données sauvegardées dans {output_path}")
        
        return df

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.process_dataset(
        'data/raw/reddit.csv',
        'data/processed/clean_reddit.csv'
    )
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(data_path):
    """Effectue une analyse exploratoire des données"""
    
    df = pd.read_csv(data_path)
    
    # Créer le dossier pour les visualisations
    os.makedirs('logs', exist_ok=True)
    
    print("📊 ANALYSE EXPLORATOIRE DES DONNÉES\n")
    
    # 1. Statistiques de base
    print("=" * 50)
    print("1. STATISTIQUES DE BASE")
    print("=" * 50)
    print(f"Nombre total de commentaires: {len(df)}")
    print(f"\nDistribution des labels:")
    print(df['label'].value_counts())
    print(f"\nPourcentages:")
    print(df['label'].value_counts(normalize=True) * 100)
    
    # 2. Longueur des textes
    print("\n" + "=" * 50)
    print("2. LONGUEUR DES TEXTES")
    print("=" * 50)
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print(df[['text_length', 'word_count']].describe())
    
    # 3. Visualisations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distribution des labels
    df['label'].value_counts().plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Distribution des Labels')
    axes[0, 0].set_xlabel('Label')
    axes[0, 0].set_ylabel('Nombre de commentaires')
    
    # Longueur des textes par label
    df.boxplot(column='text_length', by='label', ax=axes[0, 1])
    axes[0, 1].set_title('Longueur des textes par label')
    
    # Nombre de mots par label
    df.boxplot(column='word_count', by='label', ax=axes[1, 0])
    axes[1, 0].set_title('Nombre de mots par label')
    
    # Distribution de la longueur des textes
    axes[1, 1].hist(df['text_length'], bins=50)
    axes[1, 1].set_title('Distribution de la longueur des textes')
    axes[1, 1].set_xlabel('Longueur')
    axes[1, 1].set_ylabel('Fréquence')
    
    plt.tight_layout()
    plt.savefig('logs/eda_analysis.png')
    print("\n✅ Visualisations sauvegardées dans logs/eda_analysis.png")
    
    # 4. Échantillons de textes
    print("\n" + "=" * 50)
    print("3. EXEMPLES DE COMMENTAIRES")
    print("=" * 50)
    for label in [-1, 0, 1]:
        print(f"\nLabel {label}:")
        samples = df[df['label'] == label]['text'].head(3)
        for i, text in enumerate(samples, 1):
            print(f"  {i}. {text[:100]}...")

if __name__ == "__main__":
    perform_eda('data/processed/clean_reddit.csv')
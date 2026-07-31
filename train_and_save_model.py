import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
# from sklearn.base import BaseEstimator, TransformerMixin # Pour le CustomTransformer
import joblib # Pour sauvegarder et charger les modèles
import re # Pour nettoyer le texte des acteurs/genres
import os
from custom_transformers import FeatureWeightingTransformer

BASE_DIR = os.getenv("CINECLICK_BASE_DIR", "/home/izak/cineclick")

# # --- 1. Custom Transformer pour la pondération des features ---
# class FeatureWeightingTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, feature_weights=None):
#         # Définir les poids par défaut si non spécifiés
#         # Ces poids sont arbitraires et peuvent être ajustés
#         self.feature_weights = feature_weights if feature_weights is not None else {
#             'overview_clean': 1.0,
#             'directors_name': 3.0, # Le réalisateur a un poids plus élevé
#             'genres_list': 2.0,   # Les genres ont un poids moyen
#             'actors_names': 1.5    # Les acteurs ont un poids légèrement plus élevé
#         }

#     def fit(self, X, y=None):
#         # Ce transformer n'apprend rien des données, donc fit ne fait rien
#         return self

#     def transform(self, X):
#         # X doit être un DataFrame contenant les colonnes d'intérêt
#         if not isinstance(X, pd.DataFrame):
#             raise TypeError("FeatureWeightingTransformer attend un pandas.DataFrame en entrée.")

#         weighted_texts = []
#         for index, row in X.iterrows():
#             combined_text = []
#             for feature, weight in self.feature_weights.items():
#                 content = str(row.get(feature, '')).lower() # Récupérer le contenu, gérer les NaN, convertir en minuscules

#                 # Nettoyer les caractères spéciaux et les virgules pour les genres/acteurs
#                 content = re.sub(r'[^a-zA-Z0-9\s]', ' ', content)
#                 content = re.sub(r'\s+', ' ', content).strip() # Remplacer multiples espaces par un seul

#                 if content: # Ajouter le contenu seulement s'il n'est pas vide
#                     # Répéter le contenu selon son poids
#                     combined_text.extend([content] * int(weight))
#                     # Ajouter les décimales si nécessaire pour plus de finesse
#                     # Par exemple, si weight est 1.5, ajouter 0.5 fois (peut être complexe avec des mots)
#                     # Pour simplicité, on utilise int(weight) ici. Pour des poids non entiers, il faudrait
#                     # répéter les mots eux-mêmes, pas la chaîne entière.
#                     # Ex: ['mot1', 'mot2', 'mot1'] pour un poids de 1.5 de 'mot1' et 1 pour 'mot2'
#                     # Pour TF-IDF, la répétition de la chaîne est une approche simple et efficace.

#             weighted_texts.append(' '.join(combined_text))

#         return pd.Series(weighted_texts, index=X.index) # Retourner une Series pour TfidfVectorizer

print("Chargement des données...")
# Charger le dataset étendu
df = pd.read_csv(os.path.join(BASE_DIR, 'df_pour_ml.csv'))
df = df.fillna('') # Remplir les NaN avec des chaînes vides pour éviter les erreurs

print("Aperçu des données d'entraînement:\n", df[['originalTitle', 'overview_clean', 'directors_name', 'genres_list', 'actors_names']].head())

# --- 2. Définir les poids des features ---
# Vous pouvez ajuster ces valeurs en fonction de l'importance que vous donnez à chaque caractéristique.
# Par exemple, si le réalisateur est très important, donnez-lui un poids élevé.
FEATURE_WEIGHTS = {
    'overview_clean': 1.0,
    'directors_name': 3.0,
    'genres_list': 2.0,
    'actors_names': 1.5
}

# --- 3. Construction de la Pipeline ---
# Étape 1: Le Custom Transformer pour la pondération
feature_weighting_transformer = FeatureWeightingTransformer(feature_weights=FEATURE_WEIGHTS)

# Étape 2: TfidfVectorizer
# Transforme le texte pondéré en une matrice de caractéristiques TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=1, ngram_range=(1,2))

# Étape 3: NearestNeighbors (le "modèle" de recommandation)
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')

# Construire la pipeline
pipeline = Pipeline(steps=[
    ('feature_weighting', feature_weighting_transformer), # Première étape: pondération
    ('tfidf_vectorizer', tfidf_vectorizer),               # Deuxième étape: vectorisation TF-IDF
    ('nn_model', nn_model)                                # Dernière étape: recherche des plus proches voisins
])

print("\nEntraînement de la pipeline...")
# Entraîner la pipeline sur le DataFrame complet.
# Le FeatureWeightingTransformer prend le DataFrame, transforme les features pondérées,
# puis TfidfVectorizer transforme le texte pondéré,
# puis NearestNeighbors est fit sur ces vecteurs.
pipeline.fit(df)
print("Pipeline entraînée avec succès.")

# --- 4. Sauvegarde de la Pipeline et du DataFrame des films ---
# Sauvegarder toute la pipeline
joblib.dump(
    pipeline,
    os.path.join(BASE_DIR, 'movie_recommender_pipeline_weighted.pkl')
)
print("Pipeline sauvegardée sous 'movie_recommender_pipeline_weighted.pkl'")

# Sauvegarder le DataFrame original des films car le modèle retourne des indices
# et nous avons besoin de ce DF pour mapper les indices aux titres/infos des films.
df.to_csv(
    os.path.join(BASE_DIR, 'movies_data_for_app_weighted.csv'),
    index=False
)


# ============================================================
# 5. Entraînement TF-IDF pour Group To Film
# ============================================================

print("\nEntraînement du modèle TF-IDF pour Group To Film...")

df_group_path = os.path.join(BASE_DIR, "df_clean.csv")

if not os.path.exists(df_group_path):
    raise FileNotFoundError(
        f"Le fichier nécessaire à Group To Film est introuvable : "
        f"{df_group_path}"
    )

df_group = pd.read_csv(
    df_group_path,
    low_memory=False
)

# Vérification des colonnes indispensables
required_group_columns = [
    "directors_text",
    "actors_text",
    "genres_text",
]

missing_columns = [
    column
    for column in required_group_columns
    if column not in df_group.columns
]

if missing_columns:
    raise ValueError(
        "Colonnes manquantes dans df_clean.csv pour Group To Film : "
        + ", ".join(missing_columns)
    )

# Nettoyage des colonnes
for column in required_group_columns:
    df_group[column] = (
        df_group[column]
        .fillna("")
        .astype(str)
        .str.lower()
    )

# Le texte est construit dans le même ordre que df_clean.csv.
# Cela garantit que les indices de tfidf_matrix correspondent
# exactement aux lignes de df_group dans app.py.
df_group["group_features"] = (
    df_group["genres_text"]
    + " "
    + df_group["actors_text"]
    + " "
    + df_group["directors_text"]
)

# Création et entraînement du vectoriseur
tfidf_pipeline = TfidfVectorizer(
    stop_words="english",
    min_df=1,
    ngram_range=(1, 2),
)

tfidf_matrix = tfidf_pipeline.fit_transform(
    df_group["group_features"]
)

# Vérifications de sécurité
if tfidf_matrix.shape[0] != len(df_group):
    raise ValueError(
        "Le nombre de lignes de la matrice TF-IDF ne correspond pas "
        "au nombre de films dans df_clean.csv."
    )

if tfidf_matrix.shape[1] == 0:
    raise ValueError(
        "La matrice TF-IDF ne contient aucune caractéristique."
    )

# Sauvegarde atomique du pipeline TF-IDF
tfidf_pipeline_path = os.path.join(
    BASE_DIR,
    "tfidf_pipeline.joblib"
)

tfidf_pipeline_tmp = tfidf_pipeline_path + ".tmp"

joblib.dump(
    tfidf_pipeline,
    tfidf_pipeline_tmp
)

os.replace(
    tfidf_pipeline_tmp,
    tfidf_pipeline_path
)

# Sauvegarde atomique de la matrice TF-IDF
tfidf_matrix_path = os.path.join(
    BASE_DIR,
    "tfidf_matrix.joblib"
)

tfidf_matrix_tmp = tfidf_matrix_path + ".tmp"

joblib.dump(
    tfidf_matrix,
    tfidf_matrix_tmp
)

os.replace(
    tfidf_matrix_tmp,
    tfidf_matrix_path
)

print(
    "Modèle Group To Film entraîné :",
    tfidf_matrix.shape[0],
    "films et",
    tfidf_matrix.shape[1],
    "caractéristiques."
)

print(
    "Pipeline TF-IDF sauvegardée sous :",
    tfidf_pipeline_path
)

print(
    "Matrice TF-IDF sauvegardée sous :",
    tfidf_matrix_path
)




print("Données des films sauvegardées sous 'movies_data_for_app_weighted.csv'")

print("\nProcessus d'entraînement et de sauvegarde terminé avec pondération des features.")
print("Vous pouvez maintenant utiliser 'movie_recommender_pipeline_weighted.pkl' et 'movies_data_for_app_weighted.csv' dans votre application web.")

'''
Explications des modifications :

    FeatureWeightingTransformer :
        C'est une classe Python qui hérite de BaseEstimator et TransformerMixin de scikit-learn. Cela la rend compatible avec les Pipeline.
        Son constructeur __init__ prend un dictionnaire feature_weights où vous définissez l'importance de chaque feature (overview_clean, directors_name, genres_list, actors_names).
        La méthode fit est vide car ce transformer n'apprend rien des données.
        La méthode transform(X) est le cœur : elle prend un DataFrame X, itère sur chaque ligne, et pour chaque feature spécifiée dans feature_weights, elle répète le contenu textuel int(weight) fois, puis concatène tout en une seule chaîne. Cette chaîne pondérée est ensuite retournée sous forme de pd.Series.
        J'ai ajouté un nettoyage simple (re.sub) pour les virgules et les multiples espaces, ce qui est souvent utile pour les listes de genres_list ou d'acteurs.
    pipeline.fit(df) :
        Contrairement à avant où on passait seulement df['features'], maintenant on passe le DataFrame df entier à pipeline.fit(). C'est parce que FeatureWeightingTransformer a besoin d'accéder à plusieurs colonnes (overview_clean, directors_name, etc.) pour faire son travail.
        Le FeatureWeightingTransformer prendra df, produira une Series de textes pondérés, et cette Series sera ensuite passée au TfidfVectorizer pour vectorisation.
    Fichiers sauvegardés : J'ai renommé les fichiers .pkl et .csv pour indiquer qu'ils incluent la pondération (_weighted).
'''
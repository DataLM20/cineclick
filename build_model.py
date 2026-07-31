import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import os


# Chargement et préparation des données
df = pd.read_csv("df_pour_ml.csv")
df['directors_text'] = df['directors_text'].fillna('').str.lower()
df['actors_text'] = df['actors_text'].fillna('').str.lower()
df['genres_text'] = df['genres_text'].fillna('').str.lower()
df['combined_text'] = df['genres_text'] + " " + df['actors_text'] + " " + df['directors_text']

# Construction du pipeline TF-IDF
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer())
])

# Entraînement
tfidf_matrix = pipeline.fit_transform(df['combined_text'])

# Sauvegarde
joblib.dump(pipeline, "tfidf_pipeline.joblib")
joblib.dump(tfidf_matrix, "tfidf_matrix.joblib")
df.to_csv("df_clean.csv", index=False)

print("Pipeline et matrice TF-IDF enregistrés.")
from flask import Flask, render_template, request, jsonify
import requests
import random 
import pandas as pd
import joblib

from collections import Counter
import ast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from custom_transformers import FeatureWeightingTransformer

import os
from dotenv import load_dotenv
import google.generativeai as genai


def chercher_video_youtube(query, api_key):
    params = {
        "part": "snippet",
        "q": f"{query} bande annonce",
        "type": "video",
        "maxResults": 1,
        "key": api_key
    }
    r = requests.get("https://www.googleapis.com/youtube/v3/search", params=params)
    items = r.json().get("items", [])
    if items:
        return f"https://www.youtube.com/embed/{items[0]['id']['videoId']}"
    return None

load_dotenv()
app = Flask(__name__)

try:
    recommender_pipeline = joblib.load('movie_recommender_pipeline_weighted.pkl')
    movies_df = pd.read_csv('movies_data_for_app_weighted.csv')
    df = pd.read_csv("df_pour_ml.csv")
    print("Modèle de recommandation pondéré et données des films chargés avec succès.")

except FileNotFoundError:
    print("ERREUR: Le fichier 'movie_recommender_pipeline_weighted.pkl' ou 'movies_data_for_app_weighted.csv' n'a pas été trouvé.")
    print("Assurez-vous d'avoir exécuté 'train_and_save_model.py' au préalable.")
    recommender_pipeline = None
    movies_df = None



@app.route('/')
def home():
    return render_template("index.html")



@app.route("/film/<tconst>")
def film_infos(tconst):
    # On récupère les paramètres de l'URL pour savoir d'où vient l'utilisateur
    mood_origine = request.args.get('mood')
    page_origine = request.args.get('page')

    film_data = df[df["tconst"] == tconst]
    if film_data.empty:
        return "Film non trouvé", 404

    row = film_data.iloc[0]

    video_url = chercher_video_youtube(row["originalTitle"], api_key=os.getenv('YOUTUBE_API_KEY'))

    film = {
        "tconst": row["tconst"],
        "originalTitle": row["originalTitle"],
        "overview_text": row["overview_text"],
        "runtimeMinutes": row["runtimeMinutes"],
        "genres_list": row["genres_text"],
        "directors_name": row["directors_name"],
        "actors_names": row["actors_names"],
        "poster_path": row["poster_path"],
        "backdrop_path": row["backdrop_path"],
        "poster_url": f"https://image.tmdb.org/t/p/w500{row['poster_path']}" if pd.notna(row["poster_path"]) else "",
        "backdrop_url": f"https://image.tmdb.org/t/p/w780{row['backdrop_path']}" if pd.notna(row["backdrop_path"]) else "",
        "video_url": video_url,
        "startYear": row["startYear"],
        "averageRating": row["averageRating"],
        "note": row["averageRating"]*100,
        "numVotes": row["numVotes"],
        "overview_fr": row["overview_fr"],
        "title_fr": row["title_fr"]
    }

    return render_template("infos.html", film=film, mood_origine=mood_origine, page_origine=page_origine)

@app.route('/recherche', methods=['GET', 'POST'])
def recherche():
    query = request.form.get('query', '').strip()
    film = None
    if query:
        filt = df['primaryTitle'].str.contains(query, case=False, na=False)
        if filt.any():
            film_row = df.loc[filt].iloc[0]
            film = film_row.to_dict()
        print(film)
    # Préparer aussi les films par genre à afficher (facultatif ici, selon ta template)
    genres = ['Action', 'Thriller', 'Comedy', 'Fantasy', 'Drama', 'Romance']
    films_by_genre = {}
    for genre in genres:
        filt = df['genres_text'].str.contains(genre, case=False, na=False)
        films_by_genre[genre] = df.loc[filt].sample(30).to_dict(orient='records')

    return render_template('recherche.html', film=film, films_by_genre=films_by_genre, genres=genres, query=query)

@app.route('/recherche/genre/<genre>')
def films_par_genre(genre):
    filt = df['genres_text'].str.contains(genre, case=False, na=False)
    films = df.loc[filt].head(20).to_dict(orient='records')
    return render_template('recherche_genre.html', films=films, genre=genre)


# mood to film 

print("Chargement et préparation du DataFrame...")

def convertir_en_objet_python(text):
    # Fonction de sécurité pour lire les listes/dictionnaires depuis le CSV
    if isinstance(text, str):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
    return text

convertisseurs = {
    'genres_list': convertir_en_objet_python,
    'moods': convertir_en_objet_python,
    'mood_scores': convertir_en_objet_python
}

df_mood_to_film = pd.read_csv("df_mood_to_film.csv", converters=convertisseurs)

print(f"DataFrame chargé avec {len(df_mood_to_film)} films.")

# FONCTION DE RECOMMANDATION DE FILMS
def recommander_films_humeur(humeur_choisie, dataframe_films, page=1, recos_par_page=18):
    """
    Recommande des films en utilisant le score de pertinence et de qualité.
    """
    # Sécurité pour s'assurer que les colonnes nécessaires existent
    required_cols = ['moods', 'mood_scores', 'score']
    if not all(col in dataframe_films.columns for col in required_cols):
        print(f"ERREUR: Le DataFrame doit contenir les colonnes {required_cols}")
        return pd.DataFrame()

    condition_mood = dataframe_films['moods'].apply(lambda mood_list: humeur_choisie in mood_list)
    df_filtre = dataframe_films[condition_mood].copy()

    if len(df_filtre) < 2:
        return df_filtre

    df_filtre['pertinence_mood'] = df_filtre['mood_scores'].apply(lambda scores_dict: scores_dict.get(humeur_choisie, 0))
    
    scaler = MinMaxScaler()
    df_filtre[['score_normalise']] = scaler.fit_transform(df_filtre[['score']])
    df_filtre[['pertinence_normalise']] = scaler.fit_transform(df_filtre[['pertinence_mood']])
    df_filtre['score_reco_final'] = (0.5 * df_filtre['pertinence_normalise'] + 0.5 * df_filtre['score_normalise'])
    
    df_tries = df_filtre.sort_values(by='score_reco_final', ascending=False)
    
    start_index = (page - 1) * recos_par_page
    end_index = start_index + recos_par_page
    return df_tries.iloc[start_index:end_index]

@app.route("/moodtofilm")
def moodtofilm():
    # On vérifie si l'utilisateur a cliqué sur un mood
    if 'mood' in request.args:
        # --- SCÉNARIO 2 : AFFICHER LES RÉSULTATS ---
        humeur_choisie = request.args.get('mood').capitalize()
        
        # NOUVEAU : On récupère le numéro de page depuis l'URL, avec 1 comme valeur par défaut
        page_actuelle = request.args.get('page', 1, type=int)
        
        recommandations_df = recommander_films_humeur(
            humeur_choisie=humeur_choisie,
            dataframe_films=df_mood_to_film,
            page=page_actuelle # On passe le numéro de page à la fonction
        )
        
        films_a_afficher = recommandations_df.to_dict('records')
        
        # On renvoie des informations supplémentaires au HTML pour gérer la pagination
        return render_template(
            'moodtofilm.html', 
            mood_choisi=humeur_choisie, 
            films=films_a_afficher,
            page_actuelle=page_actuelle,
            recos_par_page=18 # On envoie le nombre de recos par page pour la logique d'affichage du bouton "suivant"
        )
    else:
        # --- SCÉNARIO 1 : AFFICHER LA PAGE DE CHOIX ---
        return render_template("moodtofilm.html")





# --- Chargement des objets pour group to group ---
df_group = pd.read_csv("df_clean.csv")
pipeline = joblib.load("tfidf_pipeline.joblib")
tfidf_matrix = joblib.load("tfidf_matrix.joblib")

# --- Nettoyage des colonnes textes ---
df_group['directors_text'] = df_group['directors_text'].fillna('')
df_group['actors_text'] = df_group['actors_text'].fillna('')
df_group['genres_text'] = df_group['genres_text'].fillna('')

# --- Comptage des genres à partir de la colonne genres_list ---
df_group['genres_list'] = df_group['genres_list'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
tous_les_genres = [genre for liste in df_group['genres_list'] for genre in liste]
genre_counts = Counter(tous_les_genres)
genre_count = genre_counts.most_common()  # Liste triée du plus fréquent au moins fréquent

# --- Préparation des listes pour Select2 ---
directors_list = sorted(df_group['directors_text'].str.split(', ').explode().dropna().str.title().unique())
actors_list = sorted(df_group['actors_text'].str.split(', ').explode().dropna().str.title().unique())
genres_list = sorted(set(tous_les_genres))  # Basé sur genres_list, plus fiable

# group to film
@app.route('/grouptofilm', methods=["GET", "POST"])
def grouptofilm():
    suggestions = []
    if request.method == "POST":
        genres_input = request.form.getlist("genres")
        actors_input = request.form.getlist("actors")
        directors_input = request.form.getlist("directors")

        # Texte combiné utilisateur
        user_text = " ".join([g.lower() for g in genres_input + actors_input + directors_input])
        user_vec = pipeline.transform([user_text])
        similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

        df_group['similarity'] = similarities
        top_films = df_group.sort_values(by="similarity", ascending=False).head(10)

        suggestions = [
            {
                'title': row['originalTitle'],
                'tconst': row['tconst'],
                'poster': 'https://image.tmdb.org/t/p/w500' + str(row["poster_path"])
            }
            for _, row in top_films.iterrows()
        ]

    return render_template(
        "grouptofilm.html",
        suggestions=suggestions,
        genres_list=genres_list,
        actors_list=actors_list,
        directors_list=directors_list,
        genre_count=genre_count  # (optionnel si tu veux l'afficher dans le template)
    )




@app.route('/autocomplete')
def autocomplete():
    global movies_df # Assurez-vous que movies_df est accessible globalement
    print(movies_df.columns)
    if movies_df is None:
        return jsonify([]) # Retourne une liste vide si les données ne sont pas chargées

    search_query = request.args.get('query', '').strip().lower()

    if not search_query:
        return jsonify([]) # Retourne une liste vide si la requête est vide

    # Filtrer les titres de films qui contiennent la chaîne de recherche (insensible à la casse)
    # Utilisez .str.contains() pour une correspondance partielle
    # .unique() pour éviter les doublons si votre dataset en a
    # .tolist() pour convertir le Series pandas en liste Python
    matching_titles = movies_df[
        movies_df['title_fr'].str.lower().str.contains(search_query, na=False)
    ]['title_fr'].unique().tolist()

    # Trier les résultats et limiter le nombre de suggestions
    # Un tri simple alphabétique ou par pertinence (si vous aviez un score)
    matching_titles.sort() 

    # Limiter à un certain nombre de suggestions pour ne pas surcharger le front-end
    return jsonify(matching_titles[:10]) # Limite à 10 suggestions

# --- 2. Fonction de recommandation (utilise le modèle chargé) ---
def get_movie_recommendations(movie_title, num_recommendations=5):
    if recommender_pipeline is None or movies_df is None:
        return ["Erreur: Le modèle n'est pas chargé ou les données des films sont manquantes."]

    # Trouver l'index du film entré par l'utilisateur
    matching_movies = movies_df[movies_df['title_fr'].str.lower() == movie_title.lower()]

    if matching_movies.empty:
        return [f"Désolé, le film '{movie_title}' n'a pas été trouvé dans notre base de données. Veuillez vérifier l'orthographe ou essayer un autre film."]

    # Prenez le premier film correspondant si plusieurs ont le même titre
    movie_index = matching_movies.index[0]

    # Récupérer la ligne complète du film d'entrée sous forme de DataFrame (important pour le FeatureWeightingTransformer)
    input_movie_data = movies_df.loc[[movie_index]] # Utilisez [[index]] pour retourner un DataFrame

    # Accéder aux étapes de la pipeline
    feature_weighting_transformer = recommender_pipeline.named_steps['feature_weighting']
    tfidf_vectorizer = recommender_pipeline.named_steps['tfidf_vectorizer']
    nn_model = recommender_pipeline.named_steps['nn_model']

    # --- Processus de transformation pour le film d'entrée ---
    # 1. Appliquer le FeatureWeightingTransformer au film d'entrée
    weighted_text_for_input = feature_weighting_transformer.transform(input_movie_data)

    # 2. Appliquer le TfidfVectorizer au texte pondéré du film d'entrée
    # Le vectorizer utilise son vocabulaire appris lors de l'entraînement
    input_movie_vector = tfidf_vectorizer.transform(weighted_text_for_input)

    # 3. Utiliser le modèle NearestNeighbors pour trouver les voisins les plus proches
    # Le premier voisin sera le film lui-même, donc on demande num_recommendations + 1
    distances, indices = nn_model.kneighbors(input_movie_vector, n_neighbors=num_recommendations + 1)

    # Récupérer les titres des films recommandés (ignorer le premier, qui est le film d'entrée lui-même)
    recommended_movie_indices = indices.flatten()[1:]

    recommendations = []

    for i, idx in enumerate(recommended_movie_indices):
        title = movies_df.loc[idx, ['originalTitle']]['originalTitle']
        tconst = movies_df.loc[idx, ['tconst']]['tconst']
        poster_path = movies_df.loc[idx, ['poster_path']]['poster_path']
        # La distance cosinus va de 0 (identique) à 2 (opposé), ou 0 à 1 pour certaines impl.
        # Ici, avec NearestNeighbors et metric='cosine', 0 est le plus proche.
        # On peut la convertir en similarité: 1 - distance (si distance est entre 0 et 1)
        # Ou simplement afficher la distance comme une mesure de dissimilarité.
        # Pour une similarité: si distances[0][i+1] est la distance brute
        # La similarité cosinus est (1 - distance) pour les métriques où 0 est parfait, 1 est complètement différent.
        # Les valeurs de `distances` pour `metric='cosine'` sont des distances, pas des similarités.
        # Une distance de 0 signifie que les vecteurs sont identiques, 1 signifie qu'ils sont orthogonaux (pas liés)
        # et 2 signifie qu'ils sont opposés.
        # On va afficher la distance pour montrer comment ça varie.
        dist = distances.flatten()[i+1] # +1 car on a ignoré le premier indice/distance
        recommendations.append({'title':title,
                                'tconst':tconst,
                                'poster_path':'https://image.tmdb.org/t/p/w500'+poster_path
                                }) # Plus le nombre est petit, plus c'est similaire
    return recommendations

# filmtofilm 
@app.route("/filmtofilm", methods=['GET', 'POST'])
def filmtofilm():
    recommendations = []
    movie_title_input = ""

    if request.method == 'POST':
        movie_title_input = request.form['movie_title']
        recommendations = get_movie_recommendations(movie_title_input)

    return render_template('filmtofilm.html', recommendations=recommendations, movie_title_input=movie_title_input)


# Configuration Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel("gemini-1.5-flash")
gemini_chat = model.start_chat(history=[])

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return render_template("chat.html")
    else:  # POST
        try:
            user_input = request.json["message"]
            response = gemini_chat.send_message(user_input)
            return jsonify({"reply": response.text})
        except Exception as e:
            print(f"Erreur : {e}")
            return jsonify({"reply": "Désolé, une erreur est survenue."}), 500


if __name__ == "__main__":
    app.run(debug=True)
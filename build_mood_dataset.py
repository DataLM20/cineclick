from pathlib import Path
import ast
import re
import shutil
import unicodedata

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "df_clean.csv"
OUTPUT_FILE = BASE_DIR / "df_mood_to_film.csv"
BACKUP_FILE = BASE_DIR / "df_mood_to_film_backup.csv"


# ------------------------------------------------------------------
# 1. Règles utilisées pour associer les films aux différentes humeurs
# ------------------------------------------------------------------

MOOD_RULES = {
    "Joyeux": {
        "genres": [
            "comedy",
            "animation",
            "family",
            "music",
            "musical"
        ],
        "keywords": [
            "fun",
            "funny",
            "happy",
            "joy",
            "celebration",
            "friendship",
            "holiday",
            "wedding",
            "party",
            "humour",
            "humor",
            "heureux",
            "joie",
            "amitié",
            "fête",
            "mariage",
            "vacances"
        ]
    },

    "Triste": {
        "genres": [
            "drama",
            "war"
        ],
        "keywords": [
            "death",
            "grief",
            "loss",
            "sad",
            "tragedy",
            "mourning",
            "loneliness",
            "disease",
            "separation",
            "deuil",
            "mort",
            "perte",
            "triste",
            "tragédie",
            "solitude",
            "maladie",
            "séparation"
        ]
    },

    "Romantique": {
        "genres": [
            "romance"
        ],
        "keywords": [
            "love",
            "romance",
            "relationship",
            "couple",
            "wedding",
            "passion",
            "lover",
            "amour",
            "romantique",
            "relation",
            "mariage",
            "passion",
            "couple"
        ]
    },

    "Effrayant": {
        "genres": [
            "horror",
            "thriller",
            "mystery"
        ],
        "keywords": [
            "ghost",
            "demon",
            "monster",
            "murder",
            "killer",
            "haunted",
            "terror",
            "nightmare",
            "zombie",
            "vampire",
            "fantôme",
            "démon",
            "monstre",
            "meurtre",
            "tueur",
            "terreur",
            "cauchemar"
        ]
    },

    "Énergique": {
        "genres": [
            "action",
            "adventure",
            "sport"
        ],
        "keywords": [
            "battle",
            "fight",
            "mission",
            "race",
            "warrior",
            "hero",
            "rescue",
            "explosion",
            "combat",
            "bataille",
            "mission",
            "course",
            "héros",
            "sauvetage"
        ]
    },

    "Inspirant": {
        "genres": [
            "biography",
            "documentary",
            "history",
            "sport"
        ],
        "keywords": [
            "dream",
            "success",
            "hope",
            "courage",
            "overcome",
            "achievement",
            "freedom",
            "justice",
            "destiny",
            "rêve",
            "réussite",
            "espoir",
            "courage",
            "liberté",
            "justice",
            "destin"
        ]
    },

    "Détente": {
        "genres": [
            "comedy",
            "family",
            "animation",
            "music"
        ],
        "keywords": [
            "summer",
            "holiday",
            "friendship",
            "family",
            "journey",
            "music",
            "beach",
            "vacances",
            "famille",
            "amitié",
            "voyage",
            "musique",
            "plage"
        ]
    },

    "Réflexion": {
        "genres": [
            "science fiction",
            "sci-fi",
            "mystery",
            "documentary",
            "history"
        ],
        "keywords": [
            "society",
            "humanity",
            "future",
            "identity",
            "memory",
            "existence",
            "philosophy",
            "politics",
            "science",
            "société",
            "humanité",
            "futur",
            "identité",
            "mémoire",
            "existence",
            "science"
        ]
    }
}


def normaliser_texte(value):
    """Convertit une valeur en texte normalisé sans accents."""

    if pd.isna(value):
        return ""

    texte = str(value).lower()

    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(
        caractere
        for caractere in texte
        if not unicodedata.combining(caractere)
    )

    texte = re.sub(r"[^a-z0-9\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()

    return texte


def convertir_liste(value):
    """Transforme genres_list en liste Python."""

    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    if isinstance(value, str):
        try:
            resultat = ast.literal_eval(value)

            if isinstance(resultat, list):
                return resultat
        except (ValueError, SyntaxError):
            pass

        return [
            element.strip()
            for element in value.split(",")
            if element.strip()
        ]

    return []


def calculer_humeurs(row):
    """
    Retourne :
    - la liste des humeurs du film ;
    - un dictionnaire contenant le score de chaque humeur.
    """

    genres = normaliser_texte(row.get("genres_text", ""))

    if not genres:
        genres_liste = convertir_liste(
            row.get("genres_list", [])
        )
        genres = normaliser_texte(
            " ".join(map(str, genres_liste))
        )

    texte = " ".join([
        normaliser_texte(row.get("overview_clean", "")),
        normaliser_texte(row.get("overview_text", "")),
        normaliser_texte(row.get("overview_fr", "")),
        normaliser_texte(row.get("primaryTitle", "")),
        normaliser_texte(row.get("originalTitle", "")),
        genres
    ])

    moods = []
    mood_scores = {}

    for mood, rules in MOOD_RULES.items():
        score = 0.0

        for genre in rules["genres"]:
            genre_normalise = normaliser_texte(genre)

            if genre_normalise and genre_normalise in genres:
                score += 3.0

        for keyword in rules["keywords"]:
            keyword_normalise = normaliser_texte(keyword)

            if keyword_normalise and keyword_normalise in texte:
                score += 1.0

        if score > 0:
            moods.append(mood)
            mood_scores[mood] = round(score, 3)

    if not moods:
        moods = ["Réflexion"]
        mood_scores = {"Réflexion": 0.25}

    return moods, mood_scores


def colonne_numerique(dataframe, noms_possibles, valeur_defaut=0):
    """Cherche la première colonne numérique disponible."""

    for colonne in noms_possibles:
        if colonne in dataframe.columns:
            return pd.to_numeric(
                dataframe[colonne],
                errors="coerce"
            ).fillna(valeur_defaut)

    return pd.Series(
        valeur_defaut,
        index=dataframe.index,
        dtype=float
    )


def normaliser_serie(serie):
    """Normalise une série entre 0 et 1."""

    serie = pd.to_numeric(
        serie,
        errors="coerce"
    ).fillna(0)

    if serie.nunique() <= 1:
        return pd.Series(
            0.0,
            index=serie.index
        )

    scaler = MinMaxScaler()

    return pd.Series(
        scaler.fit_transform(
            serie.to_numpy().reshape(-1, 1)
        ).flatten(),
        index=serie.index
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}"
        )

    print("Chargement de df_clean.csv...")

    df = pd.read_csv(INPUT_FILE)

    print("Dimensions initiales :", df.shape)

    if OUTPUT_FILE.exists():
        shutil.copy2(
            OUTPUT_FILE,
            BACKUP_FILE
        )

        print(
            " Ancien fichier sauvegardé :",
            BACKUP_FILE
        )

    print("Calcul des humeurs...")

    resultats = df.apply(
        calculer_humeurs,
        axis=1
    )

    df["moods"] = resultats.apply(
        lambda resultat: resultat[0]
    )

    df["mood_scores"] = resultats.apply(
        lambda resultat: resultat[1]
    )

    # --------------------------------------------------------------
    # Calcul du score général de qualité utilisé par ton application
    # --------------------------------------------------------------

    notes = colonne_numerique(
        df,
        [
            "averageRating",
            "vote_average",
            "rating"
        ]
    )

    votes = colonne_numerique(
        df,
        [
            "numVotes",
            "vote_count"
        ]
    )

    popularite = colonne_numerique(
        df,
        [
            "popularity"
        ]
    )

    # La notation est normalement comprise entre 0 et 10
    score_note = (notes.clip(lower=0, upper=10) / 10)


    score_votes = normaliser_serie(
        np.log1p(votes.clip(lower=0))
    )

    score_popularite = normaliser_serie(
        popularite.clip(lower=0)
    )

    df["score"] = (
        0.60 * score_note
        + 0.25 * score_votes
        + 0.15 * score_popularite
    ).round(6)

    df["moods"] = df["moods"].apply(repr)
    df["mood_scores"] = df["mood_scores"].apply(repr)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n Fichier créé :", OUTPUT_FILE)
    print("Dimensions finales :", df.shape)
    print("Nombre de films :", len(df))
    print("Moods manquants :", df["moods"].isna().sum())
    print("Scores manquants :", df["score"].isna().sum())

    print("\nRépartition des humeurs :")

    moods_exploses = (
        df["moods"]
        .apply(ast.literal_eval)
        .explode()
        .value_counts()
    )

    print(moods_exploses)


if __name__ == "__main__":
    main()
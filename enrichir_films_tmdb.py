import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = BASE_DIR / "films_tmdb_mai2025_aujourdhui.csv"
OUTPUT_FILE = DATA_DIR / "df_clean_nouveaux_films.csv"

# Fichier temporaire permettant de reprendre le traitement
# sans recommencer tous les appels à l'API.
CHECKPOINT_FILE = DATA_DIR / "films_tmdb_enrichis_checkpoint.jsonl"

BASE_URL = "https://api.themoviedb.org/3"

NB_ACTEURS = 3
PAUSE_ENTRE_REQUETES = 0.25
TIMEOUT = 30
MAX_RETRIES = 5

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    raise ValueError(
        "La variable TMDB_API_KEY n'est pas définie dans le fichier .env"
    )

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SESSION HTTP
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "Accept": "application/json",
        "User-Agent": "CineClickCatalogueUpdater/1.0",
    }
)


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def valeur_texte(valeur: Any) -> str:
    """Transforme une valeur potentiellement vide en chaîne propre."""
    if valeur is None:
        return ""

    if isinstance(valeur, float) and pd.isna(valeur):
        return ""

    return str(valeur).strip()


def nettoyer_texte(texte: Any) -> str:
    """
    Nettoie un texte pour le moteur de recommandation.

    Les accents sont conservés afin de rester cohérent avec
    l'ancien df_clean.csv.
    """
    texte = valeur_texte(texte).lower()

    # Supprime les apostrophes afin d'obtenir :
    # "mother's" -> "mothers"
    texte = re.sub(r"[’']", "", texte)

    # Conserve les caractères Unicode, les lettres et les chiffres.
    texte = re.sub(r"[^\w\s-]", " ", texte, flags=re.UNICODE)

    # Remplace tirets et underscores par des espaces.
    texte = texte.replace("-", " ").replace("_", " ")

    # Supprime les espaces répétés.
    texte = re.sub(r"\s+", " ", texte).strip()

    return texte


def liste_noms(elements: Any, cle: str = "name") -> list[str]:
    """Extrait une liste de valeurs depuis une liste de dictionnaires."""
    if not isinstance(elements, list):
        return []

    resultat = []

    for element in elements:
        if not isinstance(element, dict):
            continue

        valeur = valeur_texte(element.get(cle))

        if valeur:
            resultat.append(valeur)

    return resultat


def valeurs_uniques(valeurs: list[str]) -> list[str]:
    """Supprime les doublons tout en conservant l'ordre."""
    resultat = []
    valeurs_vues = set()

    for valeur in valeurs:
        if valeur and valeur not in valeurs_vues:
            resultat.append(valeur)
            valeurs_vues.add(valeur)

    return resultat


def choisir_premiere_valeur(*valeurs: Any) -> str:
    """Retourne la première valeur texte non vide."""
    for valeur in valeurs:
        texte = valeur_texte(valeur)

        if texte:
            return texte

    return ""


# ============================================================
# APPELS À TMDB
# ============================================================

def requete_tmdb(
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Exécute une requête TMDB avec plusieurs tentatives.

    Gère notamment :
    - les erreurs réseau ;
    - les limitations temporaires 429 ;
    - les erreurs serveur 5xx.
    """
    params = params.copy() if params else {}
    params["api_key"] = TMDB_API_KEY

    url = f"{BASE_URL}{endpoint}"

    for tentative in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                url,
                params=params,
                timeout=TIMEOUT,
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 404:
                print(f"    Film introuvable : {endpoint}")
                return None

            if response.status_code == 429:
                attente = int(response.headers.get("Retry-After", 2))
                attente = max(attente, tentative * 2)

                print(
                    f"    Limite TMDB atteinte. "
                    f"Nouvelle tentative dans {attente} secondes."
                )

                time.sleep(attente)
                continue

            if 500 <= response.status_code < 600:
                attente = tentative * 2

                print(
                    f"    Erreur serveur TMDB {response.status_code}. "
                    f"Nouvelle tentative dans {attente} secondes."
                )

                time.sleep(attente)
                continue

            print(
                f"    Erreur TMDB {response.status_code} pour {endpoint} : "
                f"{response.text[:300]}"
            )

            return None

        except requests.RequestException as erreur:
            attente = tentative * 2

            print(
                f"    Erreur réseau, tentative "
                f"{tentative}/{MAX_RETRIES} : {erreur}"
            )

            if tentative < MAX_RETRIES:
                time.sleep(attente)

    return None


def recuperer_details_film(
    movie_id: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    Récupère deux versions de la fiche :

    - anglais : données principales, crédits et identifiants externes ;
    - français : titre et résumé français.
    """
    details_en = requete_tmdb(
        f"/movie/{movie_id}",
        params={
            "language": "en-US",
            "append_to_response": "credits,external_ids",
        },
    )

    if details_en is None:
        return None, None

    time.sleep(PAUSE_ENTRE_REQUETES)

    details_fr = requete_tmdb(
        f"/movie/{movie_id}",
        params={
            "language": "fr-FR",
        },
    )

    return details_en, details_fr


# ============================================================
# TRANSFORMATION VERS LA STRUCTURE DE DF_CLEAN
# ============================================================

def extraire_realisateurs(details_en: dict[str, Any]) -> list[str]:
    """Extrait les réalisateurs depuis les crédits techniques."""
    credits = details_en.get("credits") or {}
    crew = credits.get("crew") or []

    realisateurs = []

    for personne in crew:
        if not isinstance(personne, dict):
            continue

        job = valeur_texte(personne.get("job")).lower()
        department = valeur_texte(
            personne.get("department")
        ).lower()

        if job == "director" or (
            department == "directing" and job == "director"
        ):
            nom = valeur_texte(personne.get("name"))

            if nom:
                realisateurs.append(nom)

    return valeurs_uniques(realisateurs)


def extraire_acteurs(
    details_en: dict[str, Any],
    nombre: int = NB_ACTEURS,
) -> list[str]:
    """
    Extrait les acteurs principaux selon l'ordre des crédits TMDB.
    """
    credits = details_en.get("credits") or {}
    cast = credits.get("cast") or []

    cast_trie = sorted(
        cast,
        key=lambda personne: personne.get("order", 999999),
    )

    acteurs = []

    for personne in cast_trie:
        nom = valeur_texte(personne.get("name"))

        if nom:
            acteurs.append(nom)

        if len(acteurs) >= nombre:
            break

    return valeurs_uniques(acteurs)


def construire_ligne_df_clean(
    source: dict[str, Any],
    details_en: dict[str, Any],
    details_fr: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Construit une ligne compatible avec la structure historique
    du fichier df_clean.csv.
    """
    details_fr = details_fr or {}

    external_ids = details_en.get("external_ids") or {}

    release_date = pd.to_datetime(
        choisir_premiere_valeur(
            details_en.get("release_date"),
            source.get("release_date"),
        ),
        errors="coerce",
    )

    if pd.isna(release_date):
        start_year = None
        decade = None
        release_date_csv = ""
    else:
        start_year = int(release_date.year)
        decade = (start_year // 10) * 10
        release_date_csv = release_date.strftime("%Y-%m-%d")

    runtime = details_en.get("runtime")

    if runtime is None or pd.isna(runtime):
        runtime = 0

    runtime = int(runtime)

    genres_list = liste_noms(
        details_en.get("genres") or []
    )

    production_countries = liste_noms(
        details_en.get("production_countries") or [],
        cle="iso_3166_1",
    )

    spoken_languages = liste_noms(
        details_en.get("spoken_languages") or [],
        cle="iso_639_1",
    )

    production_companies = (
        details_en.get("production_companies") or []
    )

    production_companies_name = liste_noms(
        production_companies
    )

    production_companies_country = liste_noms(
        production_companies,
        cle="origin_country",
    )

    directors = extraire_realisateurs(details_en)
    actors = extraire_acteurs(details_en)

    primary_title = choisir_premiere_valeur(
        details_en.get("title"),
        source.get("title"),
        details_en.get("original_title"),
    )

    original_title = choisir_premiere_valeur(
        details_en.get("original_title"),
        source.get("original_title"),
        primary_title,
    )

    title_fr = choisir_premiere_valeur(
        details_fr.get("title"),
        source.get("title"),
        primary_title,
    )

    overview_en = choisir_premiere_valeur(
        details_en.get("overview"),
        source.get("overview"),
    )

    overview_fr = choisir_premiere_valeur(
        details_fr.get("overview"),
        source.get("overview"),
        overview_en,
    )

    overview_clean = nettoyer_texte(overview_en)

    genres_text = nettoyer_texte(
        " ".join(genres_list)
    )

    directors_text = nettoyer_texte(
        ", ".join(directors)
    )

    actors_text = nettoyer_texte(
        ", ".join(actors)
    )

    overview_text = overview_clean

    combined_text = " ".join(
        partie
        for partie in [
            genres_text,
            actors_text,
            directors_text,
        ]
        if partie
    )

    combined_text = re.sub(
        r"\s+",
        " ",
        combined_text,
    ).strip()

    imdb_id = choisir_premiere_valeur(
        details_en.get("imdb_id"),
        external_ids.get("imdb_id"),
    )

    return {
        "id": int(details_en.get("id") or source["id"]),
        "tconst": imdb_id,
        "primaryTitle": primary_title,
        "originalTitle": original_title,
        "startYear": start_year,
        "runtimeMinutes": runtime,
        "backdrop_path": choisir_premiere_valeur(
            details_en.get("backdrop_path"),
            source.get("backdrop_path"),
        ),
        "budget": int(details_en.get("budget") or 0),
        "homepage": valeur_texte(details_en.get("homepage")),
        "original_language": choisir_premiere_valeur(
            details_en.get("original_language"),
            source.get("original_language"),
        ),
        "overview": overview_en,
        "popularity": float(
            details_en.get("popularity")
            or source.get("popularity")
            or 0
        ),
        "poster_path": choisir_premiere_valeur(
            details_en.get("poster_path"),
            source.get("poster_path"),
        ),
        "production_countries": production_countries,
        "release_date": release_date_csv,
        "revenue": int(details_en.get("revenue") or 0),
        "runtime": runtime,
        "spoken_languages": spoken_languages,
        "tagline": valeur_texte(details_en.get("tagline")),
        "production_companies_name": production_companies_name,
        "production_companies_country": (
            production_companies_country
        ),
        "averageRating": float(
            details_en.get("vote_average")
            or source.get("vote_average")
            or 0
        ),
        "numVotes": int(
            details_en.get("vote_count")
            or source.get("vote_count")
            or 0
        ),
        "directors_name": ", ".join(directors),
        "actors_names": ", ".join(actors),
        "genres_list": genres_list,
        "overview_clean": overview_clean,
        "decade": decade,
        "genres_text": genres_text,
        "directors_text": directors_text,
        "actors_text": actors_text,
        "overview_text": overview_text,
        "title_fr": title_fr,
        "overview_fr": overview_fr,
        "combined_text": combined_text,
    }


# ============================================================
# CHECKPOINT ET REPRISE DU TRAITEMENT
# ============================================================

def charger_checkpoint() -> list[dict[str, Any]]:
    """
    Charge les films déjà enrichis afin de reprendre après
    une interruption.
    """
    if not CHECKPOINT_FILE.exists():
        return []

    films_enrichis = []

    with CHECKPOINT_FILE.open(
        "r",
        encoding="utf-8",
    ) as fichier:
        for numero_ligne, ligne in enumerate(fichier, start=1):
            ligne = ligne.strip()

            if not ligne:
                continue

            try:
                films_enrichis.append(json.loads(ligne))

            except json.JSONDecodeError:
                print(
                    f"Ligne de checkpoint ignorée : {numero_ligne}"
                )

    return films_enrichis


def enregistrer_checkpoint(
    ligne_enrichie: dict[str, Any],
) -> None:
    """Ajoute immédiatement un film enrichi au checkpoint."""
    with CHECKPOINT_FILE.open(
        "a",
        encoding="utf-8",
    ) as fichier:
        fichier.write(
            json.dumps(
                ligne_enrichie,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier source introuvable : {INPUT_FILE.resolve()}"
        )

    df_source = pd.read_csv(INPUT_FILE)

    if "id" not in df_source.columns:
        raise ValueError(
            "La colonne 'id' est absente du fichier source."
        )

    df_source["id"] = pd.to_numeric(
        df_source["id"],
        errors="coerce",
    )

    df_source = df_source[
        df_source["id"].notna()
    ].copy()

    df_source["id"] = df_source["id"].astype(int)

    df_source = df_source.drop_duplicates(
        subset=["id"],
        keep="first",
    ).reset_index(drop=True)

    films_enrichis = charger_checkpoint()

    ids_deja_traites = {
        int(film["id"])
        for film in films_enrichis
        if film.get("id") is not None
    }

    total = len(df_source)

    print("=" * 70)
    print("ENRICHISSEMENT DU CATALOGUE CINECLICK")
    print("=" * 70)
    print(f"Fichier source      : {INPUT_FILE.resolve()}")
    print(f"Nombre de films     : {total}")
    print(f"Films déjà traités  : {len(ids_deja_traites)}")
    print(f"Films restant       : {total - len(ids_deja_traites)}")
    print("=" * 70)

    erreurs = []

    for index, source in df_source.iterrows():
        movie_id = int(source["id"])
        numero = index + 1

        if movie_id in ids_deja_traites:
            continue

        titre = valeur_texte(source.get("title"))

        print(
            f"[{numero}/{total}] "
            f"Enrichissement de {movie_id} — {titre}"
        )

        details_en, details_fr = recuperer_details_film(
            movie_id
        )

        if details_en is None:
            erreurs.append(
                {
                    "id": movie_id,
                    "title": titre,
                    "raison": "Détails TMDB indisponibles",
                }
            )

            print("    Échec : film ignoré.")
            continue

        try:
            ligne_enrichie = construire_ligne_df_clean(
                source=source.to_dict(),
                details_en=details_en,
                details_fr=details_fr,
            )

            films_enrichis.append(ligne_enrichie)
            ids_deja_traites.add(movie_id)

            enregistrer_checkpoint(ligne_enrichie)

            print(
                f"    OK — IMDb : "
                f"{ligne_enrichie['tconst'] or 'absent'} | "
                f"réalisateur : "
                f"{ligne_enrichie['directors_name'] or 'absent'}"
            )

        except Exception as erreur:
            erreurs.append(
                {
                    "id": movie_id,
                    "title": titre,
                    "raison": str(erreur),
                }
            )

            print(f"    Erreur de transformation : {erreur}")

        time.sleep(PAUSE_ENTRE_REQUETES)

    if not films_enrichis:
        raise RuntimeError(
            "Aucun film n'a pu être enrichi."
        )

    colonnes_df_clean = [
        "id",
        "tconst",
        "primaryTitle",
        "originalTitle",
        "startYear",
        "runtimeMinutes",
        "backdrop_path",
        "budget",
        "homepage",
        "original_language",
        "overview",
        "popularity",
        "poster_path",
        "production_countries",
        "release_date",
        "revenue",
        "runtime",
        "spoken_languages",
        "tagline",
        "production_companies_name",
        "production_companies_country",
        "averageRating",
        "numVotes",
        "directors_name",
        "actors_names",
        "genres_list",
        "overview_clean",
        "decade",
        "genres_text",
        "directors_text",
        "actors_text",
        "overview_text",
        "title_fr",
        "overview_fr",
        "combined_text",
    ]

    df_enrichi = pd.DataFrame(films_enrichis)

    df_enrichi = df_enrichi.reindex(
        columns=colonnes_df_clean
    )

    df_enrichi["release_date"] = pd.to_datetime(
        df_enrichi["release_date"],
        errors="coerce",
    )

    df_enrichi = df_enrichi.drop_duplicates(
        subset=["id"],
        keep="last",
    )

    df_enrichi = df_enrichi.sort_values(
        by=["release_date", "popularity"],
        ascending=[True, False],
    ).reset_index(drop=True)

    # Retour au format AAAA-MM-JJ pour le CSV.
    df_enrichi["release_date"] = (
        df_enrichi["release_date"]
        .dt.strftime("%Y-%m-%d")
        .fillna("")
    )

    df_enrichi.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("ENRICHISSEMENT TERMINÉ")
    print("=" * 70)
    print(f"Films enrichis      : {len(df_enrichi)}")
    print(f"Films en erreur     : {len(erreurs)}")
    print(f"Fichier créé        : {OUTPUT_FILE.resolve()}")
    print("=" * 70)

    if erreurs:
        erreurs_file = DATA_DIR / "films_tmdb_erreurs.csv"

        pd.DataFrame(erreurs).to_csv(
            erreurs_file,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"Liste des erreurs   : {erreurs_file.resolve()}"
        )


if __name__ == "__main__":
    main()
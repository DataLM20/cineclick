import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    raise ValueError(
        "La variable TMDB_API_KEY n'est pas définie dans le fichier .env"
    )

BASE_URL = "https://api.themoviedb.org/3"

START_DATE = "2025-05-01"
END_DATE = date.today().isoformat()

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Clé TMDB chargée correctement.")
print(f"Période : {START_DATE} → {END_DATE}")

def recuperer_films(start_date: str, end_date: str) -> list[dict]:
    films = []
    page = 1

    while True:
        print(f"Récupération de la page {page}...")

        response = requests.get(
            f"{BASE_URL}/discover/movie",
            params={
                "api_key": TMDB_API_KEY,
                "language": "fr-FR",
                "region": "FR",
                "include_adult": "false",
                "include_video": "false",
                "sort_by": "primary_release_date.asc",
                "primary_release_date.gte": start_date,
                "primary_release_date.lte": end_date,
                "vote_count.gte": 5,
                "page": page,
            },
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()

        films.extend(data.get("results", []))

        total_pages = min(data.get("total_pages", 1), 500)

        print(
            f"Page {page}/{total_pages} — "
            f"{len(films)} films récupérés"
        )

        if page >= total_pages:
            break

        page += 1
        time.sleep(0.2)

    return films


films = recuperer_films(
    start_date=START_DATE,
    end_date=END_DATE,
)

df_films = pd.DataFrame(films)

print()
print(f"Extraction terminée : {len(df_films)} films")
print(f"Films avant nettoyage : {len(df_films)}")

df_films["overview"] = (
    df_films["overview"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df_films["title"] = (
    df_films["title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df_films["vote_count"] = pd.to_numeric(
    df_films["vote_count"],
    errors="coerce"
).fillna(0)

df_films["release_date"] = pd.to_datetime(
    df_films["release_date"],
    errors="coerce"
)

df_films = df_films[
    (df_films["release_date"] >= pd.Timestamp("2025-05-01")) &
    (df_films["release_date"] <= pd.Timestamp.today())
]

df_films = df_films[
    df_films["id"].notna()
    & df_films["release_date"].notna()
    & (df_films["title"] != "")
    & (df_films["overview"] != "")
    & df_films["poster_path"].notna()
    & (df_films["vote_count"] >= 5)
].copy()

df_films = df_films.drop_duplicates(
    subset=["id"],
    keep="first"
)

df_films = df_films.sort_values(
    by=["release_date", "popularity"],
    ascending=[True, False]
).reset_index(drop=True)

print(f"Films après nettoyage : {len(df_films)}")
print("Date minimale :", df_films["release_date"].min())
print("Date maximale :", df_films["release_date"].max())

output_file = OUTPUT_DIR / "films_tmdb_mai2025_aujourdhui.csv"

df_films.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig",
)

print(f"Fichier créé : {output_file.resolve()}")
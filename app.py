import json
from flask import Flask, render_template, request, jsonify, redirect, session
import requests
import math
import random
import pandas as pd
import joblib
import hmac
import hashlib
import os
import ast
import re
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime, timezone
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.wrappers import Response
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from google.cloud import bigquery
from custom_transformers import FeatureWeightingTransformer
from prometheus_flask_exporter import PrometheusMetrics
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv("/home/izak/cineclick/.env")
MOVIEGLU_BASE_URL = os.getenv(
    "MOVIEGLU_BASE_URL",
    "https://api-gate2.movieglu.com/"
).rstrip("/")

MOVIEGLU_CLIENT = os.getenv("MOVIEGLU_CLIENT")
MOVIEGLU_API_KEY = os.getenv("MOVIEGLU_API_KEY")
MOVIEGLU_AUTHORIZATION = os.getenv("MOVIEGLU_AUTHORIZATION")
MOVIEGLU_TERRITORY = os.getenv("MOVIEGLU_TERRITORY", "FR")
MOVIEGLU_API_VERSION = os.getenv("MOVIEGLU_API_VERSION", "v201")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

app = Flask(__name__)
metrics = PrometheusMetrics(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_SAMESITE'] = "Lax"


app.wsgi_app = ProxyFix(
    app.wsgi_app,
    x_prefix=1,
    x_host=1
)


SECRET_KEY = os.getenv("TOKEN_SECRET_KEY")

def verify_token(user, token):
    expected = hmac.new(
        SECRET_KEY.encode(),
        user.encode(),
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected, token)



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

@app.before_request
def check_auth():
    if (
        request.path.startswith("/static")
        or request.path == "/login"
        or request.path == "/metrics"
    ):
        return
    if "user" not in session:
        return redirect("/login")

@app.route("/cineclick")
def cineclick():
    if "user" not in session:
        return redirect("/login")

    return render_template("cineclick.html", user=session["user"])

@app.route('/')
def home():
    if "user" not in session:
        return redirect("/login")

    return render_template("index.html", user=session.get("user"))

import pandas as pd
import os

@app.route("/nouveautes-cinema")
def nouveautes_cinema():
    if "user" not in session:
        return redirect("/login")

    films = []

    try:
        csv_path = "/home/izak/cineclick/data/nouveautes.csv"

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():

            id_film = str(int(row["movie_id"]))

            films.append({
                "tconst": id_film,
                "title": row["title"],
                "poster_path": row["poster_path"]
                    if pd.notna(row["poster_path"])
                    else "",
                "rating": float(row["vote_average"])
                    if pd.notna(row["vote_average"])
                    else 0.0
            })

    except Exception as e:
        print(f"Erreur lecture nouveautés.csv : {e}")
        films = []

    return render_template(
        "nouveautes_cinema.html",
        films=films
    )



@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = {
            "admin": "dataone26??",
            "demonstration": "Kianiro26??",
            "Audreycine": "Cine2026??liora"
        }

        if username in users and users[username] == password:
            session["user"] = username
            return redirect("/")
        else:
            return render_template("login.html", error=True)

    return render_template("login.html")

@app.before_request
def check_auth():
    if (
        request.path.startswith("/static")
        or request.path == "/login"
        or request.path == "/metrics"
    ):
        return

    if "user" not in session:
        return redirect("/login")


def calculate_distance_km(lat1, lon1, lat2, lon2):
    """
    Calcule la distance en kilomètres entre deux coordonnées GPS.
    """

    earth_radius = 6371

    lat1_rad = math.radians(float(lat1))
    lon1_rad = math.radians(float(lon1))
    lat2_rad = math.radians(float(lat2))
    lon2_rad = math.radians(float(lon2))

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return round(earth_radius * c, 2)


def detect_cinema_chain(tags):
    """
    Détecte la chaîne d'un cinéma à partir OpenStreetMap.
    """

    searchable_text = " ".join(
        [
            str(tags.get("name", "")),
            str(tags.get("brand", "")),
            str(tags.get("operator", "")),
            str(tags.get("branch", "")),
        ]
    ).lower()

    if "ugc" in searchable_text:
        return "UGC"

    if "kinepolis" in searchable_text or "kinépolis" in searchable_text:
        return "Kinepolis"

    if "pathé" in searchable_text or "pathe" in searchable_text:
        return "Pathé"

    if "gaumont" in searchable_text:
        return "Gaumont"

    return "Autre"


def get_cinema_coordinates(cinema):
    """
    Récupère les coordonnées en relation OpenStreetMap.
    """

    cinema_lat = cinema.get("lat")
    cinema_lon = cinema.get("lon")

    if cinema_lat is None or cinema_lon is None:
        center = cinema.get("center", {})
        cinema_lat = center.get("lat")
        cinema_lon = center.get("lon")

    if cinema_lat is None or cinema_lon is None:
        return None, None

    return float(cinema_lat), float(cinema_lon)


def normalize_cinema_name(value):
    if not value:
        return ""

    value = unicodedata.normalize("NFD", value)
    value = "".join(
        character
        for character in value
        if unicodedata.category(character) != "Mn"
    )

    value = value.lower()
    value = re.sub(r"\bcinema\b", "", value)
    value = re.sub(r"\bcine\b", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def match_movieglu_cinema(google_cinema, movieglu_cinemas):
    google_name = google_cinema.get("name", "")
    normalized_google_name = normalize_cinema_name(google_name)

    best_match = None
    best_score = 0

    for movieglu_cinema in movieglu_cinemas:
        movieglu_name = movieglu_cinema.get("name", "")
        normalized_movieglu_name = normalize_cinema_name(movieglu_name)

        if not normalized_google_name or not normalized_movieglu_name:
            continue

        score = SequenceMatcher(
            None,
            normalized_google_name,
            normalized_movieglu_name,
        ).ratio()

        if (
            normalized_google_name in normalized_movieglu_name
            or normalized_movieglu_name in normalized_google_name
        ):
            score += 0.25

        if score > best_score:
            best_score = score
            best_match = movieglu_cinema

    print(
        "Correspondance cinéma :",
        google_name,
        "→",
        best_match.get("name") if best_match else None,
        "score :",
        round(best_score, 2),
    )

    if best_score < 0.55:
        return None

    return best_match


def get_hybrid_cinemas(latitude, longitude, limit=10):
    """
    Google Places :
        vrais noms, adresses, téléphones et sites.

    MovieGlu Sandbox :
        identifiants utilisés pour les films et séances de démonstration.
    """

    google_cinemas = get_google_cinemas(
        latitude=latitude,
        longitude=longitude,
        limit=limit,
    )

    movieglu_cinemas = get_movieglu_cinemas(
        latitude=latitude,
        longitude=longitude,
        limit=limit,
    )

    cinemas = []

    for google_cinema in google_cinemas:
        movieglu_cinema = match_movieglu_cinema(
            google_cinema,
            movieglu_cinemas,
        )

        # Sans identifiant MovieGlu, les séances ne peuvent pas être chargées.
        movieglu_id = (
            movieglu_cinema.get("cinema_id")
            if movieglu_cinema
            else None
        )

        cinema = {
            "id": str(movieglu_id) if movieglu_id else google_cinema["google_place_id"],            "cinema_id": movieglu_id,

            # Informations réelles Google Places
            "name": google_cinema["name"],
            "chain": "Cinéma",
            "address": google_cinema["address"],
            "address2": None,
            "city": "",
            "postcode": "",
            "lat": google_cinema["lat"],
            "lon": google_cinema["lon"],
            "phone": google_cinema["phone"],
            "website": google_cinema["website"],
            "google_maps_url": google_cinema["google_maps_url"],

            # Données complémentaires
            "distance": (
                movieglu_cinema.get("distance")
                if movieglu_cinema
                else None
            ),
            "logo_url": None,
            "provider": "hybrid",
            "showtimes_available": movieglu_id is not None,
        }

        cinemas.append(cinema)

    return cinemas

def build_cinema_address(tags):
    """
    Construit une adresse lisible depuis les tags OpenStreetMap.
    """

    street_number = tags.get("addr:housenumber", "")
    street = tags.get("addr:street", "")
    postcode = tags.get("addr:postcode", "")
    city = tags.get("addr:city", "")

    street_part = " ".join(
        part for part in [street_number, street] if part
    )

    city_part = " ".join(
        part for part in [postcode, city] if part
    )

    address_parts = [
        part for part in [street_part, city_part] if part
    ]

    return ", ".join(address_parts) or "Adresse non renseignée"


def build_official_cinema_url(chain):
    """
    Retourne la page officielle générale de la chaîne.
    Une URL du cinéma est ajoutée avec les heures séances.
    """

    official_urls = {
        "UGC": "https://www.ugc.fr/",
        "Kinepolis": "https://kinepolis.fr/",
        "Pathé": "https://www.pathe.fr/",
        "Gaumont": "https://www.pathe.fr/",
    }

    return official_urls.get(chain)


def movieglu_coordinates(latitude, longitude):

    if MOVIEGLU_TERRITORY == "XX":
        return -22.0, 14.0

    return float(latitude), float(longitude)


def movieglu_headers(latitude, longitude):
    latitude, longitude = movieglu_coordinates(latitude, longitude)

    return {
        "client": MOVIEGLU_CLIENT,
        "x-api-key": MOVIEGLU_API_KEY,
        "authorization": MOVIEGLU_AUTHORIZATION,
        "territory": MOVIEGLU_TERRITORY,
        "api-version": MOVIEGLU_API_VERSION,
        "geolocation": f"{latitude};{longitude}",
        "device-datetime": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        ),
    }


def check_movieglu_configuration():
    required_values = {
        "MOVIEGLU_CLIENT": MOVIEGLU_CLIENT,
        "MOVIEGLU_API_KEY": MOVIEGLU_API_KEY,
        "MOVIEGLU_AUTHORIZATION": MOVIEGLU_AUTHORIZATION,
    }

    missing = [
        name
        for name, value in required_values.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Configuration MovieGlu manquante : "
            + ", ".join(missing)
        )


def get_google_cinemas(latitude, longitude, limit=10):
    """
    Recherche de vrais cinémas autour des coordonnées fournies
    avec Google Places API (New).
    """

    if not GOOGLE_PLACES_API_KEY:
        raise RuntimeError(
            "La variable GOOGLE_PLACES_API_KEY est absente du fichier .env"
        )

    url = "https://places.googleapis.com/v1/places:searchNearby"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": (
            "places.id,"
            "places.displayName,"
            "places.formattedAddress,"
            "places.location,"
            "places.nationalPhoneNumber,"
            "places.websiteUri,"
            "places.googleMapsUri"
        ),
    }

    payload = {
        "includedTypes": ["movie_theater"],
        "maxResultCount": min(limit, 20),
        "rankPreference": "DISTANCE",
        "languageCode": "fr",
        "regionCode": "FR",
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                },
                "radius": 15000.0,
            }
        },
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Google Places : HTTP {response.status_code} - "
            f"{response.text}"
        )

    data = response.json()
    cinemas = []

    for place in data.get("places", []):
        display_name = place.get("displayName", {})
        location = place.get("location", {})

        cinemas.append({
            "google_place_id": place.get("id"),
            "name": display_name.get("text", "Cinéma"),
            "address": place.get(
                "formattedAddress",
                "Adresse non disponible",
            ),
            "lat": location.get("latitude"),
            "lon": location.get("longitude"),
            "phone": place.get("nationalPhoneNumber"),
            "website": place.get("websiteUri"),
            "google_maps_url": place.get("googleMapsUri"),
            "provider": "google_places",
        })

    return cinemas

def get_movieglu_cinemas(latitude, longitude, limit=10):
    check_movieglu_configuration()

    url = f"{MOVIEGLU_BASE_URL}/cinemasNearby/"

    response = requests.get(
        url,
        headers=movieglu_headers(latitude, longitude),
        params={"n": limit},
        timeout=30,
    )

    if response.status_code == 204:
        return []

    if response.status_code != 200:
        message = response.headers.get("MG-message", response.text)

        raise RuntimeError(
            f"MovieGlu cinemasNearby : "
            f"HTTP {response.status_code} - {message}"
        )

    data = response.json()
    cinemas = []

    for cinema in data.get("cinemas", [])[:limit]:
        address = cinema.get("address")

        if isinstance(address, dict):
            address1 = (
                address.get("address1")
                or address.get("address")
                or ""
            )
            address2 = address.get("address2")
            city = address.get("city") or ""
            postcode = address.get("postcode") or ""
        else:
            address1 = address or ""
            address2 = None
            city = cinema.get("city") or ""
            postcode = cinema.get("postcode") or ""

        cinema_name = (
            cinema.get("cinema_name")
            or cinema.get("name")
            or "Cinéma"
        )

        cinema_latitude = (
            cinema.get("lat")
            or cinema.get("latitude")
        )

        cinema_longitude = (
            cinema.get("lng")
            or cinema.get("lon")
            or cinema.get("longitude")
        )

        cinemas.append({
            "id": str(cinema.get("cinema_id")),
            "cinema_id": cinema.get("cinema_id"),
            "name": cinema_name,
            "chain": (
                cinema.get("cinema_chain")
                or cinema.get("chain")
                or "MovieGlu"
            ),
            "lat": cinema_latitude,
            "lon": cinema_longitude,
            "distance": cinema.get("distance"),
            "address": address1,
            "address2": address2,
            "city": city,
            "postcode": postcode,
            "logo_url": cinema.get("logo_url"),
            "phone": (
                cinema.get("telephone")
                or cinema.get("phone")
            ),
            "website": cinema.get("website"),
            "provider": "movieglu",
        })

        print(
            "Cinéma MovieGlu reçu :",
            cinema.get("cinema_id"),
            cinema_name,
            cinema.get("distance"),
        )

    return cinemas

def get_movieglu_showtimes(
    cinema_id,
    latitude,
    longitude,
    show_date=None,
):
    check_movieglu_configuration()

    if show_date is None:
        show_date = datetime.now().strftime("%Y-%m-%d")

    url = f"{MOVIEGLU_BASE_URL}/cinemaShowTimes/"

    response = requests.get(
        url,
        headers=movieglu_headers(latitude, longitude),
        params={
            "cinema_id": cinema_id,
            "date": show_date,
            "sort": "popularity",
        },
        timeout=30,
    )

    if response.status_code == 204:
        return {
            "cinema": {},
            "movies": [],
        }

    if response.status_code != 200:
        message = response.headers.get("MG-message", response.text)

        raise RuntimeError(
            f"MovieGlu cinemaShowTimes : "
            f"HTTP {response.status_code} - {message}"
        )

    data = response.json()

    cinema_data = data.get("cinema", {})
    movies = []


    for film in data.get("films", []):
        if not isinstance(film, dict):
            continue

        showings = []
        raw_showings = film.get("showings") or {}

        # MovieGlu peut renvoyer showings comme dictionnaire
        # ou parfois comme liste selon la réponse.
        if isinstance(raw_showings, dict):
            showing_items = raw_showings.items()

        elif isinstance(raw_showings, list):
            showing_items = []

            for showing_item in raw_showings:
                if not isinstance(showing_item, dict):
                    continue

                format_name = (
                    showing_item.get("format")
                    or showing_item.get("type")
                    or showing_item.get("version_type")
                    or "Standard"
                )

                showing_items.append(
                    (format_name, showing_item)
                )

        else:
            showing_items = []

        for format_name, format_data in showing_items:
            times = []

            if isinstance(format_data, list):
                raw_times = format_data

            elif isinstance(format_data, dict):
                raw_times = format_data.get("times") or []

            else:
                raw_times = []

            if isinstance(raw_times, dict):
                raw_times = [raw_times]

            for session in raw_times:
                if not isinstance(session, dict):
                    continue

                start_time = (
                    session.get("start_time")
                    or session.get("time")
                )

                end_time = session.get("end_time")

                if start_time:
                    times.append({
                        "start_time": start_time,
                        "end_time": end_time,
                    })

            if times:
                showings.append({
                    "format": format_name,
                    "times": times,
                })

        poster_url = None

        images = film.get("images") or {}

        if isinstance(images, dict):
            poster = images.get("poster") or {}

            if isinstance(poster, dict):
                poster_size_1 = (
                    poster.get("1")
                    or poster.get(1)
                    or {}
                )

                if isinstance(poster_size_1, dict):
                    medium = poster_size_1.get("medium") or {}

                    if isinstance(medium, dict):
                        poster_url = medium.get("film_image")

                    elif isinstance(medium, str):
                        poster_url = medium

        rating = None
        age_advisory = None

        age_ratings = film.get("age_rating") or []

        if isinstance(age_ratings, dict):
            age_ratings = [age_ratings]

        if isinstance(age_ratings, list) and age_ratings:
            first_rating = age_ratings[0]

            if isinstance(first_rating, dict):
                rating = first_rating.get("rating")
                age_advisory = first_rating.get("age_advisory")

            elif isinstance(first_rating, str):
                rating = first_rating

        movies.append({
            "film_id": film.get("film_id"),
            "imdb_id": film.get("imdb_id"),
            "imdb_title_id": film.get("imdb_title_id"),
            "title": film.get("film_name"),
            "version_type": film.get("version_type"),
            "poster_url": poster_url,
            "rating": rating,
            "age_advisory": age_advisory,
            "showings": showings,
        })

    return {
        "cinema": {
            "cinema_id": cinema_data.get("cinema_id"),
            "name": cinema_data.get("cinema_name"),
        },
        "movies": movies,
    }


def get_overpass_cinemas(latitude, longitude, limit=10):
    overpass_urls = [
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
    ]

    query = f"""
    [out:json][timeout:15];
    nwr["amenity"="cinema"](
        around:5000,
        {latitude},
        {longitude}
    );
    out center {limit};
    """

    response = None
    last_error = None

    for overpass_url in overpass_urls:
        try:
            print("Test du serveur Overpass :", overpass_url)

            candidate_response = requests.post(
                overpass_url,
                data={"data": query},
                headers={"User-Agent": "CineClick/1.0"},
                timeout=35,
            )

            print(
                "Statut Overpass :",
                candidate_response.status_code,
            )

            content_type = candidate_response.headers.get(
                "content-type",
                ""
            )

            if (
                candidate_response.status_code == 200
                and "json" in content_type.lower()
            ):
                response = candidate_response
                break

            last_error = (
                f"{overpass_url} a répondu "
                f"{candidate_response.status_code}"
            )

        except requests.RequestException as exception:
            last_error = str(exception)
            print(
                "Serveur Overpass indisponible :",
                overpass_url,
                exception,
            )

    if response is None:
        raise RuntimeError(
            "Aucun serveur Overpass disponible. "
            f"Dernière erreur : {last_error}"
        )

    elements = response.json().get("elements", [])

    cinemas = []

    for element in elements:
        tags = element.get("tags", {})

        cinema_latitude = element.get("lat")
        cinema_longitude = element.get("lon")

        if cinema_latitude is None or cinema_longitude is None:
            center = element.get("center", {})
            cinema_latitude = center.get("lat")
            cinema_longitude = center.get("lon")

        if cinema_latitude is None or cinema_longitude is None:
            continue

        name = tags.get("name")

        if not name:
            continue

        distance = calculate_distance_km(
            latitude,
            longitude,
            float(cinema_latitude),
            float(cinema_longitude),
        )

        address_parts = [
            tags.get("addr:housenumber"),
            tags.get("addr:street"),
        ]

        address = " ".join(
            str(part)
            for part in address_parts
            if part
        )

        city = (
            tags.get("addr:city")
            or tags.get("contact:city")
            or ""
        )

        postcode = tags.get("addr:postcode", "")

        cinemas.append({
            "id": str(element.get("id")),
            "osm_id": str(element.get("id")),
            "cinema_id": None,
            "name": name,
            "chain": tags.get("brand") or tags.get("operator") or "Cinéma",
            "lat": float(cinema_latitude),
            "lon": float(cinema_longitude),
            "distance": round(distance, 1),
            "address": address or "Adresse non renseignée",
            "address2": None,
            "city": city,
            "postcode": postcode,
            "phone": (
                tags.get("phone")
                or tags.get("contact:phone")
            ),
            "website": (
                tags.get("website")
                or tags.get("contact:website")
            ),
            "provider": "openstreetmap",
        })

    cinemas.sort(
        key=lambda cinema: cinema.get("distance", float("inf"))
    )

    return cinemas[:limit]


@app.route("/geocodcine", methods=["GET", "POST"])
def geocodcine():
    if "user" not in session:
        return redirect("/login")

    cinemas = []
    ville = ""
    latitude = None
    longitude = None
    error = None

    if request.method == "POST":
        ville = request.form.get("ville", "").strip()

        if not ville:
            error = "Veuillez saisir une ville."

        else:
            try:
                geocoding_response = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={
                        "q": ville,
                        "format": "json",
                        "limit": 1,
                        "countrycodes": "fr",
                    },
                    headers={
                        "User-Agent": "CineClick/1.0"
                    },
                    timeout=20,
                )

                geocoding_response.raise_for_status()
                locations = geocoding_response.json()

                if not locations:
                    error = "Ville introuvable."

                else:
                    latitude = float(locations[0]["lat"])
                    longitude = float(locations[0]["lon"])

                    cinemas = get_hybrid_cinemas(
                        latitude=latitude,
                        longitude=longitude,
                        limit=10,
                    )

                    print(
                        f"Cinémas réels Google Places trouvés autour de "
                        f"{ville} : {len(cinemas)}",
                        flush=True,
                    )

            except Exception as exception:
                print(
                    f"Erreur geocodcine : {type(exception).__name__}: {exception}",
                    flush=True,
                )
                error = str(exception)

    return render_template(
        "geocodcine.html",
        cinemas=cinemas,
        ville=ville,
        latitude=latitude,
        longitude=longitude,
        error=error,
    )

@app.route("/api/cinemas/<cinema_id>/showtimes")
def cinema_showtimes(cinema_id):
    if "user" not in session:
        return jsonify({
            "success": False,
            "movies": [],
            "error": "Votre session a expiré.",
        }), 401

    latitude = request.args.get("lat", type=float)
    longitude = request.args.get("lon", type=float)
    show_date = request.args.get("date")

    if latitude is None or longitude is None:
        if MOVIEGLU_TERRITORY == "XX":
            latitude = -22.0
            longitude = 14.0
        else:
            return jsonify({
                "success": False,
                "movies": [],
                "error": "Coordonnées du cinéma manquantes.",
            }), 400

    try:
        result = get_movieglu_showtimes(
            cinema_id=cinema_id,
            latitude=latitude,
            longitude=longitude,
            show_date=show_date,
        )

        return jsonify({
            "success": True,
            "cinema": result["cinema"],
            "movies": result["movies"],
            "source": "MovieGlu",
            "territory": MOVIEGLU_TERRITORY,
        })

    except Exception as error:
        print(
            f"Erreur MovieGlu séances cinéma "
            f"{cinema_id} : {error}"
        )

        return jsonify({
            "success": False,
            "movies": [],
            "error": str(error),
        }), 502

@app.route("/film/<tconst>")
def film_infos(tconst):

    if tconst.isdigit():
        movie_id = int(tconst)

        url = f"https://api.themoviedb.org/3/movie/{movie_id}"

        params = {
            "api_key": os.getenv("TMDB_API_KEY"),
            "language": "fr-FR",
            "append_to_response": "credits,videos,external_ids"
        }

        try:
            response = requests.get(url, params=params, timeout=15)

            if response.status_code != 200:
                print("Erreur TMDB :", response.status_code, response.text)
                return "Film introuvable sur TMDB", 404

            data = response.json()

            credits = data.get("credits", {})

            realisateurs = [
                personne["name"]
                for personne in credits.get("crew", [])
                if personne.get("job") == "Director"
            ]

            acteurs = [
                personne["name"]
                for personne in credits.get("cast", [])[:10]
            ]

            videos = data.get("videos", {}).get("results", [])
            video_url = None

            for video in videos:
                if (
                    video.get("site") == "YouTube"
                    and video.get("type") == "Trailer"
                ):
                    video_url = (
                        f"https://www.youtube.com/embed/{video['key']}"
                    )
                    break

            vote_average = float(data.get("vote_average") or 0)

            film = {
                "tconst": (
                    data.get("external_ids", {}).get("imdb_id")
                    or str(movie_id)
                ),
                "movie_id": movie_id,
                "originalTitle": (
                    data.get("original_title")
                    or data.get("title")
                    or ""
                ),
                "title_fr": data.get("title") or "",
                "overview_text": data.get("overview") or "",
                "overview_fr": data.get("overview") or "",
                "runtimeMinutes": data.get("runtime") or 0,
                "genres_list": ", ".join(
                    genre["name"]
                    for genre in data.get("genres", [])
                ),
                "directors_name": ", ".join(realisateurs),
                "actors_names": ", ".join(acteurs),
                "poster_path": data.get("poster_path") or "",
                "backdrop_path": data.get("backdrop_path") or "",
                "poster_url": (
                    f"https://image.tmdb.org/t/p/w500"
                    f"{data['poster_path']}"
                    if data.get("poster_path")
                    else ""
                ),
                "backdrop_url": (
                    f"https://image.tmdb.org/t/p/w780"
                    f"{data['backdrop_path']}"
                    if data.get("backdrop_path")
                    else ""
                ),
                "video_url": video_url,
                "startYear": (
                    data.get("release_date", "")[:4]
                    if data.get("release_date")
                    else ""
                ),
                "averageRating": vote_average,
                "note": vote_average * 10,
                "numVotes": int(data.get("vote_count") or 0)
            }

            return render_template(
                "infos.html",
                film=film,
                mood_origine=None,
                page_origine=None
            )

        except Exception as e:
            print("Erreur récupération TMDB :", e)
            return "Erreur lors de la récupération du film sur TMDB", 500

    mood_origine = request.args.get('mood')
    page_origine = request.args.get('page')

    tconst_original = tconst

    if tconst.isdigit():
        if f"tt{tconst}" in df["tconst"].values:
            tconst = f"tt{tconst}"
        else:
            pass

    film_data = df[df["tconst"] == tconst]

    if not film_data.empty:
        row = film_data.iloc[0]

        if str(row.get("directors_name")) == "Non disponible" or pd.isna(row.get("directors_name")):
            film_data = pd.DataFrame()

    if not film_data.empty:
        # CAS 1 : Le film est présent localement ET complet 
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
            "note": row["averageRating"]*100 if pd.notna(row["averageRating"]) else 0,
            "numVotes": row["numVotes"],
            "overview_fr": row["overview_fr"],
            "title_fr": row["title_fr"]
        }
    else:
        # CAS 2 : Le film est une Nouveauté OU incomplet localement (Appel direct TMDB)
        try:
            tmdb_api_key = "dd2c821c7286a83f2222603ada4ddbbe"

            if str(tconst).startswith("tt"):
                url_tmdb = f"https://api.themoviedb.org/3/find/{tconst}?api_key={tmdb_api_key}&external_source=imdb_id&language=fr-FR&append_to_response=credits"
                res = requests.get(url_tmdb).json()
                movie_results = res.get("movie_results", [])
                movie_details = movie_results[0] if movie_results else None
            else:
                url_tmdb = f"https://api.themoviedb.org/3/movie/{tconst_original}?api_key={tmdb_api_key}&language=fr-FR&append_to_response=credits"
                response = requests.get(url_tmdb)
                movie_details = response.json() if response.status_code == 200 else None

            if not movie_details or "title" not in movie_details:
                return "Film introuvable sur le serveur local et sur TMDB", 404

            # Extraction dynamique des crédits (Acteurs & Réalisateurs)
            credits = movie_details.get("credits", {})
            cast = credits.get("cast", [])
            crew = credits.get("crew", [])

            directors = [member["name"] for member in crew if member.get("job") == "Director"]
            directors_str = ", ".join(directors) if directors else "Non disponible"

            actors = [actor["name"] for actor in cast[:5]]
            actors_str = ", ".join(actors) if actors else "Non disponible"

            video_url = chercher_video_youtube(movie_details["title"], api_key=os.getenv('YOUTUBE_API_KEY'))

            film = {
                "tconst": tconst,
                "originalTitle": movie_details.get("original_title", movie_details["title"]),
                "overview_text": movie_details.get("overview", ""),
                "runtimeMinutes": movie_details.get("runtime", 120),
                "genres_list": ", ".join([g["name"] for g in movie_details.get("genres", [])]) if "genres" in movie_details else "Non spécifié",
                "directors_name": directors_str,
                "actors_names": actors_str,
                "poster_path": movie_details.get("poster_path", ""),
                "backdrop_path": movie_details.get("backdrop_path", ""),
                "poster_url": f"https://image.tmdb.org/t/p/w500{movie_details['poster_path']}" if movie_details.get('poster_path') else "",
                "backdrop_url": f"https://image.tmdb.org/t/p/w780{movie_details['backdrop_path']}" if movie_details.get('backdrop_path') else "",
                "video_url": video_url,
                "startYear": movie_details.get("release_date", "2026")[:4],
                "averageRating": movie_details.get("vote_average", 0.0),
                "note": movie_details.get("vote_average", 0.0) * 10,
                "numVotes": movie_details.get("vote_count", 0),
                "overview_fr": movie_details.get("overview", ""),
                "title_fr": movie_details["title"]
            }
        except Exception as api_err:
            return f"Film non trouvé localement et échec de l'API externe : {api_err}", 404

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

    if 'mood' in request.args:

        humeur_choisie = request.args.get('mood').capitalize()

        page_actuelle = request.args.get('page', 1, type=int)

        recommandations_df = recommander_films_humeur(
            humeur_choisie=humeur_choisie,
            dataframe_films=df_mood_to_film,
            page=page_actuelle 
        )

        films_a_afficher = recommandations_df.to_dict('records')

        # On renvoie des informations supplémentaires au HTML
        return render_template(
            'moodtofilm.html',
            mood_choisi=humeur_choisie,
            films=films_a_afficher,
            page_actuelle=page_actuelle,
            recos_par_page=18 
        )
    else:
  
        return render_template("moodtofilm.html")


# Chargement des objets pour group to group 
df_group = pd.read_csv("df_clean.csv")
pipeline = joblib.load("tfidf_pipeline.joblib")
tfidf_matrix = joblib.load("tfidf_matrix.joblib")

# Nettoyage des colonnes textes 
df_group['directors_text'] = df_group['directors_text'].fillna('')
df_group['actors_text'] = df_group['actors_text'].fillna('')
df_group['genres_text'] = df_group['genres_text'].fillna('')

# Comptage des genres à partir de la colonne genres_list 
def safe_eval(x):
    if pd.isna(x):
        return []

    try:
        return ast.literal_eval(str(x))
    except Exception:
        return [g.strip() for g in str(x).split(",") if g.strip()]

df_group['genres_list'] = df_group['genres_list'].apply(safe_eval)
tous_les_genres = [genre for liste in df_group['genres_list'] for genre in liste]
genre_counts = Counter(tous_les_genres)
genre_count = genre_counts.most_common()  

# Préparation des listes pour Select2

directors_list = sorted(
    df_group["directors_text"]
    .fillna("")
    .str.split(r"\s*,\s*", regex=True)
    .explode()
    .str.strip()
    .loc[lambda x: x.ne("")]
    .drop_duplicates()
    .str.title()
    .tolist(),
    key=str.lower
)

actors_list = sorted(
    df_group["actors_names"]
    .fillna("")
    .str.split(r"\s*,\s*", regex=True)
    .explode()
    .str.strip()
    .loc[lambda x: x.ne("")]
    .drop_duplicates()
    .tolist(),
    key=str.lower
)

genres_list = sorted(set(tous_les_genres))

# group to film
@app.route('/grouptofilm', methods=["GET", "POST"])
def grouptofilm():
    suggestions = []
    if request.method == "POST":
        genres_input = request.form.getlist("genres")
        actors_input = request.form.getlist("actors")
        directors_input = request.form.getlist("directors")

       
        user_text = " ".join([g.lower() for g in genres_input + actors_input + directors_input])
        user_vec = pipeline.transform([user_text])
        similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

        df_group['similarity'] = similarities
        top_films = df_group.sort_values(by="similarity", ascending=False).head(21)

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
        genre_count=genre_count  
    )

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route('/autocomplete')
def autocomplete():
    global movies_df 
    print(movies_df.columns)
    if movies_df is None:
        return jsonify([]) 

    search_query = request.args.get('query', '').strip().lower()

    if not search_query:
        return jsonify([]) 


    matching_titles = movies_df[
        movies_df['title_fr'].str.lower().str.contains(search_query, na=False)
    ]['title_fr'].unique().tolist()

  
    matching_titles.sort()

    return jsonify(matching_titles[:10]) # Limite à 10 suggestions

# 2. Fonction de recommandation (utilise le modèle chargé)


def get_movie_recommendations(movie_title, num_recommendations=5):
    if recommender_pipeline is None or movies_df is None:
        return ["Erreur: Le modèle n'est pas chargé ou les données des films sont manquantes."]

    matching_movies = movies_df[movies_df['title_fr'].str.lower() == movie_title.lower()]

    if matching_movies.empty:
        return [f"Désolé, le film '{movie_title}' n'a pas été trouvé dans notre base de données. Veuillez vérifier l'orthographe ou essayer un autre film."]

    movie_index = matching_movies.index[0]
    input_movie_data = movies_df.loc[[movie_index]] 

    feature_weighting_transformer = recommender_pipeline.named_steps['feature_weighting']
    tfidf_vectorizer = recommender_pipeline.named_steps['tfidf_vectorizer']
    nn_model = recommender_pipeline.named_steps['nn_model']

    weighted_text_for_input = feature_weighting_transformer.transform(input_movie_data)

    input_movie_vector = tfidf_vectorizer.transform(weighted_text_for_input)

    distances, indices = nn_model.kneighbors(input_movie_vector, n_neighbors=num_recommendations + 1)

    recommended_movie_indices = indices.flatten()[1:]

    recommendations = []

    for i, idx in enumerate(recommended_movie_indices):
        title = movies_df.loc[idx, ['originalTitle']]['originalTitle']
        tconst = movies_df.loc[idx, ['tconst']]['tconst']
        poster_path = movies_df.loc[idx, ['poster_path']]['poster_path']
        dist = distances.flatten()[i+1] 
        recommendations.append({'title':title,
                                'tconst':tconst,
                                'poster_path':'https://image.tmdb.org/t/p/w500'+poster_path
                                }) 
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
model = genai.GenerativeModel("gemini-2.5-flash")
gemini_chat = model.start_chat(history=[])

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return render_template("chat.html")
    else:  
        try:
            data = request.get_json(force=True)
            user_input = data.get("message", "")
            response = gemini_chat.send_message(user_input)
            return jsonify({"reply": response.text})
        except Exception as e:
            import traceback
            traceback.print_exc()

            return jsonify({
                "reply": f"Erreur Gemini : {e}"
            }), 500


if __name__ == "__main__":
    app.run(debug=True)
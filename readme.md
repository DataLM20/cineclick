# 🎬 CineClik – Recommandation intelligente de films

## 🚀 À propos du projet
**CineClik** est une application web de recommandation de films développée par des apprenants Datascientest/Liora (Audrey, Karim) dans le cadre d’un projet de data Enginieer.  
Son but : **réenchanter l’expérience cinéma en salle** grâce à des suggestions personnalisées, interactives et intelligentes.

🎯 Ce prototype fonctionnel propose une interface simple pour :
- Explorer 3 fonctionnalités de recommandation innovantes
- Visualiser des KPI pertinents
- Offrir un service digital complémentaire au cinéma traditionnel

---

## 👨‍💻 Fonctionnalités principales

- **🎭 Mood to Film** : recommandations basées sur les envies du moment (Text Mining + TF-IDF)
- **🎞️ Film to Film** : suggestions de films similaires (algorithme K-Nearest Neighbors)
- **👥 Group to Film** : recommandations pour un groupe de spectateurs (TF-IDF + similarité cosinus)
- **🤖 Chat CineClik** : assistant conversationnel intégré (via l’API Gemini)
- **🎥 Bande-annonce** : intégration automatique des trailers via l’API YouTube
- **🖥️ Interface** fluide avec Flask, HTML/CSS, JS et Select2

---

## 🔒 Sécurité & bonnes pratiques

- Les **clés API** (Gemini et YouTube) sont sécurisées dans un fichier `.env`
- Utilisation du module `python-dotenv`
- Le fichier `.env` est exclu du dépôt via `.gitignore`

---

## 🔍 Étude de marché & choix des filtres

Suite à une analyse Ipsos 2026 sur les préférences des spectateurs français :

**Filtres de nettoyage appliqués :**
- ✅ Langues conservées : Français & Anglais
- ❌ Suppression : téléfilms, séries, films adultes, documentaires, émissions
- ❌ Genres exclus : biographie, musical, news
- 🕒 Films trop courts et films antérieurs à 1990 supprimés
- ⭐ Seuls les films avec une note ≥ 5/10 ont été conservés

---

## ⚙️ Technologies utilisées

- Python, Pandas, Scikit-learn
- Flask (serveur web), HTML/CSS/JS, Select2
- Gemini API (chat), YouTube API (vidéos)
- dotenv pour la gestion des clés sensibles

---

## 📁 Structure du projet

cineclik/
│
├── app.py # Application principale Flask
├── requirements.txt # Dépendances Python
├── df_clean.csv # Dataset nettoyé
├── templates/
│ ├── index.html
│ └── infos.html
├── static/ # Fichiers CSS/JS/images
└── .env # Clés API (non partagé)


---

## 🧪 Lancer le projet en local

```bash
git clone 
cd cineclik
python3 -m venv ../venv
source ../venv/bin/activate  # ou .\venv\Scripts\activate sur Windows
pip install -r requirements.txt
python app.py

Accédez ensuite à http://127.0.0.1:5000 dans votre navigateur.


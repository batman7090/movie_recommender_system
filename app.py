"""Streamlit client for the versioned recommendation API."""
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


@st.cache_data(ttl=300)
def catalog():
    response = requests.get(f"{API_URL}/movies", timeout=10)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=3600)
def poster(movie_id):
    key = os.getenv("TMDB_API_KEY")
    if not key:
        return None
    try:
        response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}",
                                params={"api_key": key}, timeout=5)
        response.raise_for_status()
        path = response.json().get("poster_path")
        return f"https://image.tmdb.org/t/p/w500{path}" if path else None
    except (requests.RequestException, ValueError):
        return None


st.set_page_config(page_title="Movie Discovery", page_icon="🎬")
st.title("🎬 Movie Discovery")
st.caption("Content-based recommendations • versioned models • observable inference")
try:
    movies = catalog()
    selected = st.selectbox("Choose a movie", movies, format_func=lambda m: f"{m['title']} ({m['id']})")
    if st.button("Find similar movies", type="primary"):
        response = requests.get(f"{API_URL}/recommendations/{selected['id']}", timeout=10)
        response.raise_for_status()
        result = response.json()
        st.caption(f"Model version: {result['model_version']}")
        for col, movie in zip(st.columns(5), result["recommendations"]):
            with col:
                url = poster(movie["id"])
                if url:
                    st.image(url)
                st.write(movie["title"])
                st.caption(f"Similarity {movie['score']:.2f}")
except (requests.RequestException, ValueError):
    st.error("The recommendation API is unavailable. Start the API and refresh this page.")

import os
import pickle
import requests

import pandas as pd
from pandas import DataFrame
import streamlit as st
from dotenv import load_dotenv

# loading the environment variables
load_dotenv()

# URL variable declarations
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/w500"

# function to fetch the poster
def fetch_poster(movie_id:int):
    """
    Function fetches the movie poster url

    movie_id: integer

    return: returns the image url
    """
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US")
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]

def recommend_movie_list(movie:str, df: DataFrame, top_n: int=5):
    """
    Function fetches the movie poster url

    movie: name of the movie
    df: the dataframe
    top_n: top movie recommendation cap

    return: tuple of lists with movie names and poster urls
    """
    movie_index = df[df["title"] == movie].index[0]
    similarities = pickle.load(open('artifacts/similarity_vectors.pkl', 'rb'))
    distances = similarities[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]

    recomended_movies, recomended_movies_poster = [], []
    for i in movie_list:
        movie_id = df.iloc[i[0]].id

        recomended_movies.append(df.iloc[i[0]].title)
        recomended_movies_poster.append(fetch_poster(movie_id))

    return recomended_movies, recomended_movies_poster


# Metric: Genre Overlap Score
def genre_overlap_score(movie_index, recommendations, df):
    input_genres = set(df.iloc[movie_index].genres)
    scores = []

    for idx in recommendations:
        rec_genres = set(df.iloc[idx].genres)
        overlap = len(input_genres & rec_genres) / len(input_genres)
        scores.append(overlap)

    return sum(scores) / len(scores)

# Precision@K using genre similarity (standard proxy) 
# Precision@K = relevant recommendations / K
def precision_at_k(movie_index, rec_indices, df, k=5):
    input_genres = set(df.iloc[movie_index].genres)
    relevant = 0

    for idx in rec_indices[:k]:
        if len(input_genres & set(df.iloc[idx].genres)) > 0:
            relevant += 1

    return relevant / k


# loading the movie data from the pickel file
movies_dict = pickle.load(open('artifacts/movie_dict.pkl', 'rb'))
movies_df = pd.DataFrame(movies_dict)
movies_list = movies_df['title'].values

# main streamlit app
st.title("Movie Recommender System")

selected_movie = st.selectbox('Select the movie from drop down', movies_list)

if st.button('Recommend'):
    names, posters = recommend_movie_list(selected_movie, movies_df)
    
    cols = st.columns(5)
    for col, name, poster in zip(cols, names, posters):
        with col:
            st.text(name)
            if poster:
                st.image(poster)
            else:
                st.write("No poster")

import pandas as pd
import streamlit as st
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: white;
    }
    .stApp {
        background-color: #0E1117;
    }
    h1 {
        text-align: center;
        color: #E50914;
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset
movies = pd.read_csv('TeluguMovies_dataset.csv')

# Clean data
movies['Genre'] = movies['Genre'].fillna('')
movies['Overview'] = movies['Overview'].fillna('')
movies['Movie'] = movies['Movie'].fillna('')

# Combine features
movies['tags'] = movies['Genre'] + " " + movies['Overview']

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Similarity
similarity = cosine_similarity(vectors)

# Fetch poster
def fetch_poster(movie_name):

    api_key = st.secrets["API_KEY"]
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_name}"
    
    try:
        response = requests.get(url)
        data = response.json()

        if 'results' in data and len(data['results']) > 0:
            poster_path = data['results'][0].get('poster_path')
            
            if poster_path:
                return "https://image.tmdb.org/t/p/w500/" + poster_path
        
        return "https://via.placeholder.com/150"

    except:
        return "https://via.placeholder.com/150"

# Recommendation function
def recommend(movie):
    index = movies[movies['Movie'] == movie].index[0]
    distances = similarity[index]
    
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    names = []
    posters = []
    
    for i in movies_list:
        movie_name = movies.iloc[i[0]].Movie
        names.append(movie_name)
        posters.append(fetch_poster(movie_name))
    
    return names, posters

# UI
st.markdown("<h1>🎬 Telugu Movie Recommender</h1>", unsafe_allow_html=True)

selected_movie = st.selectbox("Select a movie", movies['Movie'].values)

# SINGLE BUTTON ONLY
if st.button("Recommend", key="recommend_btn"):
    with st.spinner("Finding best movies for you..."):
        names, posters = recommend(selected_movie)

        st.subheader("Recommended Movies:")

        cols = st.columns(5)

        for i in range(len(names)):
            with cols[i]:
                st.image(posters[i])
                st.markdown(f"**{names[i]}**")


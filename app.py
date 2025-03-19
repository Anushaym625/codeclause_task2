import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB

# Load models and dataset
cv = pickle.load(open('cv.pkl', 'rb'))
classifier = pickle.load(open('model.pkl', 'rb'))
mlb = pickle.load(open('mlb.pkl', 'rb'))
df = pd.read_csv(r"C:\Users\anush\OneDrive\Desktop\datasets\imdb_movies.csv")  # Load your dataset

# Initialize NLTK stopwords & stemmer
nltk.download('stopwords')
ps = PorterStemmer()

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower().split()
    text = [word for word in text if word not in stopwords.words('english')]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Prediction function
def predict_genre(overview):
    processed_text = preprocess_text(overview)
    transformed_text = cv.transform([processed_text])
    prediction = classifier.predict(transformed_text)
    genres = mlb.inverse_transform(prediction)[0]  # Decode multi-label output
    return genres

# Fetch IMDb rating and certificate
def get_movie_info(overview):
    matched_movie = df[df['Overview'].str.contains(overview[:30], case=False, na=False)]
    
    if not matched_movie.empty:
        certificate = matched_movie.get('Certificate', ['Unknown'])[0]
        imdb_rating = matched_movie.get('IMDB_Rating', ['N/A'])[0]  # Ensure correct column name
        return certificate, imdb_rating
    
    return "Unknown", "N/A"

# Streamlit UI
st.set_page_config(page_title="Movie Genre Predictor", layout="centered")

# IMDb logo
st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg", width=150)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #1E1E1E;
            color: white;
        }
        .stTextInput, .stTextArea, .stButton {
            font-size: 18px;
        }
        .yellow-box {
            background-color: #FFD700;
            padding: 10px;
            border-radius: 10px;
            font-size: 20px;
            color: black;
            font-weight: bold;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 Movie Genre Predictor")
st.write("Enter a movie overview to predict its genre, IMDb rating, and certificate.")

# User input
user_input = st.text_area("🎥 **Movie Overview:**", height=150)

if st.button("🔍 Predict Movie Details"):
    if user_input.strip():
        genres = predict_genre(user_input)
        certificate, imdb_rating = get_movie_info(user_input)

        # Yellow block for genres
        st.markdown(f"<div class='yellow-box'>🎭 Predicted Genres: {', '.join(genres)}</div>", unsafe_allow_html=True)
        
        st.info(f"📜 **Certificate:** {certificate}")
        st.warning(f"⭐ **IMDb Rating:** {imdb_rating}")
        
    else:
        st.error("❌ Please enter a valid movie overview.")


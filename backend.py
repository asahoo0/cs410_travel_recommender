# backend.py

import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load the trained TF-IDF vectorizer and city descriptions CSV
def load_models():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    city_df = pd.read_csv('city_stats.csv')
    return vectorizer, city_df

# Function to calculate similarity and recommend a city
def get_city_recommendation(user_input):
    # Load the models
    vectorizer, city_df = load_models()

    # Fit the vectorizer to the city descriptions
    city_tfidf = vectorizer.fit_transform(city_df['Description'])

    # Convert user input to a TF-IDF vector (same vectorizer used for cities)
    user_tfidf = vectorizer.transform([user_input])

    # Calculate cosine similarit  y between user input and city descriptions
    cosine_similarities = cosine_similarity(user_tfidf, city_tfidf)

    # Add the cosine similarity scores to the city dataframe
    city_df['Similarity'] = cosine_similarities.flatten()

    # Check if the user mentioned 'food' in their input and apply bonus if applicable
    if 'food' in user_input.lower():
        food_bonus = 0.1
        city_df.loc[city_df['average_rating'] > 4, 'Similarity'] += food_bonus

    # Get the city with the highest similarity score
    highest_similarity_city = city_df.loc[city_df['Similarity'].idxmax()]

    # Return the city details
    return highest_similarity_city[['City', 'Similarity', 'average_rating', 'Description']]


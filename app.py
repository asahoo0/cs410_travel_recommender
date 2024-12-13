# app.py

import streamlit as st
from backend import get_city_recommendation  # Import the backend function

# Set up the Streamlit frontend
st.title("Find Your Ideal Travel Destination")

# Ask the user to describe their ideal travel destination
user_input = st.text_area(
    "Describe your ideal destination in terms of food, climate, or any other preferences:",
    placeholder="E.g., I want a city with great food and rich history..."
)

# Calculate the similarity when the user submits their description
if st.button("Find Your City"):
    # Get city recommendation from the backend
    recommendation = get_city_recommendation(user_input)

    # Display the city with the highest similarity
    st.write("Your Ideal City Match:")
    st.write(f"**City:** {recommendation['City']}")
    st.write(f"**Similarity:** {recommendation['Similarity']}")
    st.write(f"**Rating:** {recommendation['average_rating']}")
    st.write(f"**Description:** {recommendation['Description']}")


# Travel Recommender Model Creation

## Overview

This project involves building a recommendation model to suggest travel destinations based on user input. The model uses a **TF-IDF vectorizer** to transform city descriptions into numerical features and then calculates **cosine similarity** to match user input with the most similar city descriptions. Additionally, the model includes a bonus for cities with high ratings if the user mentions food in their input.

---

## Requirements

To build and run the model, you need the following libraries:

### Python Libraries:
- `pandas`
- `nltk`
- `scikit-learn`
- `joblib`

### To install dependencies:
1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. Install the required packages:

   ```bash
   pip install pandas nltk scikit-learn joblib
   ```

3. Ensure you have the necessary NLTK resources:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

---

## Data Preprocessing

The model requires a dataset with city names, descriptions, and ratings. The data is loaded from a CSV file `TA_restaurants_curated.csv` that contains restaurant data, and another CSV `city_descriptions.csv` that contains city descriptions.

Here’s the basic data cleaning process:
1. Load the CSV file and drop any rows with missing ratings, city names, or reviews.
2. Group the data by city and calculate the average rating for each city.
3. Tokenize city descriptions and remove stop words to extract meaningful keywords.

---

## Building the Model

### Steps:

1. **Loading Data**:
   Load the CSV containing city data and ratings, and drop irrelevant columns.

   ```python
   data = pd.read_csv('/content/TA_restaurants_curated.csv', on_bad_lines='skip')
   data = data.dropna(subset=['Rating', 'City', 'Reviews'])
   data = data.drop(columns=['URL_TA', 'Ranking'])
   ```

2. **Grouping by City**:
   Group by the city and calculate the average rating.

   ```python
   grouped_data = data.groupby('City').agg(
       average_rating=('Rating', 'mean')
   ).reset_index()
   ```

3. **Tokenization and Stop Words Removal**:
   Use NLTK to tokenize descriptions and remove stopwords to extract keywords from city descriptions.

   ```python
   def extract_keywords(description):
       tokens = word_tokenize(description.lower())
       filtered_words = [word for word in tokens if word.isalnum() and word not in stop_words]
       return filtered_words
   data['keywords'] = data['Description'].apply(extract_keywords)
   ```

4. **TF-IDF Vectorization**:
   Use the **TfidfVectorizer** from scikit-learn to convert the descriptions into numeric vectors.

   ```python
   vectorizer = TfidfVectorizer(stop_words='english')
   city_tfidf = vectorizer.fit_transform(data['Description'])
   ```

5. **User Input and Cosine Similarity**:
   Calculate the cosine similarity between the user’s input and the city descriptions.

   ```python
   user_input = "I want a city with great food and rich history"
   user_tfidf = vectorizer.transform([user_input])
   cosine_similarities = cosine_similarity(user_tfidf, city_tfidf)
   data['Similarity'] = cosine_similarities.flatten()
   ```

6. **Bonus for Food Mention**:
   If the user mentions "food" in their input, cities with a rating above 4 get a bonus added to their similarity score.

   ```python
   food_bonus = 0.1
   if 'food' in user_input.lower():
       data.loc[data['average_rating'] > 4, 'Similarity'] += food_bonus
   ```

7. **Saving the Model**:
   Save the **TF-IDF vectorizer** model using **joblib** for later use.

   ```python
   import joblib
   joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
   ```

8. **Saving the DataFrame**:
   Save the DataFrame containing the calculated average ratings with descriptions.

   ```python
   data.to_csv('city_stats.csv', index=False)
   ```

---

## Loading and Using the Saved Model

1. **Loading the Saved Vectorizer**:

   ```python
   loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
   ```

2. **Transforming User Input**:
   Use the loaded vectorizer to transform new user input.

   ```python
   user_input = "I want a city with great food and rich history"
   user_tfidf = loaded_vectorizer.transform([user_input])
   ```

3. **Recalculating Cosine Similarity**:
   Load the saved DataFrame with city descriptions and similarity scores, and calculate cosine similarity again.

   ```python
   loaded_city_df = pd.read_csv('city_with_similarity.csv')
   cosine_similarities = cosine_similarity(user_tfidf, loaded_city_df['Description'])
   ```

4. **Applying the Food Bonus**:
   Add the food bonus again for cities with a rating greater than 4.

   ```python
   if 'food' in user_input.lower():
       loaded_city_df.loc[loaded_city_df['average_rating'] > 4, 'Similarity'] += food_bonus
   ```

5. **Displaying the Most Similar City**:
   Finally, display the city with the highest similarity score.

   ```python
   highest_similarity_city = loaded_city_df.loc[loaded_city_df['Similarity'].idxmax()]
   print(highest_similarity_city[['City', 'Similarity', 'average_rating']])
   ```


# Travel Recommender

## Overview

The Travel Recommender is a web application built with Streamlit that recommends travel destinations based on user preferences such as food, climate, and culture. The system uses machine learning techniques, such as TF-IDF and cosine similarity, to match user descriptions with a list of curated cities and their descriptions.

### Features:
- **User Input**: A text box where users can describe their ideal travel destination (e.g., climate, food, activities).
- **Recommendation Engine**: The system calculates cosine similarity between the user's description and curated city descriptions to suggest the best match.
- **City Information**: The recommendation includes city names, ratings, descriptions, and additional details.

---

## Requirements

Before you begin, ensure you have the following installed on your machine:

### Python:
- Python 3.x (preferably 3.7 or higher)

### Dependencies:
You can install the necessary dependencies using the provided `requirements.txt` file. Here are the key dependencies:

```text
streamlit==1.18.0
pandas==2.0.0
scikit-learn==1.2.1
nltk==3.7
joblib==1.1.0
```

### To install dependencies:
1. Clone or download the repository to your local machine.
2. Navigate to the project directory.
3. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

4. Install the requirements using pip:

   ```bash
   pip install -r requirements.txt
   ```

### Download Necessary NLTK Resources:
The recommender uses NLTK for text processing. The following resources need to be downloaded:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## How to Use

1. **Run the App**:
   After installing the dependencies, run the Streamlit app by executing the following command in the project directory:

   ```bash
   streamlit run app.py
   ```

2. **User Input**:
   On the webpage that opens, enter a description of your ideal travel destination. You can mention preferences such as the type of food, climate, or activities you would like (e.g., "I want a city with great food and rich history").

3. **Get Recommendations**:
   Based on your input, the system will recommend a city from the curated list. It will display the city name, similarity score, average rating, and a brief description.

4. **City Details**:
   The app will show a list of cities sorted by similarity to your input. The city with the highest similarity score is the top recommendation.

---

## Project Structure

```
TravelRecommender/
│
├── app.py                        # Main Streamlit app
├── city_descriptions.csv         # Curated list of cities with descriptions
├── city_with_similarity.csv      # Cities with calculated similarity scores
├── tfidf_vectorizer.pkl          # Saved TF-IDF Vectorizer model
├── requirements.txt              # Dependencies for the project
└── README.md                     # Project documentation
```

---

## How the Recommendation Works

1. **Data Preprocessing**:
   - The data consists of cities with descriptions and ratings.
   - The app uses `TfidfVectorizer` from scikit-learn to convert city descriptions into numerical vectors.
   
2. **Cosine Similarity**:
   - The user's input is transformed into a TF-IDF vector.
   - The similarity between the user input and city descriptions is calculated using **cosine similarity**.

3. **Bonus for Food Preference**:
   - If the user mentions food in their input, a bonus is applied to cities with an average rating higher than 4.

4. **Top Recommendation**:
   - The city with the highest similarity score is displayed as the top recommendation.

---

## Saving and Loading Models

- **Model Saving**: 
  - The trained **TF-IDF Vectorizer** is saved to `tfidf_vectorizer.pkl` using **joblib**.
  
- **Model Loading**:
  - When re-running the application, the pre-trained model is loaded to avoid retraining.

---

## Future Improvements

- **More data**: Integrate more city data for better recommendations.
- **User Ratings**: Allow users to rate the recommended cities for continuous improvement.
- **Enhanced Algorithms**: Implement more sophisticated models like deep learning-based recommendation systems.

## Sources

- The dataset for city descriptions was curated using ChatGPT.
- The food reviews dataset for 31 cities: https://www.kaggle.com/datasets/damienbeneschi/krakow-ta-restaurans-data-raw
- Streamlit, Scikit-learn, and NLTK libraries were used for building the app and processing the data.

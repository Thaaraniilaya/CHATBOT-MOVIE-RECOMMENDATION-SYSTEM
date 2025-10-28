import pandas as pd
import numpy as np
import sys
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process # Requires: pip install fuzzywuzzy python-levenshtein
from flask import Flask, request, jsonify, render_template

# --- 0. Flask Initialization ---
app = Flask(__name__)

# --- 1. Data Loading and Model Training (Core Logic from chatbot_logic.py) ---

# NOTE: Update these paths to match where your CSVs are stored relative to app.py
movies_file_path = r"C:\Users\HP\Downloads\tmdb_5000_movies.csv\tmdb_5000_movies.csv"
credits_file_path = r"C:\Users\HP\Downloads\tmdb_5000_credits.csv\tmdb_5000_credits.csv"

try:
    movies_df = pd.read_csv(movies_file_path)
    credits_df = pd.read_csv(credits_file_path)
except FileNotFoundError:
    print("FATAL ERROR: Data files not found. Check file paths. Exiting.")
    sys.exit()

# Data Processing Steps (Unchanged from your working version)
credits_df.columns = ['id', 'title_c', 'cast', 'crew']
df = movies_df.merge(credits_df, on='id')
df = df[['id', 'title', 'genres', 'keywords', 'cast', 'crew', 'overview']] 
df.dropna(inplace=True)

# Helper functions (omitted for brevity, but they must be present in your full app.py)
def parse_and_clean_features(obj):
    L = []
    try:
        for i in literal_eval(obj):
            L.append(i['name'])
    except: pass
    return L

def get_top_3_cast(obj):
    L = []
    counter = 0
    for i in literal_eval(obj):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else: break
    return L

def get_director(obj):
    L = []
    for i in literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

df['genres'] = df['genres'].apply(parse_and_clean_features)
df['keywords'] = df['keywords'].apply(parse_and_clean_features)
df['cast'] = df['cast'].apply(get_top_3_cast)
df['crew'] = df['crew'].apply(get_director)

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else: return ''

features_to_clean = ['cast', 'crew', 'genres', 'keywords']
for feature in features_to_clean:
    df[feature] = df[feature].apply(clean_data)

def create_soup(x):
    overview = x['overview'].lower().replace(" ", "") if isinstance(x['overview'], str) else ''
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['crew']) + ' ' + ' '.join(x['genres']) + ' ' + overview

df['soup'] = df.apply(create_soup, axis=1)

# Vectorization & Similarity
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
df = df.reset_index(drop=True)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
all_titles = list(indices.index)
print("✅ Model loaded and ready.")


# --- 2. Recommendation Function ---
def get_recommendations_robust(title):
    """Returns recommendations and a status message/matched title."""
    best_match = process.extractOne(title, all_titles, score_cutoff=75)
    
    if best_match:
        matched_title = best_match[0]
        score = best_match[1]
             
        idx = indices[matched_title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_10_similar = sim_scores[1:11]
        movie_indices = [i[0] for i in top_10_similar]
        recommendations = df['title'].iloc[movie_indices].tolist()
        
        message = f"Recommendations for: **{matched_title}**"
        if matched_title != title:
            message += f" (Fuzzy Match Confidence: {score}%)"
            
        return recommendations, message

    else:
        return [], f"Error: Movie '{title}' not found in the database. Please check spelling."


# --- 3. Flask Routes (API and Frontend) ---

@app.route('/')
def index():
    """Serves the main HTML page (index.html)."""
    # Flask looks for this file in the 'templates' folder automatically
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_api():
    """API endpoint for fetching recommendations."""
    data = request.get_json()
    movie_title = data.get('title', '')

    if not movie_title:
        return jsonify({'error': 'No movie title provided'}), 400

    recommendations, message = get_recommendations_robust(movie_title)
    
    # Return results as JSON
    return jsonify({
        'status': 'success' if recommendations else 'error',
        'message': message,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    # Run the Flask server
    app.run(debug=True)
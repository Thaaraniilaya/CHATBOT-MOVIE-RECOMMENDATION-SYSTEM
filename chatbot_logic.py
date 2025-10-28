import pandas as pd
import numpy as np
import sys
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Note: Ensure you have 'fuzzywuzzy' and 'python-levenshtein' installed locally
from fuzzywuzzy import process 

# --- 1. Data Loading and Processing (CORRECTED) ---
print("Loading and processing movie data... This may take a moment.")

# Use the paths for the uploaded files as they are accessible
movies_file_path = r"C:\Users\HP\Downloads\tmdb_5000_movies.csv\tmdb_5000_movies.csv"
credits_file_path = r"C:\Users\HP\Downloads\tmdb_5000_credits.csv\tmdb_5000_credits.csv"

try:
    movies_df = pd.read_csv(movies_file_path)
    credits_df = pd.read_csv(credits_file_path)
except FileNotFoundError:
    print("\n[ERROR] Ensure data files are correctly linked.")
    sys.exit()

# Rename columns in credits_df to avoid conflicts and prepare for merge on 'id'
credits_df.columns = ['id', 'title_c', 'cast', 'crew']
df = movies_df.merge(credits_df, on='id')

# CORRECTED LINE: The title column from movies_df is just 'title'
df = df[['id', 'title', 'genres', 'keywords', 'cast', 'crew', 'overview']] 
df.dropna(inplace=True)

def parse_and_clean_features(obj):
    L = []
    try:
        for i in literal_eval(obj):
            L.append(i['name'])
    except:
        pass
    return L

def get_top_3_cast(obj):
    L = []
    counter = 0
    for i in literal_eval(obj):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
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
        else:
            return ''

features_to_clean = ['cast', 'crew', 'genres', 'keywords']
for feature in features_to_clean:
    df[feature] = df[feature].apply(clean_data)

def create_soup(x):
    overview = x['overview'].lower().replace(" ", "") if isinstance(x['overview'], str) else ''
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['crew']) + ' ' + ' '.join(x['genres']) + ' ' + overview

df['soup'] = df.apply(create_soup, axis=1)

# --- 2. Vectorization & Similarity ---

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
df = df.reset_index(drop=True)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
print("Data processing complete. Ready for recommendations.")


# --- 3. Robust Recommendation Function ---

def get_recommendations_robust(title, cosine_sim=cosine_sim, df=df, indices=indices):
    """
    Robust function that uses fuzzy matching to find the closest movie title 
    and returns recommendations.
    """
    all_titles = list(indices.index)
    
    # Find the best match with a similarity score of at least 75
    best_match = process.extractOne(title, all_titles, score_cutoff=75)
    
    if best_match:
        matched_title = best_match[0]
        score = best_match[1]
        
        # Notify the user if a correction was made
        if matched_title != title:
             print(f"\n[INFO] Searching for the closest match: '{matched_title}' (Confidence: {score}%)")
             
        idx = indices[matched_title]
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_10_similar = sim_scores[1:11]
        movie_indices = [i[0] for i in top_10_similar]
        
        return df['title'].iloc[movie_indices].tolist()

    else:
        # No good match found
        return [f"Movie '{title}' not found in the database. Please check spelling or try a different title."]

# --- 4. Interactive Chatbot Loop ---

def run_chatbot():
    """Initializes and runs the interactive chatbot loop."""
    print("\n" + "="*50)
    print("🎬 Movie Recommendation Chatbot (Content-Based) 🤖")
    print("Type a movie title you like, or type 'exit' to quit.")
    print("="*50)
    
    while True:
        try:
            user_input = input("\n> I liked: ")
        except EOFError:
            print("\nGoodbye! Happy watching.")
            break
        
        if user_input.lower() == 'exit':
            print("Goodbye! Happy watching.")
            break

        if not user_input.strip():
            print("Please enter a movie title.")
            continue

        recommendations = get_recommendations_robust(user_input)
        
        if "not found" not in recommendations[0]:
            print(f"\nIf you liked '{user_input}', you might also like these 10 movies:")
            for i, rec_title in enumerate(recommendations):
                print(f"{i+1}. {rec_title}")
        else:
            print(recommendations[0])

if __name__ == "__main__":
    run_chatbot()
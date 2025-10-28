import pandas as pd
import streamlit as st
import sys
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process 

# --- 1. Data Loading and Processing (Copied from chatbot_logic.py) ---

# Use Streamlit's cache decorator to avoid reloading and re-calculating everything on every user interaction
@st.cache_data
def load_data_and_train_model():
    """Loads data, trains the model, and returns necessary components."""
    try:
        # NOTE: Update these paths if your data files are not in the same directory as app.py
        movies_file_path = "tmdb_5000_movies.csv.zip/tmdb_5000_movies.csv"
        credits_file_path = "tmdb_5000_credits.csv.zip/tmdb_5000_credits.csv"
        movies_df = pd.read_csv(movies_file_path)
        credits_df = pd.read_csv(credits_file_path)
    except FileNotFoundError:
        st.error("Data files not found. Please check paths or file accessibility.")
        sys.exit()

    # Data Processing Steps (Unchanged from your working code)
    credits_df.columns = ['id', 'title_c', 'cast', 'crew']
    df = movies_df.merge(credits_df, on='id')
    df = df[['id', 'title', 'genres', 'keywords', 'cast', 'crew', 'overview']] 
    df.dropna(inplace=True)

    # Helper functions (Defined internally to stay within the cached function)
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

    # Vectorization & Similarity
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    df = df.reset_index(drop=True)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    return cosine_sim, df, indices, list(indices.index)

# Load the model and data components once
cosine_sim, df, indices, all_titles = load_data_and_train_model()


# --- 2. Recommendation Logic (Reusing and adapting your robust function) ---

def get_recommendations_robust(title):
    """
    Robust function that uses fuzzy matching and Streamlit feedback.
    """
    best_match = process.extractOne(title, all_titles, score_cutoff=75)
    
    if best_match:
        matched_title = best_match[0]
        score = best_match[1]
        
        # Display the correction to the user in the app
        if matched_title != title:
            st.info(f"🔎 **Searching for:** '{matched_title}' (Confidence: {score}%)")
             
        idx = indices[matched_title]
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_10_similar = sim_scores[1:11]
        movie_indices = [i[0] for i in top_10_similar]
        
        return df['title'].iloc[movie_indices].tolist(), matched_title

    else:
        st.error(f"Movie **'{title}'** not found in the database. Please try a different title or check the spelling.")
        return [], None


# --- 3. Streamlit Interface (The Front End) ---

# Set the title and layout
st.title("🎬 Content-Based Movie Recommender")
st.markdown("Enter the name of a movie you liked, and I'll find 10 similar ones.")

# Input field for the user
user_input = st.text_input("I liked this movie:", placeholder="e.g., The Dark Knight Rises or Avatar")

# Recommendation button
if st.button("Get Recommendations"):
    if user_input:
        with st.spinner('Generating recommendations...'):
            recommendations, matched_title = get_recommendations_robust(user_input)
            
            if recommendations:
                st.success(f"### Results for: {matched_title}")
                
                # Display recommendations as a numbered list
                for i, rec_title in enumerate(recommendations):
                    st.markdown(f"**{i+1}.** {rec_title}")
            
    else:
        st.warning("Please enter a movie title to get recommendations.")

# Optional: Add a brief explanation of how it works
st.sidebar.title("About the System")
st.sidebar.markdown("""
This application uses a **Content-Based Filtering** algorithm. 
It recommends movies based on the similarity of their **genres, keywords, cast, and director** to the movie you enter.
""")
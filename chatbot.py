import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Data Preparation & Cleaning ---

# Load the datasets
movies_df = pd.read_csv(r"C:\Users\HP\Downloads\tmdb_5000_movies.csv\tmdb_5000_movies.csv")
credits_df = pd.read_csv(r"C:\Users\HP\Downloads\tmdb_5000_credits.csv\tmdb_5000_credits.csv")

# Merge the two DataFrames on the ID column
# Rename 'movie_id' in credits to 'id' for merging
credits_df.columns = ['id', 'title', 'cast', 'crew']
df = movies_df.merge(credits_df, on='id')

# Keep only the columns relevant for content-based filtering
df = df[['id', 'title_x', 'genres', 'keywords', 'cast', 'crew', 'overview']]
df.rename(columns={'title_x': 'title'}, inplace=True)
df.dropna(inplace=True)

# Function to parse the stringified JSON into a clean list of names/values
def parse_and_clean_features(obj):
    """Converts stringified JSON to a list of names/values."""
    # literal_eval safely converts a string of Python-like data structure to an object
    L = []
    try:
        for i in literal_eval(obj):
            L.append(i['name'])
    except:
        pass # Return an empty list if parsing fails
    return L

# Function to limit cast list to top 3 and extract the Director
def get_top_3_cast(obj):
    """Gets the names of the top 3 actors."""
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
    """Extracts the director's name from the crew list."""
    L = []
    for i in literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# Apply the cleaning functions
df['genres'] = df['genres'].apply(parse_and_clean_features)
df['keywords'] = df['keywords'].apply(parse_and_clean_features)
df['cast'] = df['cast'].apply(get_top_3_cast)
df['crew'] = df['crew'].apply(get_director) # This column will now only contain the Director

# Further cleaning: lowercasing and removing spaces from names
# This ensures 'Chris Evans' and 'Chris Pratt' are treated as different words ('chrisevans' vs 'chrispratt')
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

# Create a single metadata 'soup' string for each movie
def create_soup(x):
    """Combines all relevant features into a single, space-separated string."""
    # Adding overview as well, just in case a movie lacks other data
    overview = x['overview'].lower().replace(" ", "") if isinstance(x['overview'], str) else ''
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['crew']) + ' ' + ' '.join(x['genres']) + ' ' + overview

df['soup'] = df.apply(create_soup, axis=1)


# --- 2. Text Vectorization & 3. Similarity Calculation ---

# Use CountVectorizer to convert the 'soup' (text) into a matrix of token counts
# Stopwords are filtered out to focus on meaningful words
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])

# Compute the Cosine Similarity matrix
# This calculates the similarity score between every pair of movies
cosine_sim = cosine_similarity(count_matrix, count_matrix)


# --- 4. Recommendation Function (The Chatbot Logic) ---

# Construct a reverse map of movie titles to their index in the DataFrame
df = df.reset_index()
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim, df=df, indices=indices):
    """
    Function to get movie recommendations based on Cosine Similarity.
    
    Args:
        title (str): The title of the movie the user liked.
    
    Returns:
        list: A list of the top 10 most similar movie titles.
    """
    if title not in indices:
        return [f"Movie '{title}' not found. Please check spelling or try a different title."]

    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores (highest score first)
    # The movie itself (score of 1.0) will be at index 0, so we skip it later
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 10 most similar movies (skipping the first one)
    top_10_similar = sim_scores[1:11]

    # Return the titles of the top 10 recommended movies
    movie_indices = [i[0] for i in top_10_similar]
    
    return df['title'].iloc[movie_indices].tolist()

# --- Example of Chatbot Interaction ---
print("\n--- Example Recommendations ---")
movie_title = 'The Dark Knight Rises'
recommendations = get_recommendations(movie_title)

print(f"If you liked '{movie_title}', you might also like these 10 movies:")
if isinstance(recommendations, list):
    for i, rec_title in enumerate(recommendations):
        print(f"{i+1}. {rec_title}")

# Example of a movie not found
print("\n--- Error Example ---")
print(get_recommendations('Lord of the Rings')) # Title should be 'The Lord of the Rings: The Return of the King'
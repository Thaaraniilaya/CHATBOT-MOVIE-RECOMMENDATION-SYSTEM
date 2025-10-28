import pandas as pd

# Define the file paths as they are accessible in the environment
movies_file_path = r"C:\Users\HP\Downloads\tmdb_5000_movies.csv\tmdb_5000_movies.csv"
credits_file_path = r"C:\Users\HP\Downloads\tmdb_5000_credits.csv\tmdb_5000_credits.csv"

# Load the datasets into pandas DataFrames
try:
    movies_df = pd.read_csv(movies_file_path)
    credits_df = pd.read_csv(credits_file_path)

    print("✅ Datasets loaded successfully!")
    
    # Display the first 5 rows of each DataFrame to verify
    print("\n--- Movies DataFrame (movies_df.head()) ---")
    print(movies_df.head())
    
    print("\n--- Credits DataFrame (credits_df.head()) ---")
    print(credits_df.head())

except FileNotFoundError:
    print(f"Error: One or both files were not found. Please ensure both files are available in the current directory or check the file paths.")
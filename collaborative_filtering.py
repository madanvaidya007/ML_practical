# collaborative_filtering.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("ratings.csv")

# Create User-Item Matrix
user_movie_matrix = df.pivot_table(index='userId', columns='movieId', values='rating')

# Fill missing ratings with 0 (can also use mean imputation)
user_movie_matrix_filled = user_movie_matrix.fillna(0)

# Compute similarity between users
user_similarity = cosine_similarity(user_movie_matrix_filled)

# Convert to DataFrame
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

print("User Similarity Matrix:\n", user_similarity_df)

# Function to recommend movies to a user
def recommend_movies(user_id, top_n=2):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id)  # exclude the user itself

    weighted_scores = {}
    for other_user, similarity_score in similar_users.items():
        other_user_ratings = user_movie_matrix.loc[other_user]
        for movie, rating in other_user_ratings.dropna().items():
            if pd.isna(user_movie_matrix.loc[user_id, movie]):
                if movie not in weighted_scores:
                    weighted_scores[movie] = 0
                weighted_scores[movie] += similarity_score * rating

    recommendations = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [movie for movie, score in recommendations]

# Example: Recommend for user 1
recommended = recommend_movies(1) 
print("\nRecommended movies for user 1:", recommended)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI
from dotenv import load_dotenv
import os

client = OpenAI()


def get_ratings() -> pd.DataFrame:

    # Create a user-item matrix
    ratings_data = {
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3],
        'movie_id': [1, 2, 3, 2, 4, 1, 4, 5],
        'rating': [5, 4, 5, 4, 5, 4, 5, 4]
    }
    ratings_df = pd.DataFrame(ratings_data)

    return ratings_df


def get_movies() -> pd.DataFrame:
    movies_data = {
        'movie_id': [1, 2, 3, 4, 5],
        'title': ['The Matrix', 'Titanic', 'Inception', 'The Godfather', 'Toy Story'],
        'genres': ['Action|Sci-Fi', 'Romance|Drama', 'Action|Sci-Fi|Thriller', 'Crime|Drama',
                   'Animation|Children'],
        'ratings': [4.8, 4.2, 4.9, 4.7, 4.5]
    }
    movies_df = pd.DataFrame(movies_data)

    return movies_df


def vectorize_genres(movies_df: pd.DataFrame) -> pd.DataFrame:
    tfidf = TfidfVectorizer(stop_words='english')

    # Fill any NaN values
    movies_df['genres'] = movies_df['genres'].fillna('')

    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim


def get_recommendations(title, movies_df, cosine_sim):
    idx = movies_df[movies_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 3 similar movies
    sim_scores = sim_scores[1:4]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]


def get_collaborative_recommendations(user_id, model_knn, user_item_matrix, movies_df, n_recommendations=3):
    distances, indices = model_knn.kneighbors(
        user_item_matrix.loc[user_id].values.reshape(1, -1),
        n_neighbors=n_recommendations
    )

    recommendations = user_item_matrix.index[indices.flatten()][1:]
    return movies_df[movies_df['movie_id'].isin(recommendations)].title.values


def chat_with_gpt(prompt) -> str:
    messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150
    )

    return response.choices[0].message.content


def main():
    move = "The Matrix"
    movies_df = get_movies()
    cosine_sim = vectorize_genres(movies_df)
    print(get_recommendations(move, movies_df, cosine_sim))

    # Merge with movies_df
    ratings_df = get_ratings()
    merged_df = pd.merge(ratings_df, movies_df, on='movie_id')

    # Create a user-item ratings matrix
    user_item_matrix = merged_df.pivot_table(
        index='user_id', columns='title', values='rating').fillna(0)

    # Train NearestNeighbors model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(user_item_matrix)

    print(get_collaborative_recommendations(
        1, model_knn, user_item_matrix, movies_df))

    user_input = "Can you recommend movies like Inception?"
    recommendations = get_recommendations('Inception', movies_df, cosine_sim)
    gpt_prompt = f"""
        The user asked: '{user_input}'. Based on their request, I recommend:
        {','.join(recommendations)}. Provide this in a conversational style.
    """
    response = chat_with_gpt(gpt_prompt)
    print(response)


if __name__ == "__main__":
    load_dotenv()
    main()

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate

# Load Data
ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
movies_df = pd.read_csv('ml-latest-small/movies.csv')
movies_df = movies_df[movies_df['movieId'].isin(ratings_df['movieId'])]

# Processing Genres
movies_df["genres"] = (
    movies_df["genres"]
    .str.strip()
    .str.replace(" ", "")
    .str.replace(r"[^a-zA-Z|]", "", regex=True)
    .str.lower()
)

movies_df["genres"] = movies_df["genres"].str.split("|")

# Transforming genres into binary columns
mlb = MultiLabelBinarizer()
genre_df = pd.DataFrame(mlb.fit_transform(movies_df["genres"]), columns=mlb.classes_, index=movies_df.index)

# Merging genre binary columns with original df
movies_df = pd.concat([movies_df.drop(columns=["genres"]), genre_df], axis=1)

# Extracting movie genres without unnecessary columns
movie_genres = movies_df.drop(columns=['movieId', 'title'])

# Compute Content Similarity
content_similarity = cosine_similarity(movie_genres, movie_genres)

movies_df = pd.read_csv('ml-latest-small/movies.csv')
movies_df = movies_df[movies_df['movieId'].isin(ratings_df['movieId'])]

# Processing Genres
movies_df["genres"] = (
    movies_df["genres"]
    .str.strip()
    .str.replace(" ", "")
    .str.replace(r"[^a-zA-Z|]", "", regex=True)
    .str.lower()
)

# Splitting genres into lists
movies_df["genres"] = movies_df["genres"].str.split("|")
# Bayesian Rating Adjustment
global_mean_rating = ratings_df['rating'].mean()
# Calculate the count of ratings for each product (movieId)
rating_counts = ratings_df.groupby('movieId')['rating'].count()
C = rating_counts.median()

def calculate_bayesian_rating(df, C, global_mean_rating):
    movie_stats = df.groupby('movieId').agg(
        R=('rating', 'mean'),  # Mean rating for the movie
        v=('rating', 'count') # Number of ratings for the movie
    ).reset_index()
    movie_stats['bayesian_rating'] = (
        (movie_stats['v'] * movie_stats['R'] + C * global_mean_rating) / (movie_stats['v'] + C)
    )
    return movie_stats

# Calculate Bayesian ratings using C
movie_stats = calculate_bayesian_rating(ratings_df, C=C, global_mean_rating=global_mean_rating)

# Inspect the resulting DataFrame
print(movie_stats.head())
ratings_df = ratings_df.merge(movie_stats[['movieId', 'bayesian_rating']], on='movieId', how='left')
from sklearn.preprocessing import MultiLabelBinarizer, QuantileTransformer

# Normalize Ratings
def normalize_ratings(df):
    df['normalized_rating'] = (df['rating'] - df['bayesian_rating']) / (df['rating'].std() + 1e-8)
    return df
ratings_df = normalize_ratings(ratings_df)

user_mapper = dict(zip(np.unique(ratings_df["userId"]), range(ratings_df["userId"].nunique())))
movie_mapper = dict(zip(np.unique(ratings_df["movieId"]), range(ratings_df["movieId"].nunique())))
movie_inv_mapper = {v: k for k, v in movie_mapper.items()}

# defines NCF Model
num_users = len(user_mapper)
num_movies = len(movie_mapper)
embedding_dim = 50

user_input = Input(shape=(1,), name='user_input')
movie_input = Input(shape=(1,), name='movie_input')
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding')(user_input)
movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_dim, name='movie_embedding')(movie_input)
user_vector = Flatten()(user_embedding)
movie_vector = Flatten()(movie_embedding)
concat = Concatenate()([user_vector, movie_vector])
from tensorflow.keras.regularizers import l2

dense_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(concat)
dense_2 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dense_1)
dense_3 = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(dense_2)
output = Dense(1, activation='linear')(dense_3)

ncf_model = Model(inputs=[user_input, movie_input], outputs=output)
ncf_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# trains NCF Model
train_data = ratings_df.sample(frac=0.8, random_state=42)
test_data = ratings_df.drop(train_data.index)

train_user = train_data['userId'].map(user_mapper).values
train_movie = train_data['movieId'].map(movie_mapper).values
train_labels = train_data['normalized_rating'].values

test_user = test_data['userId'].map(user_mapper).values
test_movie = test_data['movieId'].map(movie_mapper).values
test_labels = test_data['normalized_rating'].values

ncf_model.fit([train_user, train_movie], train_labels,
              validation_data=([test_user, test_movie], test_labels),
              epochs=10, batch_size=256)

# Predictions
def predict_ratings(user_id, movie_ids):
    user_array = np.array([user_mapper[user_id]] * len(movie_ids))
    movie_array = np.array([movie_mapper[movie_id] for movie_id in movie_ids])
    predictions = ncf_model.predict([user_array, movie_array]).flatten()
    return dict(zip(movie_ids, predictions))

# Recommendations
def hybrid_recommendation(user_id, k=10, alpha=0.7, reverse=False):
    user_movies = train_data[train_data['userId'] == user_id]['movieId'].tolist()
    unrated_movies = list(set(movie_mapper.keys()) - set(user_movies))

    # NCF Predictions
    ncf_predictions = predict_ratings(user_id, unrated_movies)

    # Content Similarity
    content_scores = np.zeros(len(movie_mapper))
    for movie_id in user_movies:
        if movie_id in movie_mapper:
            movie_idx = movie_mapper[movie_id]
            content_scores += content_similarity[movie_idx]
    content_scores /= len(user_movies) if user_movies else 1

    # Combine Scores
    hybrid_scores = {
        movie_id: alpha * ncf_predictions.get(movie_id, 0) + (1 - alpha) * content_scores[movie_mapper[movie_id]]
        for movie_id in unrated_movies
    }

    sorted_movies = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=not reverse)
    return [movie_id for movie_id, _ in sorted_movies[:k]]

# Content-Based Recommendations
def get_content_based_recommendations(title_string, n_recommendations=10):
    try:
        idx = movies_df[movies_df['title'] == title_string].index[0]
        sim_scores = list(enumerate(content_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:(n_recommendations+1)]
        similar_movies = [i[0] for i in sim_scores]
        return [movies_df.iloc[movie]['title'] for movie in similar_movies]
    except IndexError:
        return []

# Explain Recommendations
def explain_recommendation(user_id, movie_id, alpha=0.7):
    user_highly_rated = train_data[(train_data['userId'] == user_id) & (train_data['rating'] >= 4)]['movieId'].tolist()
    user_highly_rated_titles = movies_df[movies_df['movieId'].isin(user_highly_rated)]['title'].tolist()
    
    user_highly_rated_genres = movies_df[movies_df['movieId'].isin(user_highly_rated)]['genres'].tolist()
    genre_flat_list = [genre for genres in user_highly_rated_genres for genre in genres]
    
    # Count dominant genres
    from collections import Counter
    genre_counts = Counter(genre_flat_list)
    dominant_genres = [genre for genre, count in genre_counts.most_common(3)]  # Top 3 genres
    
    # NCF Score
    ncf_score = predict_ratings(user_id, [movie_id])[movie_id]

    # Content Similarity Score
    content_scores = np.zeros(len(movie_mapper))
    for user_movie_id in user_highly_rated:
        if user_movie_id in movie_mapper:
            user_movie_idx = movie_mapper[user_movie_id]
            content_scores += content_similarity[user_movie_idx]
    content_score = content_scores[movie_mapper[movie_id]] / len(user_highly_rated) if user_highly_rated else 0

    # Hybrid Score
    hybrid_score = alpha * ncf_score + (1 - alpha) * content_score

    # Find similar movies for explanation
    similar_movies = [
        movie_inv_mapper[idx]
        for idx in np.argsort(-content_similarity[movie_mapper[movie_id]])[:10]
        if idx != movie_mapper[movie_id]
    ]
    similar_movie_titles = movies_df[movies_df['movieId'].isin(similar_movies)]['title'].tolist()

    # Movie metadata for explanation
    recommended_movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
    recommended_movie_genres = movies_df[movies_df['movieId'] == movie_id]['genres'].values[0]

    # Explanation
    explanation = {
        "Recommended Movie": recommended_movie_title,
        "Genres": recommended_movie_genres,
        "Hybrid Score": round(hybrid_score, 2),
        "NCF Contribution": round(alpha * ncf_score, 2),
        "Content Contribution": round((1 - alpha) * content_score, 2),
        "You Liked": user_highly_rated_titles[:10],  # Movies the user rated >= 4
        "Similar Movies to This Recommendation": similar_movie_titles,
        "Reasoning": f"You seem to enjoy movies in genres like {', '.join(dominant_genres)}. "
                     f"we recommend '{recommended_movie_title}'. This movie shares similar themes and is also popular "
                     f"among users with tastes like yours."
    }
    return explanation


def explain_disrecommendation(user_id, movie_id, alpha=0.7):
    # Get user's low-rated movies (< 3)
    user_low_rated = train_data[(train_data['userId'] == user_id) & (train_data['rating'] < 3)]['movieId'].tolist()
    user_low_rated_titles = movies_df[movies_df['movieId'].isin(user_low_rated)]['title'].tolist()

    # Collect genres for the low-rated movies
    user_low_rated_genres = movies_df[movies_df['movieId'].isin(user_low_rated)][['title', 'genres']].to_dict('records')

    # Get genres of the disrecommended movie
    disrecommended_movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
    disrecommended_movie_genres = movies_df[movies_df['movieId'] == movie_id]['genres'].values[0]

    # Find common genres with low-rated movies
    disliked_genres = set(
        genre for movie in user_low_rated_genres for genre in movie['genres']
    )
    common_genres = disliked_genres.intersection(disrecommended_movie_genres)

    # NCF score
    ncf_score = predict_ratings(user_id, [movie_id])[movie_id]

    # Content similarity score
    content_scores = np.zeros(len(movie_mapper))
    for user_movie_id in user_low_rated:
        if user_movie_id in movie_mapper:
            user_movie_idx = movie_mapper[user_movie_id]
            content_scores += content_similarity[user_movie_idx]
    content_score = content_scores[movie_mapper[movie_id]] / len(user_low_rated) if user_low_rated else 0

    # Hybrid Score
    hybrid_score = alpha * ncf_score + (1 - alpha) * content_score

    # Reasoning
    if content_score == 0:
        res = f"This movie, '{disrecommended_movie_title}', is different from your taste in general based on genres and content score."
    elif common_genres:
        res = f"The movie '{disrecommended_movie_title}' shares common genres ({', '.join(common_genres)}) with movies you've rated poorly."
    else:
        res = f"The movie '{disrecommended_movie_title}' is different from movies you've liked and aligns more with those you tend to dislike."

    # Explanation Output
    explanation = {
        "Disrecommended Movie": disrecommended_movie_title,
        "Genres": disrecommended_movie_genres,
        "Hybrid Score": round(hybrid_score, 2),
        "NCF Contribution": round(ncf_score, 2),
        "Content Contribution": round(content_score, 2),
        "You Disliked": user_low_rated_titles[:5],
        "Genres of Movies You Disliked": [
            f"'{movie['title']}' had genres: {movie['genres']}" for movie in user_low_rated_genres
        ],
        "Reasoning": res,
    }
    return explanation


def explainable_recommendation_with_disrecommendation(user_id, k=10, alpha=0.7):
    recommendations = hybrid_recommendation(user_id, k, alpha)
    disrecommendations = hybrid_recommendation(user_id, k, alpha, reverse=True)  # Use reverse logic for dis-recommendations

    explainable_recs = [explain_recommendation(user_id, movie_id, alpha) for movie_id in recommendations]
    explainable_disrecs = [explain_disrecommendation(user_id, movie_id, alpha) for movie_id in disrecommendations]

    return explainable_recs, explainable_disrecs


st.set_page_config(page_title="The Recommender", layout="centered")
st.title("🎬 The Recommender")

st.sidebar.header("Chat Options")
user_id = st.sidebar.number_input("Enter User ID:", min_value=1, value=1, step=1)

if "messages" not in st.session_state:
    st.session_state.messages = []  # Messages will store tuples like ("user", "message") or ("assistant", "message")

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.write(content)

# Chat Input
if user_input := st.chat_input("Type your message here..."):
    # Add user message to session state
    st.session_state.messages.append(("user", user_input))

    # Generate bot response based on user input
    response = ""
    if "recommend" in user_input.lower():
        # Generate recommendations with explanations
        recommendations = hybrid_recommendation(user_id, k=5)
        explanation_list = []
        for movie_id in recommendations:
            explanation = explain_recommendation(user_id, movie_id)
            explanation_list.append(
                f"- **{explanation['Recommended Movie']}**\n"
                f"  - Genres: {', '.join(explanation['Genres'])}\n"
                f"  - Score: {explanation['Hybrid Score']}\n"
                f"  - What others liked (70%, higher is better): {explanation['NCF Contribution']}\n"
                f"  - Similar to your tastes (30%, higher is better): {explanation['Content Contribution']}\n"
                f"  - Reason: {explanation['Reasoning']}\n"
            )
        response = f"Here are some movies I recommend for you:\n\n" + "\n".join(explanation_list)
    
    elif "avoid" in user_input.lower() or "not recommend" in user_input.lower():
        # Generate disrecommendations with explanations
        disrecommendations = hybrid_recommendation(user_id, k=5, reverse=True)
        explanation_list = []
        for movie_id in disrecommendations:
            explanation = explain_disrecommendation(user_id, movie_id)
            explanation_list.append(
                f"- **{explanation['Disrecommended Movie']}**\n"
                f"  - Genres: {', '.join(explanation['Genres'])}\n"
                f"  - Score: {explanation['Hybrid Score']}\n"
                f"  - What others liked (70%, higher is better): {explanation['NCF Contribution']}\n"
                f"  - Similar to your tastes (30%, higher is better): {explanation['Content Contribution']}\n"
                f"  - Reason: {explanation['Reasoning']}\n"
            )
        response = f"I suggest avoiding these movies:\n\n" + "\n".join(explanation_list)
    
    elif "i watched" in user_input.lower():
        # Generate content-based recommendations
        watched_title = user_input.split("i watched")[-1].strip()
        recommendations = get_content_based_recommendations(watched_title, n_recommendations=5)
        if recommendations:
            response = f"Because you watched '{watched_title}', you might like:\n- " + "\n- ".join(recommendations)
        else:
            response = f"Sorry, I couldn't find similar movies for '{watched_title}'. Try another title!"
    
    else:
        response = "I'm sorry, I didn't understand that. Try asking for recommendations, movies to avoid, or explanations!"

    # Add bot response to session state
    st.session_state.messages.append(("assistant", response))
    with st.chat_message("user"):
        st.write(user_input)
    # Display bot response
    with st.chat_message("assistant"):
        st.write(response)
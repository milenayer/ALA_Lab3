import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

def custom_svd(A):
    M_V = A.T @ A

    eigenvalues_V, V_matrix = np.linalg.eigh(M_V)

    sorted_indices = np.argsort(eigenvalues_V)[::-1]
    eigenvalues_V = eigenvalues_V[sorted_indices]
    V_matrix = V_matrix[:, sorted_indices]

    Vt = V_matrix.T

    singular_values = np.sqrt(np.maximum(eigenvalues_V, 0))
    m, n = A.shape
    Sigma = np.zeros((m, n))

    min_dim = min(m, n)
    Sigma[:min_dim, :min_dim] = np.diag(singular_values[:min_dim])

    U_cols = []

    tolerance = 1e-9
    rank = np.sum(singular_values > tolerance)

    for i in range(rank):
        sigma = singular_values[i]
        v_i = V_matrix[:, i]
        u_i = (A @ v_i) / sigma
        U_cols.append(u_i)

    if rank < m:
        if m > n:
            U_matrix = np.column_stack(U_cols)
        else:
            U_matrix = np.column_stack(U_cols)
    else:
        U_matrix = np.column_stack(U_cols)

    return U_matrix, Sigma, Vt

def verify_svd(A, U, Sigma, Vt):
    A_reconstructed = U @ Sigma @ Vt
    error = np.linalg.norm(A - A_reconstructed) / np.linalg.norm(A)

    print(f"Original matrix A:\n{A}")
    print("\nMatrix A reconstructed with SVD (U * Sigma * Vt):")
    print(np.round(A_reconstructed, 6))
    print(f"\nError: {error:.2e}")

    if error < 1e-9:
        print("Success")
    else:
        print("Error is too big")

A_test = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
], dtype=float)

# task 1
U_custom, Sigma_custom, Vt_custom = custom_svd(A_test)
verify_svd(A_test, U_custom, Sigma_custom, Vt_custom)

# task 2
df_ratings = pd.read_csv('ratings.csv')
df_movies = pd.read_csv('movies.csv')

ratings_matrix = df_ratings.pivot(index='userId', columns='movieId', values='rating')

print(f"Initial ratings matrix: {ratings_matrix.shape}")

thresh_users = 50
thresh_movies = 20

ratings_matrix = ratings_matrix.dropna(thresh=thresh_users, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=thresh_movies, axis=1)

print(f"Matrix after filtering: {ratings_matrix.shape}")

user_ids = ratings_matrix.index
movie_ids = ratings_matrix.columns

ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values

user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

print("Data prepared and demeaned.")

# part 1
k_viz = 3
U_viz, sigma_viz, Vt_viz = svds(R_demeaned, k=k_viz)
print(f"SVD performed with k={k_viz} for visualization.")

def plot_users(U_matrix, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    num_to_plot = min(20, U_matrix.shape[0])
    U_plot = U_matrix[:num_to_plot, :]

    ax.scatter(U_plot[:, 0], U_plot[:, 1], U_plot[:, 2], s=50)

    ax.set_title(title)
    ax.set_xlabel('Hidden Feature 1')
    ax.set_ylabel('Hidden Feature 2')
    ax.set_zlabel('Hidden Feature 3')
    plt.show()

plot_users(U_viz, f'User Similarity (k={k_viz})')

def plot_movies(Vt_matrix, title):
    V_matrix = Vt_matrix.T
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    num_to_plot = min(20, V_matrix.shape[0])
    V_plot = V_matrix[:num_to_plot, :]

    ax.scatter(V_plot[:, 0], V_plot[:, 1], V_plot[:, 2], s=50)

    ax.set_title(title)
    ax.set_xlabel('Hidden Feature 1')
    ax.set_ylabel('Hidden Feature 2')
    ax.set_zlabel('Hidden Feature 3')
    plt.show()

plot_movies(Vt_viz, f'Movie Similarity (k={k_viz})')

# part 2
k_pred = 3
U_pred, sigma_pred, Vt_pred = svds(R_demeaned, k=k_pred)

Sigma_diag = np.diag(sigma_pred)

all_user_predicted_ratings = U_pred @ Sigma_diag @ Vt_pred + user_ratings_mean.reshape(-1, 1)

preds_df = pd.DataFrame(all_user_predicted_ratings,
                        columns=movie_ids,
                        index=user_ids)

print(f"\nPrediction performed with k={k_pred}. Predicted DF shape: {preds_df.shape}")

only_predicted_df = preds_df.mask(ratings_matrix.notna())

print("'only_predicted_df' table created (existing ratings replaced with NaN).")

def get_movie_recommendations(user_id, num_recommendations=10):
    if user_id not in only_predicted_df.index:
        return f"User ID {user_id} is not present in the filtered dataset."

    user_predictions = only_predicted_df.loc[user_id].sort_values(ascending=False)

    top_movie_ids = user_predictions.head(num_recommendations).index.tolist()

    recommendations = df_movies[df_movies['movieId'].isin(top_movie_ids)].copy()

    recommendations.loc[:, 'Predicted_Rating'] = [
        user_predictions.loc[mid] for mid in recommendations['movieId']
    ]

    return recommendations[['title', 'genres', 'Predicted_Rating']].sort_values(by='Predicted_Rating',
                                                                                ascending=False).reset_index(drop=True)

target_user_id = 1

print(f"\nRecommendations for user ID={target_user_id} (k={k_pred})")
recommendations = get_movie_recommendations(target_user_id)
print(recommendations.to_markdown(index=True))

target_user_id_2 = 4
print(f"\nRecommendations for user ID={target_user_id_2} (k={k_pred})")
print(get_movie_recommendations(target_user_id_2).to_markdown(index=True))
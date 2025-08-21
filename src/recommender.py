import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, ratings_path, movies_path):
        # Load ratings
        self.ratings = pd.read_csv(
            ratings_path,
            sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"]
        )

        # Load movies
        movies = pd.read_csv(
            movies_path,
            sep="|",
            encoding="latin-1",
            header=None
        )
        movies = movies[[0, 1]]  # keep only movie_id and title
        movies.columns = ["movie_id", "title"]

        # Merge
        self.df = pd.merge(self.ratings, movies, on="movie_id")

        # Build similarity matrix
        self._build_similarity()

    def _build_similarity(self):
        user_movie_matrix = self.df.pivot_table(
            index="user_id",
            columns="title",
            values="rating"
        )
        user_movie_matrix = user_movie_matrix.fillna(0)

        similarity_matrix = cosine_similarity(user_movie_matrix.T)
        self.similarity_df = pd.DataFrame(
            similarity_matrix,
            index=user_movie_matrix.columns,
            columns=user_movie_matrix.columns
        )

    def recommend(self, movie_title, n=5):
        if movie_title not in self.similarity_df.columns:
            return f"❌ Movie '{movie_title}' not found in dataset."
        similar_scores = self.similarity_df[movie_title] \
            .sort_values(ascending=False)[1:n+1]
        return similar_scores.index.tolist()

##from src.recommender import MovieRecommender

# Initialize recommender with dataset paths
rec = MovieRecommender("data/u.data", "data/u.item")

# Get recommendations
print("🎬 Recommendations for Star Wars (1977):")
print(rec.recommend("Star Wars (1977)", 5))

import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.utils.math import cosine_similarity

from constants import EMBEDDING_MODEL

model_name = EMBEDDING_MODEL
model_kwargs = {"device": "cpu"}  # cuda
encode_kwargs = {"normalize_embeddings": True}  # helps with cosine similarity, euclidean distance
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


def embed_query(query):
    """
    Uses the embedding model to embed the query.
    .embed_query returns a list but numpy arrays are easier to work with
    reshaping because langchain wants a matrix
    """
    return np.array(embedding_model.embed_query(query)).reshape(1, -1)


building = embed_query("building")
print("Embedding for building:", building)  # embedding vector
print("Embedding dimension:", building.shape, "\n")  # embedding dimension, depends on the model

building = embed_query("building")
movie = embed_query("movie")
cinema = embed_query("cinema")

print("How similar are Building and Cinema ?", cosine_similarity(building, cinema))
print("How similar are Building + Movie and Cinema ?", cosine_similarity(building + movie, cinema), "\n")

review1 = embed_query("This movie is a masterpiece with brilliant performances.")
review2 = embed_query("A total waste of time, the plot was dull and predictable.")
positive = embed_query("positive review")
negative = embed_query("negative review")

# related concepts have high scores, so a positive and a negative movie review are positive because of the related topic
# otherwise you can use embeddings as input for a classifier
print("How similar are Review 1 and Review 2 ?", cosine_similarity(review1, review2), "\n")

# Compute and print cosine similarity for review1
similarity_positive_review1 = cosine_similarity(review1, positive)
similarity_negative_review1 = cosine_similarity(review1, negative)
print("How similar are Review 1 and Positive?", similarity_positive_review1)
print("How similar are Review 1 and Negative?", similarity_negative_review1, "\n")

# Compute and print cosine similarity for review2
similarity_positive_review2 = cosine_similarity(review2, positive)
similarity_negative_review2 = cosine_similarity(review2, negative)
print("How similar are Review 2 and Positive?", similarity_positive_review2)
print("How similar are Review 2 and Negative?", similarity_negative_review2)


# Lets play a game. Find similar equations to Building + Movie = Cinema, or King - Man + Woman = Queen, the person with the highest score wins
# your code below:

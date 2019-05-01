import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def exp1(movie_title):
    # Load a movie metadata dataset
    movie_metadata_keyword = pd.read_csv('merged_cleaned.csv')[['original_title', 'overview', 'vote_count']].set_index('original_title').dropna()
    movie_metadata_keyword = movie_metadata_keyword[movie_metadata_keyword['vote_count']>10]

    movie_metadata_keyword.drop('vote_count', axis=1)
    movie_metadata_keyword.shape



    # Create tf-idf matrix for text comparison
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movie_metadata_keyword['overview'].dropna())

    similarity = cosine_similarity(tfidf_matrix)
    similarity -= np.eye(similarity.shape[0])

    movie = movie_title
    n_plot = 10
    index = movie_metadata_keyword.reset_index(drop=True)[movie_metadata_keyword.index.str.startswith(movie)].index[0]

    # Final findings
    similar_movies_index = np.argsort(similarity[index])[::-1][:n_plot]
    similar_movies_score = np.sort(similarity[index])[::-1][:n_plot]
    similar_movie_titles = movie_metadata_keyword.iloc[similar_movies_index].index

    return similar_movies_score, similar_movie_titles

if __name__ == '__main__':
    exp1('Toy Story')
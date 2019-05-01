import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def exp2(movie_title):
    movie_metadata_keyword = pd.read_csv('merged_cleaned.csv', low_memory=True)[['original_title', 'keywords', 'genres', 'vote_count']].set_index('original_title').dropna()
    movie_metadata_keyword = movie_metadata_keyword[movie_metadata_keyword['vote_count']>10].drop('vote_count', axis=1)
    movie_metadata_keyword = movie_metadata_keyword[movie_metadata_keyword.astype(str)['keywords'] != '[]']

    for index, row in movie_metadata_keyword.iterrows():
        temp = row['keywords'].replace("'", '"')
        try:
            x = json.loads(temp)
        except :
            dirty_str = re.search(r"([\w]+['][\w]+)", row['keywords'])
            try:
                dirty_str = dirty_str.groups()[0]
                fix_str = dirty_str.replace("'", "")
                temp = temp.replace(dirty_str.replace("'", '"'), fix_str)
                x = json.loads(temp)
            except:
                pass

        try:
            y = json.loads(row['genres'].replace("'", '"'))
        except :
            continue

        keywords = []
        for item in x:
            if item.get('name'):
                keywords.append(item.get('name'))

        for item in y:
            if item.get('name'):
                keywords.append(item.get('name'))


        movie_metadata_keyword['keywords'][index] = ','.join(keywords)

    movie_metadata_keyword.drop('genres', axis=1)
    print(movie_metadata_keyword.shape)


    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movie_metadata_keyword['keywords'])


    # Cosine similarity between all movie-descriptions+genere
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
    similar_movies_score, similar_movie_titles = exp2('Toy Story')
    plt.barh(similar_movie_titles, similar_movies_score)
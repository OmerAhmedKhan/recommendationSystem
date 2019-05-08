import argparse
import pandas as pd
from surprise import KNNBaseline
from surprise import Dataset


def knn_experiment(movie_title):
    movie_items = pd.read_csv('item.csv')[['movie_id', 'title']].set_index('title').dropna()

    # Training KNN Dataset
    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    algo = KNNBaseline(sim_options=sim_options)
    algo.fit(trainset)

    # movie_id and title mapping
    row = movie_items.index.str.startswith(movie_title)
    try:
        raw_id = str(movie_items[row]['movie_id'].values[0])
    except:
        print('Movie not Found')
        return

    # Getting KNN id of the provided movie
    inner_id = algo.trainset.to_inner_iid(raw_id)

    # Get top 10 matched results
    neighbors = algo.get_neighbors(inner_id, k=10)
    neighbors = (algo.trainset.to_raw_iid(inner_id)
                           for inner_id in neighbors)

    neighbors_ids = [x for x in neighbors]
    for x in movie_items['movie_id']:
        if str(x) in neighbors_ids:
            print(movie_items[movie_items['movie_id'] == x].index.values[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ContentBased Recommendations for a movie')
    parser.add_argument('movie_title', type=str, help='title of a movie')

    args = parser.parse_args()

    knn_experiment(args.movie_title)
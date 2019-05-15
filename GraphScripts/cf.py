import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

import argparse
from surprise import KNNBaseline
from surprise import SVD, accuracy
from surprise import Reader, Dataset, KNNBasic, evaluate
from sklearn.neighbors import KNeighborsClassifier
from surprise.model_selection import cross_validate


#Dataset loading and spliting

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

users = pd.read_csv('u.user', sep='|', names=u_cols,
encoding='latin-1')
i_cols = ['movie_id', 'title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('u.item', sep='|', names=i_cols, encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

ratings = pd.read_csv('u.data', sep='\t', names=r_cols,
encoding='latin-1')

ratings = ratings.drop('timestamp', axis=1)

#Create train and test datasets
X = ratings.copy()
y = ratings['user_id']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state=42)




# Function that computes the root mean squared error (or RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def baseline(user_id, movie_id):
    return 2.5

# Function to compute the RMSE score obtained on the testing set by a model
def score(cf_model):
    # Construct a list of user-movie tuples from the testing dataset
    id_pairs = zip(X_test['user_id'], X_test['movie_id'])
    # Predict the rating for every user-movie tuple
    y_pred = np.array([cf_model(user, movie) for (user, movie) in
                       id_pairs])
    # Extract the actual ratings given by the users in the test data
    y_true = np.array(X_test['rating'])
    # Return the final RMSE score
    return rmse(y_true, y_pred)
baseline_score=score(baseline)
print("print the score for CF baseline ", baseline_score)


# Build the ratings matrix using pivot table function
r_matrix = X_train.pivot_table(values='rating', index='user_id', columns='movie_id')
# Each user is a row and each column is a movie
# print("r matrix: \n", r_matrix.head())
#Mean Ratings

def cf_user_mean(user_id, movie_id):
    if movie_id in r_matrix:
        mean_rating = r_matrix[movie_id].mean()
    else:
        mean_rating = 2.5

    return mean_rating
cf_user_mean_score = score(cf_user_mean)
print("print the score for CF mean", cf_user_mean_score)


# ####  Implementation of Weighted Mean  ####
# ratings by user u to movie m by using cosine score as our similarity function

r_matrix_dummy = r_matrix.copy().fillna(0)
cosine_sim = cosine_similarity(r_matrix_dummy, r_matrix_dummy)

cosine_sim = pd.DataFrame(cosine_sim, index=r_matrix.index, columns=r_matrix.index)
# print("cosine sim ", cosine_sim.head())


def cf_user_wmean(user_id,movie_id):
    # check that the movie exist in r_max
    if movie_id in r_matrix:
        sim_scores = cosine_sim[user_id]
        # Get the similarity scores for the user in question with every other user
        sim_scores = cosine_sim[user_id]
        # Get the user ratings for the movie in question
        m_ratings = r_matrix[movie_id]
        # Extract the indices containing NaN in the m_ratings series
        idx = m_ratings[m_ratings.isnull()].index
        # Drop the NaN values from the m_ratings Series
        m_ratings = m_ratings.dropna()
        # Drop the corresponding cosine scores from the sim_scores series
        sim_scores = sim_scores.drop(idx)
        # Compute the final weighted mean
        wmean_rating = np.dot(sim_scores, m_ratings)/ sim_scores.sum()
    else:
        # Default to a rating of 3.0 in the absence of any information
        wmean_rating = 2.5
    return wmean_rating

cf_user_wmean_score = score(cf_user_wmean)
print("print the score for CF weighted mean", score(cf_user_wmean))

#demographics

merged_df = pd.merge(X_train, users)


gender_mean = merged_df[['movie_id', 'sex', 'rating']].groupby(['movie_id', 'sex'])['rating'].mean()

users = users.set_index('user_id')

#__________________________________
#GENDER BASED CF
def gcf(uid, mid):
    if mid in r_matrix:
        gen = users.loc[uid]['sex']
        if gen in gender_mean[mid]:
            gender_rating = gender_mean[mid][gen]
        else:
            gender_rating = 2.5
    else:
        gender_rating = 2.5
    return gender_rating

gcf_score = score(gcf)
print("print the score for CF gender ", gcf_score)


age_mean = merged_df[['movie_id', 'age', 'rating']].groupby(['movie_id', 'age'])['rating'].mean()

#__________________________________
#AGE BASED CF
def agecf(uid, mid):
    if mid in r_matrix:
        age = users.loc[uid]['age']
        if age in age_mean[mid]:
            age_rating = age_mean[mid][age]
        else:
            age_rating = 2.5
    else:
        age_rating = 2.5
    return age_rating

agecf_score = score(agecf)
print("print the score for CF age ", agecf_score)

#__________________________________
# AGE & GENDER CF
age_gender_mean = merged_df[['age', 'rating', 'movie_id', 'sex']].pivot_table(values = 'rating', index = 'movie_id',
                                                                                 columns = ['sex', 'age'], aggfunc = 'mean')

def age_gender_cf(uid, mid):
    if mid in age_gender_mean.index:
        user = users.loc[uid]
        age = user['age']
        sex = user['sex']
        if sex in age_gender_mean.loc[mid]:
            if age in age_gender_mean.loc[mid][sex]:
                rating = age_gender_mean.loc[mid][sex][age]
                if np.isnan(rating):
                    rating = 2.5
                return rating
    return 2.5

age_gender_cf_score = score(age_gender_cf)
print("print the score for CF age and gender ", age_gender_cf_score)

#__________________________________
# GENDER & OCCUPATION CF
gen_occ_mean = merged_df[['sex', 'rating', 'movie_id', 'occupation']].pivot_table(values = 'rating', index = 'movie_id',
                                                                                 columns = ['occupation', 'sex'], aggfunc = 'mean')

def goc_cf(uid, mid):
    if mid in gen_occ_mean.index:
        user = users.loc[uid]
        gen = user['sex']
        job = user['occupation']
        if job in gen_occ_mean.loc[mid]:
            if gen in gen_occ_mean.loc[mid][job]:
                rating = gen_occ_mean.loc[mid][job][gen]
                if np.isnan(rating):
                    rating = 2.5
                return rating
    return 2.5

goc_cf_score = score(goc_cf)
print("print the score for CF age and occupation ", goc_cf_score)

#KNN
reader = Reader()
data = Dataset.load_from_df(ratings, reader)

sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

svd = SVD()

#evaluate(svd, data, measures=['RMSE'])
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()

algos = SVD()
algos.fit(trainset)
prediction = algos.test(testset)
print(prediction[:3])
print(accuracy.rmse(prediction, verbose=True))


#
# # Predict a certain item
# userid = str(196)
# itemid = str(302)
# actual_rating = 4
# print(svd.predict(userid, itemid, actual_rating))


def knn_experiment(movie_title):
    movie_items = pd.read_csv('item.csv')[['movie_id', 'title']].set_index('title').dropna()
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

movie_title = input("Please enter movier for recommendations  : ")
print("You entered: " + movie_title)
knn_experiment(movie_title)
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='ContentBased Recommendations for a movie')
#     parser.add_argument('movie_title', type=str, help='title of a movie')
#
#     args = parser.parse_args()
#
#     knn_experiment(args.movie_title)

scores = {'age_gender': age_gender_cf_score, 'age': agecf_score, 'gender_job': goc_cf_score, 'gender': gcf_score,
            'mean': cf_user_mean_score, 'w_mean': cf_user_wmean_score, 'baseline':baseline_score}

names = list(scores.keys())
values = list(scores.values())
for i in range(0,len(names)):
    plt.bar(i, values[i], tick_label = names[i])
plt.xticks(range(0,len(names)),names)
plt.ylabel("Score")
plt.xlabel("Collaborative Filters")
plt.title("Collaborative Filters score")
plt.show()

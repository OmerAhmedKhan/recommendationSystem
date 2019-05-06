import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

#load the user file
users = pd.read_csv('Phase 2/users.csv')
#print("here is the user dataset \n", users.head())
#print("Dimension of the users dataset: \n", users.shape)

#load the u.items file
movies = pd.read_csv('Phase 2/item.csv')
#cleaning the data we dont need
#movies = movies[['movie_id', 'title']]
#print("here is the movies dataset \n", movies.head())
#print("Dimension of the movies dataset: \n", movies.shape)

#load the u.data file
ratings = pd.read_csv('Phase 2/data.csv')
#ratings = ratings.drop('timestamp', axis=1)
#print("here is the ratings dataset \n", ratings.head())
#print("Dimension of the ratings dataset: \n", ratings.shape)


x = ratings.copy()
y = ratings['user_id']

x_train, x_test, y_train, t_test = train_test_split(x, y, test_size = 0.25, stratify=y, random_state=42)


#Function that computes the root mean squared error (or RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

#Function to compute the RMSE score obtained on the testing set by a model
def score(cf_model):
    #Construct a list of user-movie tuples from the testing dataset
    id_pairs = zip(x_test['user_id'], x_test['movie_id'])
    #Predict the rating for every user-movie tuple
    y_pred = np.array([cf_model(user, movie) for (user, movie) in
                       id_pairs])
    #Extract the actual ratings given by the users in the test data
    y_true = np.array(x_test['rating'])
    #Return the final RMSE score
    return rmse(y_true, y_pred)

#Build the ratings matrix using pivot table function
r_matrix = x_train.pivot_table(values='rating', index='user_id', columns='movie_id')
#Each user is a row and each column is a movie
#print("r matrix: \n", r_matrix.head())

##### Implementation of Weighted Mean  ####
# ratings by user u to movie m by using cosine score as our similarity function

r_matrix_dummy = r_matrix.copy().fillna(0)
cosine_sim = cosine_similarity(r_matrix_dummy, r_matrix_dummy)

cosine_sim = pd.DataFrame(cosine_sim, index=r_matrix.index, columns=r_matrix.index)
#print("cosine sim ", cosine_sim.head())

def cf_user_wmean(user_id, movie_id):
    #check that the movie exist in r_max
    if movie_id in r_matrix:
        sim_scores = cosine_sim[user_id]
        #Get the similarity scores for the user in question with every other user
        sim_scores = cosine_sim[user_id]
        #Get the user ratings for the movie in question
        m_ratings = r_matrix[movie_id]
        #Extract the indices containing NaN in the m_ratings series
        idx = m_ratings[m_ratings.isnull()].index
        #Drop the NaN values from the m_ratings Series
        m_ratings = m_ratings.dropna()
        #Drop the corresponding cosine scores from the sim_scores series
        sim_scores = sim_scores.drop(idx)
        #Compute the final weighted mean
        wmean_rating = np.dot(sim_scores, m_ratings)/ sim_scores.sum()
    else:
        #Default to a rating of 3.0 in the absence of any information
        wmean_rating = 3.0
        return wmean_rating


print("print the score for CF weighted mean", score(cf_user_wmean))


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error



u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

users = pd.read_csv('u.user', sep='|', names=u_cols,
 encoding='latin-1')
X_train = pd.read_csv('X_train.csv')
r_matrix = X_train.pivot_table(values='rating', index='user_id',
columns='movie_id')
X_test = pd.read_csv('X_test.csv')


item = pd.read_csv('item.csv')
item.drop(['Unnamed: 0', 'video release date'], axis = 1, inplace = True)


#Function that computes the root mean squared error (or RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def score(cf_model):

    #Construct a list of user-movie tuples from the testing dataset
    id_pairs = zip(X_test['user_id'], X_test['movie_id'])

    #Predict the rating for every user-movie tuple
    y_pred = np.array([cf_model(user, movie) for (user, movie) in id_pairs])

    #Extract the actual ratings given by the users in the test data
    y_true = np.array(X_test['rating'])

    #Return the final RMSE score
    return rmse(y_true, y_pred)


merged_df = pd.merge(X_train, users)
merged_df = merged_df.drop("Unnamed: 0", axis = 1)


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

scores = {'age_gender': age_gender_cf_score, 'age': agecf_score, 'gender_job': goc_cf_score, 'gender': gcf_score,
            'mean': 1.0233040312606156, 'w_mean': 1.0174483808407588}

names = list(scores.keys())
values = list(scores.values())
for i in range(0,len(names)):
    plt.bar(i, values[i], tick_label = names[i])
plt.xticks(range(0,len(names)),names)
plt.ylabel("Score")
plt.xlabel("Collaborative Filters")
plt.title("Collaborative Filters score")
plt.show()

import pandas as pd
import numpy as np
import warnings; warnings.simplefilter('ignore')
from ast import literal_eval

df1 = pd.read_csv("merged_cleaned.csv")
df2 = pd.read_csv("ratings_combined.csv")


df1.head()
#df2.head

df1['genres'] = df1['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

vote_counts=df1[df1['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages=df1[df1['vote_average'].notnull()]['vote_average'].astype('int')

# C is the mean vote across the whole report
C= vote_averages.mean()


# We will use 95th percentile as our cutoff. In other words,
# for a movie to feature in the charts, it must have more votes
# than at least 95% of the movies in the list.

m = vote_counts.quantile(0.94)
m


# Lets see how many movies are qualified according to what we have done so far.
qualified= df1[(df1['vote_count']>=m) & (df1['vote_count'].notnull()) &
               (df1['id'].notnull())]
qualified['vote_count']=qualified['vote_count'].astype('int')
qualified['vote_average']=qualified['vote_average'].astype('int')


qualified.shape

def weighted_rating(x):
    v=x['vote_count']
    R=x['vote_average']
    return(v/(v+m)*R)+(m/(m+v)*C)

qualified['wr'] = qualified.apply(weighted_rating, axis='columns')
qualified=qualified.sort_values('wr',ascending=False)
qualified.head(10)


s = df1.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = df1.drop('genres', axis=1).join(s)
s.shape

def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)


    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title',  'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)

    return qualified


##  THE following command is needed to return by genre ##
##  build_chart('Romance').head(10)    ##

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import networkx as nx
import math
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
# Used to calculate pairwise distance
from sklearn.metrics.pairwise import pairwise_distances


#Create users:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,encoding='latin-1') # --> (943, 5)

#Create ratings:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1') # --> (100000, 4)

#Creating items:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1') # --> (1682, 24)




x = ratings.copy()
y = ratings['user_id']

x_train, x_test, y_train, t_test = train_test_split(x, y, test_size = 0.25, stratify=y, random_state=42)

# Number of unique users
n_users = ratings.user_id.unique().shape[0]

# Number of unique movies
n_movies = ratings.movie_id.unique().shape[0]


# Create a user - item matrix
data_matrix = x_train.pivot_table(values='rating', index='user_id', columns='movie_id')
data_matrix_dummy = data_matrix.copy().fillna(0)
cosine_sim = cosine_similarity(data_matrix_dummy, data_matrix_dummy)


# User - User similarity in array with cosine similarity
user_similarity = cosine_sim
# Item - Item similarity in array with cosine similarity
item_similarity = cosine_sim


us_sim = user_similarity[:100, :100]



user_row = [i for i in range(0, len(us_sim))]
user_col = [i for i in range(0, len(us_sim))]


g = nx.Graph()
g.add_nodes_from(user_row)
g.add_nodes_from(user_col)

similarities = []
for i in user_row:
    for j in user_col:
        # if us_sim[i][j] > 0.2:
        #     similarities.append(us_sim[i][j])
        if not(math.isnan(us_sim[i][j])) and us_sim[i][j] > 0.2:
            g.add_edge(i, j, weight = us_sim[i][j])


nx.draw(g, node_size = 200, alpha = 0.7, node_color = 'red', node_shape = 'h', edge_color = 'black', style = 'dotted')
plt.show()

degrees = [val for (node, val) in g.degree()]
print(degrees)

# BETWEENESS
print("Betweenness")
b = nx.betweenness_centrality(g)
for v in g.nodes():
    print("%0.2d %5.3f" % (v, b[v]))

# DEGREE CENTRALITY
print("Degree centrality")
d = nx.degree_centrality(g)
for v in g.nodes():
    print("%0.2d %5.3f" % (v, d[v]))

# CLOSENESS CENTRALITY
print("Closeness centrality")
c = nx.closeness_centrality(g)
for v in g.nodes():
    print("%0.2d %5.3f" % (v, c[v]))


# *********************************************************************
# Attempt to make weighted graph, may come back for it later

# print(len(similarities))
# elarge = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] > 0.2]
# esmall = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] <= 0.2]
#
# pos = nx.spring_layout(g)
#
# nx.draw_networkx_edges(g, pos, edgelist=elarge, edge_color = 'black', width=6)
# nx.draw_networkx_edges(g, pos, edgelist=esmall, width=4, alpha=0.5, edge_color='green', style='dashed')
#
# print(list(g.node()))
# nx.draw(g, node_size = 200, alpha = 0.5, node_color = 'red', node_shape = 'h')
# **********************************************************************

# EIGENVALUE PLOT
L = nx.normalized_laplacian_matrix(g)
e = numpy.linalg.eigvals(L.A)
print("Largest eigenvalue:", max(e))
print("Smallest eigenvalue:", min(e))
plt.subplot(2, 1, 1)
plt.title("Eigenvalue Histogram")
plt.hist(e, bins=100)  # histogram with 100 bins
plt.xlim(0, 2)  # eigenvalues between 0 and 2
plt.xlabel("Eigenvalue")
plt.ylabel("Count")
plt.show()

# DEGREE PLOT
degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='b')
plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d for d in deg])
ax.set_xticklabels(deg)
plt.show()


# Clustering for graph
print("Clustering Coefficient")
print(nx.clustering(g))
print("Average Clustering Coefficient: ",nx.average_clustering(g))

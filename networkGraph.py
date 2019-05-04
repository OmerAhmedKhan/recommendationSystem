import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
from collections import defaultdict

df = pd.read_csv('g_cleaned.csv')


keyword = [word for word in df['keywords']] # --> list of sublists, each sublists contains the keywords
genre = [gen for gen in df['genres']] # --> list of stringsm each string is a genre
director = [dir for dir in df['director']] # --> list of strings, each string is a director
title = [tit for tit in df['title']] # --> list of strings, each string is a movie title

df.set_index('title', inplace = True) # --> change the index of the df to point to the title column instead of the id

titdir = dict(zip(title, director)) # --> Ψreate a dictionary, with title as key and director as value



g = nx.Graph() # --> create the graph

g.add_nodes_from(title, bipartite = 0) # --> create and add nodes from the title list with bipartite att = 0


# __________________________________________________________________
# FOR KEYWORDS

# kw = [] # --> empty list to append the keywords in
#
# for key in keyword:
#     kw.append(key.strip('\n').split(',')) # --> filling the empty list of keywords with sublists of keywords for every movie --> [[kw kw kw], [kw kw kw kw], ...]
#
# for k in kw[0]:
#     print((k.split(" ")[0])) # --> prints the first keyword of the first movie (kw[0])
# __________________________________________________________________



#__________________________________________________________________
# Creates the nodes for the genres
gen = ["comedy", "crime", "sciencefiction", "war", "drama", "romance", "horror",
        "thriller", "action", "adventure", "fantasy", "mystery", "animation",
        "family", "music", "history", "documentary", "western", "foreign",
        "tvmovie"]

g.add_nodes_from(gen, bipartite = 1) # --> create and add nodes from the director list, with bipartite att = 1

for tit in title:
    for ge in gen:
        if ge in df.loc[tit, 'genres']:
            g.add_edge(tit, ge)
            print(tit,": ", ge)
#___________________________________________________________________


# dd = defaultdict(set)
# for tit, dir in titdir.items():
#    dd[dir].add(tit)
# dd = {tit: dir for tit, dir in dd.items() if len(dir) > 1}
# print(dd)
"""
Used to check if different keys with same value exist
"""

#__________________________________________________________________
# Creates the nodes for the directors
# for dir in director:
#     for tit, dire in titdir.items(): # --> returns the movies with the same director
#          if dire == dir:
#             g.add_edge(tit, dir)
#             print(tit)
#g.add_nodes_from(director, bipartite = 1) # --> create and add nodes from the director list, with bipartite att = 1
#___________________________________________________________________
"""
Creates a bipartite graph for directors and movie titles
"""


g.remove_nodes_from(["Comforting Skin", "Way of Blue Sky"])

title_nodes, genre_nodes = bipartite.sets(g)

color_map = []

for node in g:
    if node in title_nodes:
        color_map.append('red')
    elif node in genre_nodes:
        color_map.append('green')

print(bipartite.density(g, genre_nodes))

nx.draw(g, node_color = color_map, with_labels = True)
plt.show()

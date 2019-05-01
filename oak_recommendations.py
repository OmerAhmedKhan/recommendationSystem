from Experiment1 import exp1
from Experiment2 import exp2


import argparse

parser = argparse.ArgumentParser(description='ContentBased Recommendations for a movie')
parser.add_argument('movie_title', type=str, help='title of a movie')

args = parser.parse_args()

exp1_score, exp1_title = exp1(args.movie_title)
print(exp1_title)

exp2_score, exp2_title = exp2(args.movie_title)
print(exp2_title)

final_results = []

for x in exp1_title:
    if x in list(exp2_title):
        final_results.append(x)
        
if len(final_results) < 5:
    for i in range(len(final_results), 5):
        if exp1_title[i] not in final_results:
            final_results.append(exp1_title[i])

for x in exp2_title[::-2]:
    if len(final_results) >= 10:
        break

    final_results.append(x)

fix_results = []
for x in final_results:
    if args.movie_title in x:
        fix_results.append(x)

for x in final_results:
    if x not in fix_results:
        fix_results.append(x)

print(fix_results)
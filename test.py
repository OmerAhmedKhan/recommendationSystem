x = {
    'Toy Story 3': 0.67,
    'Toy Story 2': 0.65,
    'Small Soldier': 0.61,
    'xyz': 0.44,
    'horror': 0.51,
}

y = {
    'Barbie': 0.7,
    'Toy story another': 0.67,
    'Lego': 0.6,
    'Child play': 0.52,
    'Toy': 0.59,
}


q = {}
w = {}
for key, value in x.items():
    if value >= 0.6:
        q.update({key: value*2*1.3})
    else:
        q.update({key: value * 1.3})

for key, value in y.items():
    if value >= 0.6:
        w.update({key: value * 2*1.2})
    else:
        w.update({key: value*1.2})

print(sorted(q.items(), key=lambda kv: kv[1]))
print(sorted(w.items(), key=lambda kv: kv[1]))
import numpy as np

def similarity(a, b):
    return 1 - (np.sum(np.abs(np.array(a) - np.array(b))) / len(a))

def retrieve_case(new_case, cases):
    best_score = -1
    best_case = None

    for c in cases:
        score = similarity(new_case, c["kriteria_numeric"])
        if score > best_score:
            best_score = score
            best_case = c

    return best_case, best_score

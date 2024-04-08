from frlearn.uncategorised import fuzzy_quantifiers as fuzz
import numpy as np


# Definition of the semi-fuzzy quantifier
def quantifier(sets):
    if len(sets[0]) == 0:
        return 1
    else:
        tab = set(sets[0])
        for el in sets[1:]:
            tab.intersection_update(el)
        return len(list(tab)) / len(sets)


a = 0.1
b = 0.7
# generalized fuzzy median of a and b
g_fuzzy_median = fuzz.gen_fuzzy_median(a, b)
print("Generalized fuzzy median: ", g_fuzzy_median)

# compute the top and the bottom
X = [1, 5, 7, 3]
fuzzy_sets = [[0.1, 0.5, 0, 0.9],
              [1, 0.2, 0.5, 0.7],
              [0.4, 0.5, 0.7, 1],
              [0.4, 0.5, 0.7, 1]
              ]
gamma = 0.4
top = fuzz.top(gamma, quantifier, X, fuzzy_sets)
bottom = fuzz.bottom(gamma, quantifier, X, fuzzy_sets)
print("Top: ", top, "Bottom: ", bottom)

# quantifier fuzzification mechanism OWA
n = 100
QFM = fuzz.QFM_OWA(X, quantifier, fuzzy_sets, n)
print("QFM: ", QFM)


# Evaluation of unary sentences of the form “RIM X’s are A’s”

# Definition of a RIM quantifier
def rim(p):
    return fuzz.some(p)


# Definition of the set X
sets = [i + 1 for i in range(10)]


# Definition of the fuzzy set A
def A(x):
    if x <= 2:
        return 0
    elif 2 < x <= 5:
        return (x - 2) / 10
    else:
        return 1


A_list = []
for x in sets:
    A_list.append(A(x))

# using Yager's model
eval_Yager = fuzz.y_eval_unary_sentence(sets, rim, A)
print("Evaluation of the unary sentence using Yager model: ", eval_Yager)

# using Zadeh’s model
eval_Zadeh = fuzz.z_eval_unary_sentence(sets, rim, A_list)
print("Evaluation using Zadeh model: ", eval_Zadeh)


# Evaluation of a binary sentence

# Define a new fuzzy set B
def B(x):
    if x <= 4:
        return x / 5
    elif 4 < x <= 8:
        return (x - 4) / 8
    else:
        return (x - 8) / 10


B_list = []
for x in sets:
    B_list.append(B(x))


# Define the implicator (Kleene-Dienes implicator)
def I(x):
    return max(1 - A(x), B(x))


# using Yager’s weighted implication-based binary quantification model
eval_binary_y = fuzz.y_eval_binary_sentence(sets, I, rim, A_list, B_list)
print("Evaluation of the binary sentence form with A and B using Yager model: ", eval_binary_y)

# using a fuzzy quantifier based a WOWA
eval_binary_w = fuzz.W_implicator(sets, I, rim, A_list, B_list)
print("Evaluation of the binary sentence form with A and B using a fuzzy quantifier based a WOWA: ", eval_binary_w)

# using Zadeh's model
eval_binary_z = fuzz.z_eval_binary_sentence(rim, A_list, B_list)
print("Evaluation of the binary sentence form with A and B using Zadeh' model: ", eval_binary_z)

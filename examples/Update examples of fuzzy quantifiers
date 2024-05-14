from frlearn.uncategorised import fuzzy_v as fuzz
from frlearn.uncategorised import quantifiers
import numpy as np

print(
    "We present some examples of the evaluation of unary and binary sentences using Zadeh's model, Yager's model and WOWA's model")
print(
    '***************************************************************************************************************************')
print()

# Definition of the RIM quantifiers some and most
some = quantifiers.QuadraticSigmoid(0.1, 0.4)
most = quantifiers.QuadraticSigmoid(0.3, 0.9)

# Definition of the base set
set_v = [i + 1 for i in range(10)]


# Definition of fuzzy sets
def A(x):
    if x <= 2:
        return 0
    elif 2 < x <= 5:
        return (x - 2) / 10
    else:
        return 1


A_list = [A(x) for x in set_v]


def B(x):
    if x <= 4:
        return (x) / 5
    elif 4 < x <= 8:
        return (x - 4) / 8
    else:
        return (x - 8) / 10


B_list = [B(x) for x in set_v]


# Definition of the Kleene-Dienes's implicator of A and B
def Kleene_Dienes(x):
    return max(1 - A(x), B(x))


Kleene_Dienes_list = [Kleene_Dienes(x) for x in set_v]

print(A_list, 'and ', B_list, 'are two fuzzy sets of the crisp set ', set_v)
print()

# Evaluation of unary sentences of the form “Λ X’s are A’s”, where Λ is a RIM quantifier,in Zadeh’s model and Yager's model
Zadeh_unary_evaluation = fuzz.unary_zadeh_model(most, A_list)
print("The evaluation of the unary sentence 'Most elements of", set_v, "are ", A_list, "' \n using Zadeh's model is ",Zadeh_unary_evaluation)
print()


# Yager's model
# Definition of the values of the monotone measure in the unary Yager's model
def mu_yager_unary(x):
    return some((len(x) / len(set_v)))


set_v_sorted_1 = [i for _, i in sorted(zip(A_list, set_v))]
mu_yager_unary_val = []
for i in range(len(set_v_sorted_1)):
    mu_yager_unary_val.append(mu_yager_unary(set_v_sorted_1[i:]))

Yager_unary_evaluation = fuzz.unary_yager_model(A_list, mu_yager_unary_val)
print("The evaluation of the unary sentence 'Most elements of", set_v, "are ", A_list, "' \n using Yager's model is ",
      Yager_unary_evaluation)
print()

# Evaluation of binary sentences of the form ”Λ 𝐴’s are 𝐵’s”, where Λ is a RIM quantifier

# Binary Zadeh's model
Zadeh_binary_evaluation = fuzz.binary_zadeh_model(most, A_list, B_list)
print("The evaluation of the binary sentence 'Most elements of", A_list, "are ", B_list, "' \n using Zadeh's model is ",
      Zadeh_unary_evaluation)
print()


# Binary Yager's model
# Define the values of the monotone measure for the binary Yager's model
def mu_yager(x):
    A_sorted = sorted(A_list)
    sum_v = 0
    for i in range(len(x)):
        sum_v += A_sorted[i]
    return most(sum_v / sum(A_list))


set_v_sorted_2 = [i for _, i in sorted(zip(Kleene_Dienes_list, set_v))]
mu_yager_val = []
for i in range(len(set_v_sorted_2)):
    mu_yager_val.append(mu_yager(set_v_sorted_2[i:]))

# Evaluation
Yager_binary_evaluation = fuzz.binary_yager_model(Kleene_Dienes_list, mu_yager_val)
print("The evaluation of the binary sentence 'Most elements of", A_list, "are ", B_list, "' \n using Yager's model is ",
      Yager_binary_evaluation)
print()


# WOWA's model
# Define the monotone measure for the binary WOWA's model
def mu_wowa(x):
    return some(sum([el for el in x if el in A_list]) / sum(A_list))


set_v_sorted_3 = [i for _, i in sorted(zip(Kleene_Dienes_list, set_v))]
mu_wowa_val = []
for i in range(len(set_v_sorted_3)):
    mu_wowa_val.append(mu_wowa(set_v_sorted_3[i:]))

# Evaluation
WOWA_binary_evaluation = fuzz.binary_WOWA_model(Kleene_Dienes_list, mu_wowa_val)
print("The evaluation of the binary sentence 'Most elements of", A_list, "are ", B_list, "' \n using WOWA's model is ",
      WOWA_binary_evaluation)


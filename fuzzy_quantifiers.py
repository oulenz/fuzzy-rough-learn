from itertools import chain, combinations


def choquet_integral(set_v, f, mu):
    """"
        Implementation of the Choquet_integral
        parameters:
        ----------
        set_v: table (list)
               The base set X on which the mathematics function f is defined
        f: function
           Define your own real-valued function f
        mu: function
            Define a monotone measure on set_v
        """
    set_sorted = sorted(set_v, key=lambda x: f(x))
    subsets = []
    integral = 0
    for i in range(len(set_sorted)):
        subsets.append([])
        subsets[i] = set_sorted[i:]
        if i == 0:
            integral = mu(subsets[0]) * f(set_sorted[0])
        else:
            integral += mu(subsets[i]) * (f(set_sorted[i]) - f(set_sorted[i - 1]))
    return integral


def more_than(k, p):
    """"
    Examples of RIM quantifiers which represent the quantifiers “more than 100 ∗ 𝑘%”
    parameters:
    ----------
    k, p: float
    ”"""
    if p > k:
        return 1
    else:
        return 0


def exists(p):
    return more_than(0, p)


def at_least(k, p):
    """"
        Examples of RIM quantifiers which represent the quantifiers “at least 100 ∗ 𝑘%”
        parameters:
        ----------
        k, p: float
        ”"""
    if p >= k:
        return 1
    else:
        return 0


def for_all(p):
    return at_least(1, p)


def zadeh_function(a, b, p):
    """"
    Zadeh’s S-function
    parameters: 
    ----------
    a, b, p: float such that 0 <= a<b <=1
    """
    if p <= a:
        return 0
    elif a <= p <= (a + b) / 2:
        return (2 * (p - a) ** 2) / (b - a) ** 2
    elif (a + b) / 2 <= p <= b:
        return 1 - ((2 * (p - b) ** 2) / (b - a) ** 2)
    else:
        return 1


def most(p):
    return zadeh_function(0.3, 0.9, p)


def some(p):
    return zadeh_function(0.1, 0.4, p)


def z_eval_unary_sentence(set_v, rim, A):
    """
    Evaluation of unary sentences of the form “Λ X’s are A’s” in Zadeh’s model, where Λ is a RIM quantifier
    parameters:
    ----------
    rim: function
        Definition of the RIM quantifier
    A: table (list)
       List of value of the Fuzzy set
    set_v: table (list)
           The base set X on which the fuzzy set is define
    """

    return rim(sum(A) / len(set_v))


def z_eval_binary_sentence(rim, A, B):
    """
    Evaluation of binary sentences of the form “Λ A’s are B’s” in Zadeh’s model, where Λ is a RIM quantifier
    parameters:
    ----------
    rim: function
         Definition of the RIM quantifier
    A, B: table (list)
          List of value of the Fuzzy sets
    """
    return rim(sum([el for el in A if el in B]) / sum(A))


def y_eval_unary_sentence(set_v, rim, A):
    """
        Evaluation of unary sentences of the form “Λ X’s are A’s” in Yager’s OWA model, where Λ is a RIM quantifier
        parameters:
        ----------
        rim: function
            Definition of the RIM quantifier
        A: function
           Function defining the fuzzy set A
        set_v: table (list)
               The base set X on which the fuzzy set is defined
        """
    def mu(x):
        return rim((len(x)/len(set_v)))
    return choquet_integral(set_v, A, mu)


def A_min(gamma, A, set_v):
    """"
    The sets A_min of Definition 2.17
    :parameters:
    -----------
    gamma: float
           0 <= gamma <= 1
    A: table (list)
       list of values of the fuzzy set A
    set_v: table (list)
           The base set X on which the fuzzy set is defined
    """
    val = []
    if gamma > 0:
        for i in range(len(set_v)):
            if A[i] >= 0.5 * (gamma + 1):
                val.append(set_v[i])
    elif gamma == 0:
        for i in range(len(set_v)):
            if A[i] > 0.5:
                val.append(set_v[i])
    return val


def A_max(gamma, A, set_v):
    """"
    The sets A_max of Definition 2.17
    parameters:
    -----------
    gamma: float
           0 <= gamma <= 1
    A: table (list)
       list of values of the fuzzy set A
    set_v: table (list)
           The base set X on which the fuzzy set is defined
    """
    val = []
    if gamma > 0:
        for i in range(len(set_v)):
            if A[i] > 0.5 * (1 - gamma):
                val.append(set_v[i])
    elif gamma == 0:
        for i in range(len(set_v)):
            if A[i] >= 0.5:
                val.append(set_v[i])
    return val


def three_valued_cut(gamma, A, set_v, ):
    """"The three-valued cut of 𝐴 at gamma"""
    return max(max(A_min(gamma, A, set_v), max(A_max(gamma, A, set_v))))


def gen_fuzzy_median(a, b):
    """"
    The generalized fuzzy median
    :parameters:
    ----------
    a, b: float
          0 <= a, b <= 1
    """
    if min(a, b) > 0.5:
        return min(a, b)
    elif max(a, b) < 0.5:
        return max(a, b)
    else:
        return 0.5


def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def complement(x, y):
    return [el for el in y if el not in x]


def combination_tables(tab):
    n = len(tab)
    combinaisons = []

    def gen_com(i, combinaison):
        if i == n:
            combinaisons.append(combinaison)
            return
        for element in tab[i]:
            new_combinaison = combinaison.copy()
            new_combinaison.append(element)
            gen_com(i + 1, new_combinaison)

    gen_com(0, [])
    return combinaisons


def top_bottom(gamma, semi_fuzzy_quantifier, set_v, fuzzy_sets):
    tab_A_min = []
    tab_A_max = []
    tab_B = []
    for i in range(len(set_v)):
        tab_A_min.append([])
        tab_A_max.append([])

        if gamma > 0:
            for j in range(len(set_v)):
                if fuzzy_sets[i][j] >= 0.5 * (gamma + 1):
                    tab_A_min[i].append(set_v[j])
                if fuzzy_sets[i][j] > 0.5:
                    tab_A_max[i].append(set_v[j])
        if gamma == 0:
            for j in range(len(set_v)):
                if fuzzy_sets[i][j] > 0.5:
                    tab_A_min[i].append(set_v[j])
                if fuzzy_sets[i][j] >= 0.5:
                    tab_A_max[i].append(set_v[j])
    complement_tab = []
    powerset_table = []
    for i in range(len(tab_A_max)):
        complement_tab.append(complement(tab_A_min[i], tab_A_max[i]))
    for i in range(len(complement_tab)):
        powerset_table.append(list(powerset(complement(tab_A_min[i], tab_A_max[i]))))
    tab_B_final = []
    for i in range(len(tab_A_min)):
        tab_B.append([])
        for element in tab_A_min[i]:
            tab_B[i].append(element)

    tab = []
    for i in range(len(powerset_table)):
        tab.append([])
        for j in powerset_table[i]:
            tab[i].append(j)
    tab_B_save = tab_B
    for i in range(len(tab_B)):
        tab_B_final.append([])
        for k in range(len(tab[i])):
            for j in tab[i][k]:
                tab_B[i].append(j)
            tab_B_final[i].append(tab_B[i])
    for i in range(len(tab_B_save)):
        tab_B_final[i].append(tab_B_save[i])
    tab_combination = combination_tables(tab_B_final)
    val = []
    for i in range(len(tab_combination)):
        val.append(semi_fuzzy_quantifier(tab_combination[i]))
    return max(val), min(val)


def top(gamma, quantifier, set_v, fuzzy_sets):
    """"
    Implementation of the top function
    parameters:
    ----------
    gamma: float
           0 <= gamma <= 1
    quantifier: function
                Define the semi-fuzzy quantifier function and use it as parameter
    set_v: table (list)
           The base set X on which the fuzzy set is defined
    fuzzy_sets: table of table (list containing lists)
                Define the value of each fuzzy sets in the same list
    """
    return top_bottom(gamma, quantifier, set_v, fuzzy_sets)[0]


def bottom(gamma, quantifier, set_v, fuzzy_sets):
    """"
        Implementation of the bottom function
        parameters:
        ----------
        gamma: float
               0 <= gamma <= 1
        quantifier: function
                    Define the semi-fuzzy quantifier function and use it as parameter
        set_v: table (list)
               The base set X on which the fuzzy set is defined
        fuzzy_sets: table of table (list containing lists)
                    Define the value of each fuzzy sets in the same list
        """
    return top_bottom(gamma, quantifier, set_v, fuzzy_sets)[1]


def QFM(gamma, semi_fuzzy_quantifier, set_v, fuzzy_sets):
    """"
    quantifier fuzzification mechanism
    parameters
    ----------
    gamma: float
               0 <= gamma <= 1
        quantifier: function
                    Define the semi-fuzzy quantifier function and use it as parameter
        set_v: table (list)
               The base set X on which the fuzzy set is defined
        fuzzy_sets: table of table (list containing lists)
                    Define the value of each fuzzy sets in the same list
                    :param fuzzy_sets:
                    :param set_v:
                    :param gamma:
                    :param semi_fuzzy_quantifier:
    """
    if bottom(gamma, semi_fuzzy_quantifier, set_v, fuzzy_sets) > 0.5:
        return bottom(gamma, semi_fuzzy_quantifier, set_v, fuzzy_sets)
    elif top(gamma, semi_fuzzy_quantifier, set_v, fuzzy_sets) < 0.5:
        return top(gamma, semi_fuzzy_quantifier, set_v, fuzzy_sets)
    else:
        return 0.5


def QFM_OWA(set_v, semi_fuzzy_quantifier, fuzzy_sets, n):
    """""
  The quantifier fuzzification mechanism OWA
    Parameters:
    ----------
    set_v: table (list)
           The base set X on which the fuzzy set is defined
    quantifier: function
                Define the semi-fuzzy quantifier function and use it as parameter
    fuzzy_sets: table of table (list containing lists)
                Define the value of each fuzzy sets in the same list
    n: int
       number of subdivisions of the interval [0,1]
    """
    h = 1 / n
    result = 0.5 * ((top(0, semi_fuzzy_quantifier, set_v, fuzzy_sets) +
                     bottom(0, semi_fuzzy_quantifier, set_v, fuzzy_sets)) / 2 +
                    (top(1, semi_fuzzy_quantifier, set_v, fuzzy_sets) +
                     bottom(1, semi_fuzzy_quantifier, set_v, fuzzy_sets)) / 2)
    for i in range(1, n):
        result += (top(i * h, semi_fuzzy_quantifier, set_v, fuzzy_sets) +
                   bottom(i * h, semi_fuzzy_quantifier, set_v, fuzzy_sets)) / 2
    result *= h
    return result


def Q_squared(rim, A, B):
    return rim(len([el for el in A if el in B]) / len(A))


def Q_arrow(rim, A, B, set_v):
    return rim((len([el for el in set_v if el not in A]) + len([el for el in A if el in B])) / len(set_v))


def W_implicator(set_v, f, rim, A, B=None):
    """"
    Evaluation of a binary sentence using a fuzzy quantifier based
    on a Weighted Ordered Weighted Averaging (WOWA)
    :parameter
    set_v: table (list)
           The base set X on which the fuzzy sets are defined
    f: function
       Define the Implicator
    rim: function
         define the rim quantifier
    A, B: table (list)
          List of the values of the fuzzy sets A and B
    """
    def mu(x):
        return rim(sum([el for el in x if el in A]) / sum(A))
    return choquet_integral(set_v, f, mu)


def y_eval_binary_sentence(set_v, f, rim, A, B=None):
    """"
      Evaluation of a binary sentence using Yager’s weighted implication-based binary quantification model
      :parameter
      set_v: table (list)
             The base set X on which the fuzzy sets are defined
      f: function
         Define the Implicator
      rim: function
           define the rim quantifier
      A, B: table (list)
            List of the values of the fuzzy sets A and B
      """
    def mu(x):
        A_sorted = sorted(A)
        sum_v = 0
        for i in range(len(x)):
            sum_v += A_sorted[i]
        return rim(sum_v / sum(A))
    return choquet_integral(set_v, f, mu)

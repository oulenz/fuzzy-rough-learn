"""
    Some fuzzy quantifiers.
    All the following functions are the implementations of definitions of some fuzzy quantifiers in [1].

    References
    ----------
    [1] THEERENS, Adnan et CORNELIS, Chris.
    Fuzzy rough sets based on fuzzy quantification.
    Fuzzy Sets and Systems, 2023, vol. 473, p. 108704.
"""
from frlearn.uncategorised import choquet_integral as ci

def unary_zadeh_model(rim, A):
    """
    Evaluation of unary sentences of the form “Λ X’s are A’s” in Zadeh’s model, where Λ is a RIM quantifier
    parameters:
    ----------
    rim: function
        Definition of the RIM quantifier
    A: table (list)
       List of value of the Fuzzy set
    """

    return rim(sum(A) / len(A))


def binary_zadeh_model(rim, A, B):
    """
    Evaluation of binary sentences of the form “Λ A’s are B’s” in Zadeh’s model, where Λ is a RIM quantifier
    parameters:
    ----------
    rim: function
         Definition of the RIM quantifier
    A, B: table (list)
          List of value of the Fuzzy sets
    """
    return rim(sum([min(a, b) for a, b in zip(A, B)]) / sum(A))


def unary_yager_model(rim, base_set, A):
    """
        Evaluation of unary sentences of the form “Λ X’s are A’s” in Yager’s OWA model, where Λ is a RIM quantifier
        parameters:
        ----------
        rim: function
            Define the RIM quantifier
        base_set: list
                  Define the base set on which the fuzzy set A is defined
        A: list
           Define the values of the fuzzy set A
        """

    def mu_yager_unary(x):
        return rim(len(x) / len(base_set))

    set_v_sorted_1 = [i for _, i in sorted(zip(A, base_set))]
    mu_yager_unary_val = []
    for i in range(len(set_v_sorted_1)):
        mu_yager_unary_val.append(mu_yager_unary(set_v_sorted_1[i:]))

    return ci.choquet_integral(A, mu_yager_unary_val)


def binary_yager_model(rim, impl, base_set, A, B=None):
    """
      Evaluation of a binary sentence using Yager’s weighted implication-based binary quantification model
      parameters:
      ----------
      rim: function
            Define the RIM quantifier

      impl: list
         Define the list of values of the Implicator of A and B

      base_set: list
                  Define the base set on which the fuzzy set A is defined
        A, B: list
           Define the values of the fuzzy sets A and B
    """

    def mu_yager_binary(x):
        A_sorted = sorted(A)
        sum_v = 0
        for i in range(len(x)):
            sum_v += A_sorted[i]
        return rim(sum_v / sum(A))

    set_v_sorted_2 = [i for _, i in sorted(zip(impl, base_set))]
    mu_yager_binary_val = []
    for i in range(len(set_v_sorted_2)):
        mu_yager_binary_val.append(mu_yager_binary(set_v_sorted_2[i:]))

    return ci.choquet_integral(impl, mu_yager_binary_val)


def binary_WOWA_model(rim, impl, base_set, A, B=None):
    """"
    Evaluation of a binary sentence using a fuzzy quantifier based
    on a Weighted Ordered Weighted Averaging (WOWA)
    parameters:
    ---------
      rim: function
            Define the RIM quantifier

      impl: list
         Define the list of values of the Implicator of A and B

      base_set: list
                  Define the base set on which the fuzzy set A is defined
        A, B: list
           Define the values of the fuzzy sets A and B
    """

    def mu_wowa(x):
        return rim(sum([el for el in x if el in A]) / sum(A))

    set_v_sorted_3 = [i for _, i in sorted(zip(impl, base_set))]
    mu_wowa_val = []
    for i in range(len(set_v_sorted_3)):
        mu_wowa_val.append(mu_wowa(set_v_sorted_3[i:]))
    return ci.choquet_integral(impl, mu_wowa_val)

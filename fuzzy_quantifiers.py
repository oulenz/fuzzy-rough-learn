"""
    Some fuzzy quantifiers.
    All the following functions are the implementations of defintions of some fuzzy quantifiers in [1].

    References
    ----------
    [1] THEERENS, Adnan et CORNELIS, Chris.
    Fuzzy rough sets based on fuzzy quantification.
    Fuzzy Sets and Systems, 2023, vol. 473, p. 108704.
"""


def choquet_integral(f, mu):
    """
        Implementation of the Choquet_integral
        parameters:
        ----------
        f: list
           Define a list that contains the values of the function
        mu: list
            Define a list that contains the values of the monotone measure.
    """
    n = len(mu)
    f_sorted = sorted(f)
    integral = 0
    for i in range(n):
        if i == 0:
            integral = mu[0] * f_sorted[0]
        else:
            integral += mu[i] * (f_sorted[i] - f_sorted[i - 1])
    return integral


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
    return rim(sum([el for el in A if el in B]) / sum(A))


def unary_yager_model(A, mu):
    """
        Evaluation of unary sentences of the form “Λ X’s are A’s” in Yager’s OWA model, where Λ is a RIM quantifier
        parameters:
        ----------
        rim: function
            Define the RIM quantifier
        A: list
           Define the values of the fuzzy set A

      mu: list
         Define the list of values the monotone measure
        """

    return choquet_integral(A, mu)


def binary_yager_model(impl, mu):
    """"
      Evaluation of a binary sentence using Yager’s weighted implication-based binary quantification model
      :parameter
      impl: list
         Define the list of values of the Implicator of A and B

      mu: list
         Define the list of values of the monotone measure mu
      """

    return choquet_integral(impl, mu)


def binary_WOWA_model(impl, mu):
    """"
    Evaluation of a binary sentence using a fuzzy quantifier based
    on a Weighted Ordered Weighted Averaging (WOWA)
    :parameter
    impl: list
         Define the list of values of the Implicator of A and B

      mu: list
         Define the list of values of the monotone measure mu
    """

    return choquet_integral(impl, mu)

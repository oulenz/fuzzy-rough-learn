"""
Implementation of the Choquet integral [1] in the most general way
References
    ----------
    [1] THEERENS, Adnan et CORNELIS, Chris.
    Fuzzy rough sets based on fuzzy quantification.
    Fuzzy Sets and Systems, 2023, vol. 473, p. 108704.
"""

def choquet_integral(f, mu):
    """
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

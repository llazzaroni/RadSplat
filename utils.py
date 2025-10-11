import numpy as np

def distribute_values(total : int, n_groups : int) -> np.ndarray:
    """
    This function is used to distribut an 'amount' total among n different groups

    :Param total -> (int) the amount to distribute
    :Param n_groups -> (int) the number of groups to distribute the amount among

    :Return (np.ndarray) the array of the values distributed
    """

    random_values = np.random.rand(n_groups)
    random_values /= random_values.sum()

    return random_values * total

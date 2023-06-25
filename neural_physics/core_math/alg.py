import numpy as np

def least_squares(A, b):
    """
    Calculate the solution of the least squares problem min (||b-Ax||)
    @param A: the Matrix A in the equation (n x m) where n is the number of data points m the DOF
    @param b: the Matrix b in the equation (m x 1)
    """
    return np.linalg.lstsq(A, b, rcond=None)

""" Functions to solve orbiting bodies problems.
Written by: Jeremie X. J. Bannwarth
"""
import numpy as np
from orbitutils.solvers import rkf45

def two_body_3d_rates(t, Y, m1=1., m2=.1):
    """Find the state derivatives for the two body problem in 3D.

    Parameters
    ----------
    t : float
        Time to evaluate the derivatives at.
    Y : numpy.array
        State vector.
    m1 : float
        Mass of the first body (kg).
    m2 : float
        Mass of the second body (kg).

    Returns
    -------
    F : numpy.array
        Array of state derivatives.
    """
    # Extract vectors from state vector
    R1 = Y[0:3]
    R2 = Y[3:6]
    V1 = Y[6:9]
    V2 = Y[9:12]
    
    # Constants
    G = 6.64742e-11
    
    # Position vector of m2 relative to m1
    R = R2 - R1
    r = np.linalg.norm(R)
    
    # Compute derivatives
    F = np.zeros(Y.shape)
    F[0:3] = V1 # dR1/dt = V1
    F[3:6] = V2 # dR2/dt = V2
    F[6:9] = G*m2*R/r**3
    F[9:12] = - G*m1*R/r**3
    
    return F


def two_body_3d(R1_0, R2_0, V1_0, V2_0, m1, m2, tSpan=np.array([0., 10.0])):
    """ Compute the position and velocity of two bodies in 3D over time.

    Parameters
    ----------
    R1_0 : numpy.array
        Initial position of the first body.
    R2_0 : numpy.array
        Initial position of the second body.
    V1_0 : numpy.array
        Initial velocity of the first body.
    V2_0 : numpy.array
        Initial velocity of the second body.
    m1 : float
        Mass of the first body (kg).
    m2 : float
        Mass of the second body (kg).
    tSpan : numpy.array
        Range of times to solve for.

    Returns
    -------
    ys : numpy.array
        State time response.
    ts : numpy.array
        Time vector.
    """
    Y0 = np.concatenate((R1_0, R2_0, V1_0, V2_0))
    
    # Create anonymous function to pass m1 and m2
    rates = lambda t, Y: two_body_3d_rates(t, Y, m1, m2)
    
    ys, ts = rkf45(rates, Y0, tSpan)
    
    return (ys, ts)
    
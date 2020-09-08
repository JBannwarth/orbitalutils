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
    F[0:3] = V1  # dR1/dt = V1
    F[3:6] = V2  # dR2/dt = V2
    F[6:9] = G * m2 * R / r**3
    F[9:12] = - G * m1 * R / r**3

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
    def rates(t, Y): return two_body_3d_rates(t, Y, m1, m2)

    ys, ts = rkf45(rates, Y0, tSpan)

    return (ys, ts)


def n_body_3d_rates(t, Y, M):
    """Find the state derivatives for the N body problem in 3D.

    Parameters
    ----------
    t : float
        Time to evaluate the derivatives at.
    Y : numpy.array
        State vector.
    M : numpy.array
        Array of the masses of the N bodies (kg).

    Returns
    -------
    F : numpy.array
        Array of state derivatives.
    """
    # Extract vectors from state vector
    n = M.shape[0]

    # Store the vectors for each mass in a different column
    R = np.reshape(Y[0:n * 3], (3, n), order='F')
    V = np.reshape(Y[n * 3:], (3, n), order='F')

    # Constants
    G = 6.67259e-11

    # Find acceleration
    A = np.zeros(V.shape)
    for m in range(n):
        R_m = R[:, m]
        R_other = np.delete(R, m, 1) - np.reshape(R_m, (3, 1))
        r_other = np.linalg.norm(R_other, axis=0)
        M_other = np.delete(M, m, 0)
        A[:, m] = np.sum(G * M_other * R_other / r_other**3, axis=1)

    # Assign the rates
    F = np.concatenate((np.reshape(V, (n * 3,), order='F'),
                        np.reshape(A, (n * 3,), order='F')))
    return F


def n_body_3d(R_0, V_0, M, tSpan=np.array([0., 10.0])):
    """ Compute the position and velocity of N bodies in 3D over time.

    Parameters
    ----------
    R_0 : numpy.array
        Vector of initial positions of the N bodies in the form
        [x1 y1 z1 x2 y2 z2 ...]
    V_0 : numpy.array
        Vector of initial velocities of the N bodies in the form
        [vx1 vy1 vz1 vx2 vy2 vz2 ...]
    M : float
        Vector of masses (kg) in the form
        [m1 m2 m3 ...]
    tSpan : numpy.array
        Range of times to solve for.

    Returns
    -------
    ys : numpy.array
        State time response.
    ts : numpy.array
        Time vector.
    """
    n = M.shape[0]
    Y0 = np.concatenate((np.reshape(R_0, (n * 3,), order='F'),
                         np.reshape(V_0, (n * 3,), order='F')))

    # Create anonymous function to pass m1 and m2
    def rates(t, Y): return n_body_3d_rates(t, Y, M)

    ys, ts = rkf45(rates, Y0, tSpan)

    return (ys, ts)


def orbit_rates(t, Y, m1, m2):
    """Find the state derivatives for the relative orbit problem.

    m1 has a non-rotating cartesian coordinate frame.

    Parameters
    ----------
    t : float
        Time to evaluate the derivatives at.
    Y : numpy.array
        State vector (km or km/s).
    m1 : float
        Mass of body 1 (kg).
    m2 : float
        Mass of body 2 (kg).

    Returns
    -------
    F : numpy.array
        Array of state derivatives.
    """
    # Store the vectors for each mass in a different column
    R = Y[0:3]
    RDot = Y[3:6]

    # Constants
    G = 6.67259e-20

    # Find acceleration
    mu = G * (m1 + m2)
    r = np.linalg.norm(R)
    RDDot = - R * mu / r**3

    # Assign the rates
    F = np.concatenate((RDot, RDDot))
    return F


def orbit(R_0, V_0, M, tSpan):
    """ Compute the position and velocity of m1 relative to m2.

    m1 has a non-rotating cartesian coordinate frame.

    Parameters
    ----------
    R_0 : numpy.array
        Initial position of m1 relative to m2 (km).
    V_0 : numpy.array
        Initial velocity of m1 relative to m2 (km/s).
    M : numpy.array
        Vector of masses (kg) in the form [m1 m2].
    tSpan : numpy.array
        Range of times to solve for.

    Returns
    -------
    ys : numpy.array
        State time response.
    ts : numpy.array
        Time vector.
    """
    Y_0 = np.concatenate((R_0, V_0))

    # Create anonymous function to pass m1 and m2
    def rates(t, Y): return orbit_rates(t, Y, M[0], M[1])

    ys, ts = rkf45(rates, Y_0, tSpan)

    return (ys, ts)

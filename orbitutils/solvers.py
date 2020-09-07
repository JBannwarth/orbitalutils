""" Implementation of numerical solvers.
Based on Chapter 1 of
    Orbital Mechanics for Engineering Students (4th Edition) by Howard Curtis.
Written by: J.X.J. Bannwarth
"""
import numpy as np
from copy import deepcopy


def rk14_step(y, t, f, h, order=4):
    """Use Runge-Kutta algorithm to find y(t+h).

    Parameters
    ----------
    y : numpy.array
        State at time t.
    t : float
        Time y is evaluated at.
    f : function
        Function to compute the derivative of y.
    h : float
        (Fixed) time-step.
    order : int, optional
        Order of the Runge-Kutta algorithm used.

    Returns
    -------
    yN : numpy.array
        Estimated state at time t+h.
    """
    A = {1: [0.0], 2: [0.0, 1.0], 3: [0.0, 0.5, 1.0], 4: [0.0, 0.5, 0.5, 1.0]}
    B = {
        1: None,
        2: [[0.0], [1.0]],
        3: [[0.0, 0.0], [0.5, 0.0], [-1.0, 2.0]],
        4: [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
    }
    C = {
        1: [1.0],
        2: [0.5, 0.5],
        3: [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6],
        4: [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6],
    }
    A = np.array(A.get(order, "Invalid order"))
    B = np.array(B.get(order, "Invalid order"))
    C = np.array(C.get(order, "Invalid order"))

    tTilde = t + A * h

    fTilde = np.zeros((order, y.shape[0]))
    yTilde = np.zeros((order, y.shape[0]))

    # Stage 1
    fTilde[0, :] = f(t, y)

    # Stage 2+
    for m in range(1, fTilde.shape[0]):
        yTilde[m, :] = y + h * np.dot(B[m, 0:m], fTilde[0:m, :])
        fTilde[m, :] = f(tTilde[m], yTilde[m, :])

    Phi = np.dot(C, fTilde)
    yN = y + h * Phi
    return yN


def rk14(f, y0, tMax=10.0, h=0.01, order=4):
    """Use Runge-Kutta algorithm to numerically solve ODE.

    Parameters
    ----------
    f : function
        Function to compute the derivative of y.
    y0 : numpy.array
        Initial value of state.
    tMax : float
        Time to solve until.
    h : float
        (Fixed) time-step.
    order : int, optional
        Order of the Runge-Kutta algorithm used.

    Returns
    -------
    ys : numpy.array
        State time response.
    ts : numpy.array
        Time vector
    """
    # Assign output vectors
    ts = np.arange(0.0, tMax, h)
    ys = np.zeros((ts.shape[0], y0.shape[0]))

    # Initial conditions if non-zero
    ys[0, :] = y0

    # Solve for each time-step
    for n in range(1, ts.shape[0]):
        ys[n, :] = rk14_step(ys[n - 1, :], ts[n - 1], f, h, order)

    return (ys, ts)


def rkf45(f, y0, tSpan=np.array([0., 10.0]), tol=1.e-8):
    """Use variable step-size Runge-Kutta algorithm to numerically solve ODE.

    Parameters
    ----------
    f : function
        Function to compute the derivative of y.
    y0 : numpy.array
        Initial value of state.
    tSpan : numpy.array
        Span of time to solve for.
    tol : float
        Allowable truncation error.

    Returns
    -------
    ys : numpy.array
        State time response.
    ts : numpy.array
        Time vector
    """
    # Constants
    A = np.array([0., 1./4., 3./8., 12./13., 1., 1./2.])
    B = np.array([[0., 0., 0., 0., 0.],
                  [1./4., 0., 0., 0., 0.],
                  [3./32., 9/32., 0., 0., 0.],
                  [1932./2197., -7200./2197., 7296./2197., 0., 0.],
                  [439./216., -8., 3680./513., -845./4104., 0.],
                  [-8./27., 2., -3544./2565., 1859./4104., -11./40.]])
    C4 = np.array([25./216., 0., 1408./2565., 2197./4104., -1./5., 0.])
    C5 = np.array([16./135., 0., 6656./12825., 28561./56430., -9./50., 2./55.])

    nStates = y0.shape[0]
    t0 = tSpan[0]
    tMax = tSpan[1]

    # Solve for each time-step
    ts = [t0]
    ys = [deepcopy(y0)]
    h = (tMax - t0)/100.

    t = deepcopy(ts[-1])
    y = deepcopy(ys[-1])
    while ts[-1] < tMax:
        tTilde = t + A * h
        fTilde = np.zeros((6, nStates))
        yTilde = np.zeros((6, nStates))

        # Stage 1
        fTilde[0, :] = f(t, y)

        # Stage 2+
        for m in range(1, fTilde.shape[0]):
            yTilde[m, :] = y + h * np.dot(B[m, 0:m], fTilde[0:m, :])
            fTilde[m, :] = f(tTilde[m], yTilde[m, :])

        Phi4 = np.dot(C4, fTilde)
        Phi5 = np.dot(C5, fTilde)
        yN4 = y + h * Phi4
        yN5 = y + h * Phi5

        # Truncation error
        err = np.max(np.abs(yN5 - yN4))

        # Maximum truncation error
        yMax = np.max(np.abs(y))
        errAllow = tol*np.max([yMax, 1.0])

        # Change in step size
        delta = (errAllow/(err+np.finfo(float).eps))**0.2

        # Save if error is in bounds
        if err <= errAllow:
            h = np.min([h, tMax-t])
            t += h
            y += h*Phi5
            ts.append(deepcopy(t))
            ys.append(deepcopy(y))

        # Update time step
        h = np.min([delta*h, 4.*h])
    ts = np.array(ts)
    ys = np.array(ys)
    return (ys, ts)


def heun(f, y0, tMax=10.0, h=0.01, tol=1.e-6, iterMax=100):
    """Use Heun's method to numerically solve ODE.

    Parameters
    ----------
    f : function
        Function to compute the derivative of y.
    y0 : numpy.array
        Initial value of state.
    tMax : float
        Time to solve until.
    h : float
        (Fixed) time-step.
    tol : float
        Tolerance between y_n+1 and y*.
    iterMax : int
        Maximum number of iterations for a single time-step.

    Returns
    -------
    ys : numpy.array
        State time response.
    ts : numpy.array
        Time vector
    """
    # Assign output vectors
    ts = np.arange(0.0, tMax, h)
    ys = np.zeros((ts.shape[0], y0.shape[0]))

    # Initial conditions if non-zero
    ys[0, :] = y0

    # Solve for each time-step
    for n in range(1, ts.shape[0]):
        d = np.inf
        y1 = ys[n-1, :]

        f1 = f(ts[n-1], y1)
        y2 = y1 + h*f1
        idx = 0
        while d > tol and idx < iterMax:
            y2p = y2
            f2 = f(ts[n], y2p)
            y2 = y1 + h*(f1+f2)/2.

            # Find max error
            d = np.max(np.abs((y2-y2p)/(y2+np.finfo(float).eps)))
            idx += 1
        ys[n, :] = y2p

    return (ys, ts)    

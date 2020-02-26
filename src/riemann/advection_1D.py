import numpy as np


def advection_1D(q_l, q_r, problem_data):
    r"""
    Gudnov upwind solver in 1d for x.
    """
    # Solver riemann problem for each (v1, v2)
    # num_waves = q_l.shape[1]
    # Return values
    wave = np.empty(q_l.shape)
    s = np.empty((q_l.shape[1:]))
    amdq = np.zeros(q_l.shape)
    apdq = np.zeros(q_l.shape)

    wave[:] = q_r - q_l
    s[:] = problem_data["v1"]
    apdq[:] = np.maximum(s, 0.0) * wave
    amdq[:] = np.minimum(s, 0.0) * wave

    return wave, s, amdq, apdq

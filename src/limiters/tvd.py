import numpy as np

mimod = 1

def limit(wave, s, limiter, dtdx):
    wave_norm2 = wave**2
    wave_zero_mask = np.array((wave == 0), dtype=float)
    wave_nonzero_mask = (1.0 - wave_zero_mask)

    dotls = wave[1:]*wave[:-1]
    spos = np.array(s > 0.0, dtype=float)

    r = np.ma.array((spos*dotls[:-1] + (1-spos)*dotls[1:]))
    r /= np.ma.array(wave_norm2[1:-1])

    r.fill_value = 0
    r = r.filled()

    
    cfl = np.abs(s*dtdx)
    wlimitr = limiter(r, cfl)
    wave[1:-1] = wave[1:-1]*wave_zero_mask[1:-1] + wlimitr * wave[1:-1]*wave_nonzero_mask[1:-1]

    return wave

def minmod(r, cfl):
    r"""
    Minmod vectorized limiter
    """
    a = np.ones((2,)+r.shape)
    b = np.zeros((2,)+r.shape)

    a[1,:] = r
    b[1,:] = np.minimum(a[0], a[1])

    return np.maximum(b[0], b[1])
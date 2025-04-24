#PKGS
##########################################################################################
##########################################################################################
##########################################################################################
# Import Libraries
# Standard libraries
import warnings

# Data manipulation and analysis
import numpy as np
import random
# Plotting and visualization
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
# Ignore warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='divide by zero encountered in divide')

##########################################################################################
##########################################################################################
##########################################################################################
def _calculate_synthetic_data(t, c, sc, num_simulations, *, seed=None,
                              cutoff_s=5000,
                              butter_order=1):
    '''
   """
    Generate synthetic light curves preserving only the noise characteristics
    of an input time series by randomizing residuals.

    The input residuals are obtained by high‐pass filtering (Butterworth) with
    reflected padding to avoid edge artifacts. Extreme outliers in the residuals
    (|z| > 3) are replaced by samples drawn from the “safe” pool (|z| < 1).
    The procedure is repeated for `num_simulations` by shuffling both the
    residuals and the time differences, producing flattened arrays of
    time, counts, and count errors.

    Parameters
    ----------
    t : array_like, shape (n,)
        Time stamps of the original light curve.
    c : array_like, shape (n,)
        Count rates (or fluxes) corresponding to each time stamp.
    sc : array_like, shape (n,)
        1-sigma uncertainties on the counts `c`.
    num_simulations : int
        Number of synthetic light curve realizations to generate.
    
    Keyword Arguments
    -----------------
    seed : int or array_like or None, optional
        Seed or RNG state for reproducible shuffling.  Defaults to `None`.
    cutoff_s : float, default 5000
        Cut‐off period (in the same units as `t`) for the high‐pass Butterworth
        filter; wavelengths longer than this are removed.
    butter_order : int, default 1
        Order of the high‐pass Butterworth filter.

    Returns
    -------
    tsim : ndarray, shape (n * num_simulations,)
        Concatenated time arrays for each simulation.  Each block of length `n`
        starts at the original `t[0]` plus the preceding simulation’s total
        duration, with shuffled time differences.
    simc : ndarray, shape (n * num_simulations,)
        Synthetic count rates: the shuffled, filtered residuals shifted to have
        the same mean as the original `c`.
    ssimc : ndarray, shape (n * num_simulations,)
        Synthetic 1-sigma count uncertainties, computed from the shuffled
        relative errors and post‐filter counts, with outliers (|z| > 3)
        similarly replaced by “safe” values.
    '''
    rng = np.random.default_rng(seed)

    # --- Basic setup ---------------------------------------------------
    t   = np.asarray(t, dtype=np.float64)
    c   = np.asarray(c, dtype=np.float64)
    sc  = np.asarray(sc, dtype=np.float64)
    n   = c.size

    dt  = np.median(np.diff(t))
    fs  = 1.0 / dt
    fc  = 1.0 / cutoff_s
    wn  = fc / (fs / 2)
    b, a = signal.butter(butter_order, wn, btype='high')

    # --- Reflected padding to prevent boundary artifacts ---------------
    pad_len = cutoff_s

    c_padded = np.pad(c, pad_width=pad_len, mode='reflect')
    resid_padded = signal.filtfilt(b, a, c_padded)
    resid = resid_padded[pad_len:-pad_len]
    
    # --- Z-score filtering to remove outliers in residuals ------------
    z_resid = (resid - np.mean(resid)) / (np.std(resid) + 1e-10)
    outlier_mask = np.abs(z_resid) > 3
    safe_mask    = np.abs(z_resid) < 1

    if np.any(outlier_mask) and np.any(safe_mask):
        safe_vals = resid[safe_mask]
        replace_vals = rng.choice(safe_vals, size=outlier_mask.sum(), replace=True)
        resid[outlier_mask] = replace_vals

    # --- Normalize errors ----------------------------------------------
    sc_prop = np.divide(sc, c, out=np.zeros_like(c), where=c != 0)
    
    # --- Generate simulations ------------------------------------------

    n = len(t)
    duration = t.max() - t.min()
    total   = n * num_simulations

    tsim   = np.zeros(total)
    simc   = np.zeros(total)
    ssimc  = np.zeros(total)

    mean_diff_t = np.mean(np.diff(t))

    base_time = t[0]

    for k in range(num_simulations):
        perm = rng.permutation(n)
        start = k * n
        end   = start + n

        # shuffle your diffs
        mixed_diffs    = np.concatenate([[mean_diff_t], np.diff(t)])
        shuffled_diffs = mixed_diffs[perm]

        # fill the k-th block
        tsim[start]       = base_time + k * duration
        tsim[start:end]   = tsim[start] + np.cumsum(shuffled_diffs)
        simc[start:end]   = resid[perm] + np.mean(c)
        ssimc[start:end]  = abs(sc_prop[perm] * simc[start:end])

        # --- Z‑score filtering on ssimc to remove outliers ----------------
        z_ssimc = (ssimc - np.mean(ssimc)) / (np.std(ssimc) + 1e-10)
        outlier_mask = np.abs(z_ssimc) > 3
        safe_mask    = np.abs(z_ssimc) < 1

        if np.any(outlier_mask) and np.any(safe_mask):
            # sample replacement values from the “safe” pool
            safe_vals = ssimc[safe_mask]
            replace_vals = rng.choice(safe_vals,
                                    size=outlier_mask.sum(),
                                    replace=True)
            ssimc[outlier_mask] = replace_vals
        
    
    # --- Output flattened ---------------------------------------------
    return tsim, simc, ssimc
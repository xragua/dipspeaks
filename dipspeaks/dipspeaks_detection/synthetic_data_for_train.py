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
    Generate `num_simulations` synthetic light curves that keep only the
    noise characteristics of the original curve (any real peaks/dips are
    destroyed by shuffling the residuals).

    Uses reflected padding to prevent edge artifacts in the trend.

    Values in the residuals with |zscore| > 3 are replaced with random values
    drawn from |zscore| < 1.
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
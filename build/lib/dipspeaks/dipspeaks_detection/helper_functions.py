#PKGS
##########################################################################################
##########################################################################################
##########################################################################################
# Import Libraries
# Standard libraries

import warnings

# Data manipulation and analysis
import numpy as np


# Ignore warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='divide by zero encountered in divide')


################################## HELPER  ############################################
#########################################################################################
def rebin_snr(t, x, sy, snr_threshold):
    '''
    Rebin the signal based on the signal-to-noise ratio (SNR) threshold.

    Parameters:
    - t: Array of time values.
    - x: Array of signal values.
    - sy: Array of signal uncertainties (standard deviations).
    - snr_threshold: The SNR threshold for rebinned data.

    Returns:
    - t_new: Rebinned time values.
    - c_new: Rebinned signal values.
    - sc_new: Rebinned uncertainties.
    '''
    # Initialize lists for results
    w = []  # Weights for SNR calculation
    c_bin = []  # Binned counts
    t_bin = []  # Binned time
    sc_bin = []  # Binned uncertainties

    c_new = []  # Rebinned counts
    t_new = []  # Rebinned time
    sc_new = []  # Rebinned uncertainties

    # Mask to filter out non-positive uncertainties
    mask = np.where(sy > 0)[0]
    t, x, sy = t[mask], x[mask], sy[mask]

    # Iterate through the signal
    for i in range(len(x)):

        # Calculate weight and add to the bins
        weight = 1 / (sy[i] ** 2)
        w.append(weight)
        t_bin.append(t[i])
        c_bin.append(x[i])
        sc_bin.append(sy[i])

        # Calculate the current SNR
        sc_weight = np.sqrt(1 / np.sum(w))  # Combined uncertainty
        c_weight = np.sum(np.array(c_bin) * np.array(w)) / np.sum(w)  # Weighted count
        snr_now = sc_weight / (c_weight + 1e-10)  # SNR

        # Check if SNR is below threshold
        if snr_now <= snr_threshold:
            # Convert lists to arrays for final calculations
            w = np.array(w)
            c_bin = np.array(c_bin)
            sc_bin = np.array(sc_bin)
            t_bin = np.array(t_bin)

            # Compute the rebinned values
            sc_new.append(np.sqrt(1 / np.sum(w)))
            c_new.append(np.sum(c_bin * w) / np.sum(w))
            t_new.append(np.sum(t_bin * w) / np.sum(w))

            # Reset bins for next group
            w, c_bin, t_bin, sc_bin = [], [], [], []

    # Handle the remaining bin if any points left
    if w:
        w = np.array(w)
        c_bin = np.array(c_bin)
        sc_bin = np.array(sc_bin)
        t_bin = np.array(t_bin)

        sc_new.append(np.sqrt(1 / np.sum(w)))
        c_new.append(np.sum(c_bin * w) / np.sum(w))
        t_new.append(np.sum(t_bin * w) / np.sum(w))

    # Convert results to numpy arrays
    return np.array(t_new), np.array(c_new), np.array(sc_new)

def _moving_average(data, window_size):
    window_size = int(window_size)
    pad_width = window_size // 2

    # Pad the data array with edge values to handle edge cases
    padded_data = np.pad(data, pad_width, mode='edge')

    # Compute the moving average using convolution
    weights = np.ones(window_size) / window_size
    moving_avg = np.convolve(padded_data, weights, mode='valid')

    # Adjust the length of the moving average if necessary
    if len(moving_avg) > len(data):
        moving_avg = moving_avg[:len(data)]

    return moving_avg

#########################################################################################

def _base_calculator(y):
    n = len(y)
    max_window = max(int(n / 25), 3)
    #print(max_window)
    
    # Collect all valid moving averages in a list using the defined helper _moving_average
    moving_averages = [
        _moving_average(y, i) for i in np.linspace(1, max_window, 25)
        if len(_moving_average(y, i)) == n
    ]

    if not moving_averages:
        return np.array([])

    # Convert list of moving averages to a 2D NumPy array and compute the minimum
    s = np.array(moving_averages)
    base = np.min(s, axis=0)
    
    # Return the element-wise minimum between base and original y
    return np.minimum(base, y)

def scale(x, y):
    """
    Scale the `x` data to match the range of the `y` data. The purpose is facilitate creation of plots.

    This function scales the values in the `x` array to match the range of the `y` array.
    It linearly transforms the `x` values such that they span the same range as `y`.

    Parameters
    ----------
    x : array-like
        The input data to be scaled.
    y : array-like
        The data whose range is used for scaling `x`.

    Returns
    -------
    x_new : array-like
        The scaled version of `x`, with values transformed to the range of `y`.
    """

    x_new = ((max(y) - min(y)) / (max(x) - min(x))) * (x - max(x)) + max(y)
    return x_new

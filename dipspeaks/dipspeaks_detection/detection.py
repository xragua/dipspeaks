#PKGS
##########################################################################################
##########################################################################################
##########################################################################################
# Import Libraries
# Standard libraries

import warnings

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Plotting and visualization


# SciPy for scientific computing


from scipy.signal import (
    find_peaks, peak_widths
)


# Scikit-learn for machine learning

# Ignore warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='divide by zero encountered in divide')

##########################################################################################
##########################################################################################
##########################################################################################

def _detection(t, x, sy):
    '''
    Find peaks in the signal and calculate their properties including SNR (uncertainty-based).

    Parameters:
    - t: Array of time values.
    - x: Array of signal values.
    - sy: Array of signal uncertainties (standard deviations).
    - maxlen: Maximum allowed length for the peak's duration.
    - minlen: Minimum allowed length for the peak's duration.

    Returns:
    - DataFrame containing detected peak/dip properties including SNR.
    '''
    # Initialize lists to collect peak properties
    pddip, pdprominence, pdwidth, pdwidth05 = [], [], [], []
    pdwidth_height, pdleft_ips, pdright_ips = [], [], []
    pdleft_ips05, pdright_ips05 = [], []
    dispersions = []

    minlen= np.mean(np.diff(t))

    # Invert the signal to find dips as peaks
    x_new = x - min(x)

    # Set thresholds for prominence and width (adjust as needed)
    prominence_threshold = 1e-7
    width_threshold = 1e-7

    # Find peaks
    peaks, properties = find_peaks(x_new, width=width_threshold, 
                                   prominence=prominence_threshold, rel_height=0.9)

    # Compute the widths of the peaks at 50% of their prominence
    width_05, width_height_05, left_ips_05, right_ips_05 = peak_widths(x_new, peaks, rel_height=0.5)

    # Collect properties in lists
    pddip.append(peaks)
    pdprominence.append(properties['prominences'])
    pdwidth.append(properties['widths'])
    pdleft_ips.append(properties['left_ips'])
    pdright_ips.append(properties['right_ips'])
    pdwidth_height.append(properties['width_heights'])
    pdwidth05.append(width_05)
    pdleft_ips05.append(left_ips_05)
    pdright_ips05.append(right_ips_05)

    # Calculate dispersions for each peak
    for i in range(len(peaks)):
        left = int(properties['left_ips'][i] - 1)
        right = int(properties['right_ips'][i] + 1)
        dispersion_value = np.median(np.abs(np.diff(x_new[left:right])))
        dispersions.append(dispersion_value)

    # Concatenate collected properties into a DataFrame
    data = {
        'pos': np.concatenate(pddip),
        'prominence': np.concatenate(pdprominence),
        'width': np.concatenate(pdwidth),
        'width_height': np.concatenate(pdwidth_height),
        'left_ips': np.concatenate(pdleft_ips),
        'right_ips': np.concatenate(pdright_ips),
        'left_ips05': np.concatenate(pdleft_ips05),
        'right_ips05': np.concatenate(pdright_ips05),
    }
    dips = pd.DataFrame(data)

    # Convert peak positions to time values
    dips['t'] = t[dips.pos.astype(int)]

    # Refine peak start and end times via interpolation
    pose = dips.right_ips.astype(int)
    posi = dips.left_ips.astype(int)
    pose05 = dips.right_ips05.astype(int)
    posi05 = dips.left_ips05.astype(int)
    diff_num = np.insert(np.diff(t), 0, 0)
    ti = t[posi] + diff_num[posi] * (dips.left_ips - posi)
    te = t[pose] + diff_num[pose] * (dips.right_ips - pose)
    ti05 = t[posi05] + diff_num[posi05] * (dips.left_ips05 - posi05)
    te05 = t[pose05] + diff_num[pose05] * (dips.right_ips05 - pose05)
    dips['ti'] = ti
    dips['te'] = te
    dips['ti05'] = ti05
    dips['te05'] = te05

    # Calculate additional properties
    dips['duration'] = dips.te - dips.ti
    dips['taua'] = dips.t - dips.ti
    dips['taub'] = dips.te - dips.t
    dips['psi'] = (dips.taub - dips.taua) / (dips.taub + dips.taua)
    dips['duration05'] = dips.te05 - dips.ti05
    dips['rduration'] = np.abs(dips.duration05 / dips.duration)
    dips['relprominence'] = dips.prominence / np.abs(dips.width_height)
    dips['density'] = dips.prominence / dips.duration
    dips['dispersion'] = np.array(dispersions) / dips.relprominence

    # Calculate SNR using uncertainty-based noise estimation
    peak_snr = []
    for i, peak_idx in enumerate(dips['pos']):
        # Use prominence as peak amplitude
        peak_amplitude = dips['prominence'].iloc[i]
        left_boundary = int(dips['left_ips'].iloc[i])
        right_boundary = int(dips['right_ips'].iloc[i])
        # Define window for noise estimation (extend by at least 5 points)
        left_idx = max(0, left_boundary - max(int(right_boundary - left_boundary), 5))
        right_idx = min(len(x), right_boundary + max(int(right_boundary - left_boundary), 5))

        # Estimate noise level using uncertainties if available
        if len(sy) > 0:
            noise_level = np.mean(sy[left_idx:right_idx])
        else:
            noise_level = np.std(x_new)
        snr = peak_amplitude / (noise_level + 1e-10)
        peak_snr.append(snr)

    dips['snr'] = peak_snr

    # Filter peaks based on duration thresholds
    dips = dips[(dips.duration >= minlen)].reset_index(drop=True)
    dips = dips[(dips.prominence >0)].reset_index(drop=True)
    dips = dips.sort_values(by='t', ascending=False).reset_index(drop=True)

    return dips.reset_index(drop=True)

#PKGS
##########################################################################################
##########################################################################################
##########################################################################################
# Import Libraries
# Standard libraries
import os
import glob
import itertools
import pickle
import random
import warnings

# Data manipulation and analysis
import numpy as np
import pandas as pd
import random
# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import cm

# SciPy for scientific computing
from scipy import signal, stats as s
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.fftpack import fft
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import curve_fit
from scipy.signal import (
    find_peaks, peak_widths, peak_prominences, savgol_filter, find_peaks_cwt
)
from scipy.spatial.distance import pdist, cdist
from scipy.special import erf
from scipy.stats import kde, mode, skewnorm, norm
from scipy import signal

# Scikit-learn for machine learning
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, Birch, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

# TensorFlow and Keras for deep learning
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from statsmodels.tsa.arima_process import ArmaProcess

# Ignore warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='divide by zero encountered in divide')

##########################################################################################
##########################################################################################
##########################################################################################

def _detection(t, x, sy, maxlen, minlen):
    '''
    Find peaks in the signal and calculate their properties including SNR (uncertainty-based).

    Parameters:
    - t: Array of time values.
    - x: Array of signal values.
    - sy: Array of signal uncertainties (standard deviations).
    - maxlen: Maximum allowed length for the peak's duration.
    - minlen: Minimum allowed length for the peak's duration.

    Returns:
    - dips: DataFrame containing detected peak properties including SNR.
    '''
    # Initialize lists to collect peak properties
    pddip, pdprominence, pdwidth, pdwidth05 = [], [], [], []
    pdwidth_height, pdleft_ips, pdright_ips = [], [], []
    pdleft_ips05, pdright_ips05 = [], []
    dispersions = []

    # Invert the signal to find dips as peaks
    x_new = x - min(x)
    
    print(minlen,maxlen)

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
        # Exclude the peak region itself
        noise_region = np.concatenate((x_new[left_idx:left_boundary],
                                       x_new[right_boundary + 1:right_idx]))
        # Estimate noise level using uncertainties if available
        if len(sy) > 0:
            noise_level = np.mean(sy[left_idx:right_idx])
        else:
            noise_level = np.std(x_new)
        snr = peak_amplitude / (noise_level + 1e-10)
        peak_snr.append(snr)

    dips['snr'] = peak_snr

    # Filter peaks based on duration thresholds
    dips = dips[(dips.duration <= maxlen) & (dips.duration >= minlen)]
    dips = dips.sort_values(by='t', ascending=False).reset_index(drop=True)

    return dips.reset_index(drop=True)

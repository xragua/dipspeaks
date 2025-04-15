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

#########################################################################################
pipeline = Pipeline([
    ('normalizer', Normalizer()),
    ('scaler', MinMaxScaler())
])

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


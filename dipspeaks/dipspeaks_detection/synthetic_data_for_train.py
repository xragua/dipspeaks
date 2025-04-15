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
def _calculate_synthetic_data(t, c, sc, num_simulations):
    '''
    Create synthetic data reflecting count errors in the main lightcurve.

    Parameters:
    - t: array-like, time data
    - c: array-like, counts data
    - sc: array-like, count errors
    - num_simulations: int, number of simulations to generate

    Returns:
    - tsim, simc, ssimc: synthetic time series, counts, and errors
    '''
    # Detrend the original counts using the Savitzky-Golay filter
    detrend = signal.savgol_filter(c, window_length=100, polyorder=8, mode='nearest')
    det_c = (c - detrend)

    # Identify and replace outliers based on Z-score thresholding
    mean_value = np.mean(det_c)
    std_dev = np.std(det_c)
    z_threshold = 3
    z_scores = (det_c - mean_value) / std_dev
    outliers = np.abs(z_scores) > z_threshold
    det_c = np.where(outliers, mean_value, det_c)

    n = len(det_c)

    # Pre-allocate arrays for synthetic data
    simc = np.empty(n * num_simulations)
    ssimc = np.empty(n * num_simulations)
    tsimc = np.empty(n * num_simulations)

    mean_diff_t = np.mean(np.diff(t))
    std_dev_counts = np.std(c)
    std_dev_scounts = np.std(sc)

    std_counts_proportion = (sc/c)
    for k in range(num_simulations):
        
        perm_index = np.random.permutation(n)
        start_index = k * n
        end_index = start_index + n

        mixed_diffs = np.concatenate([[mean_diff_t], np.diff(t)])
        shuffled_diffs = mixed_diffs[perm_index]

        tsimc[start_index] = t[0] + k * (max(t) - min(t))
        tsimc[start_index:end_index] = tsimc[start_index] + np.cumsum(shuffled_diffs)

        # Simulate counts data using Gaussian noise
        #gaussian_random_numbers = np.random.normal(loc=0, scale=std_dev_counts, size=n)
        simcounts_ = random.choices(det_c, k=len(det_c))
        simcounts = simcounts_-np.mean(simcounts_)+np.mean(c)
        simc[start_index:end_index] = simcounts
        
        # Simulate errors to resemble original errors
        #sgaussian_random_numbers = np.random.normal(loc=0, scale=std_dev_scounts, size=n)
        ssimc[start_index:end_index] = random.choices(std_counts_proportion, k=len(sc))*simcounts
        
    # Create a mask where both simc and ssimc are greater than 0
    mask = (simc <= 0) | (ssimc <= 0)
    
    # Assign the mean of the entire array to each element where the mask is True
    simc[mask] = np.mean(simc)
    ssimc[mask] = np.mean(ssimc)
    
    stb = tsimc

    return stb, simc, ssimc

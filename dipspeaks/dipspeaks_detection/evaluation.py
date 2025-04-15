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
#########################################################################################

def _modified_z_score(errors_train, errors):
    '''
    Calculate the modified Z-score for a set of errors.
    It is more robust to outliers because it uses the Median Absolute Deviation (MAD).
    '''
    median_error = np.median(errors_train)
    mad_error = np.median(np.abs(errors_train - median_error))
    z_scores = 0.6745 * (errors - median_error) / (mad_error + 1e-10)  # Adding a small value to prevent division by zero
    return z_scores

    # Improved Outlier Probability Calculation
def _outlier_probability(errors_train, errors, threshold=3.5):
    '''
    Calculate the outlier probability based on a modified Z-score.
    Points with higher modified Z-scores are more likely to be outliers.
    '''
    z_scores = _modified_z_score(errors_train, errors)
    # Use survival function for both tails (greater than abs(z))
    outlier_probabilities = 2 * norm.sf(np.abs(z_scores))  # 2 times the survival function for two-tailed probability
    outlier_flags = np.abs(z_scores) > threshold  # Flagging potential outliers
    return outlier_probabilities, outlier_flags
    
def _real_probability(real_rate, sim_rate):
    """
    Estimate the probability that a detected event is real,
    based on the real and simulation detection rates.
    """
    if real_rate == 0:
        return 0.0  # No events detected, so probability of being real is zero
    return max(0.0, min(1.0, (real_rate - sim_rate) / real_rate))
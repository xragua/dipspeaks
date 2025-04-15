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


from .helper_functions import (
     rebin_snr,
     _moving_average,
    _base_calculator,
)

from .detection import (
    _detection,
)

from .synthetic_data_for_train import (
    _calculate_synthetic_data,
)

from .autoencoder_model import (
    _clean_autoencoder,
)

from .evaluation import(
    _modified_z_score,
    _outlier_probability,
    _real_probability,
)


#########################################################################################
#########################################################################################
#########################################################################################

def detect_dips_and_peaks(lc, snr=0.15 ,index_time=0, index_rate=1, index_error_rate=2, maxlen=10000, minlen=1, filename='lc',num_simulations=50, show_plot = True):
    num_simulations=num_simulations
    print('collecting candidates')
    
    #lOAD LC............................................................................
    series = pd.read_csv(lc, sep=r'\s+', header=None, skiprows=21, comment='!')#lOAD LC

    t = np.array(series[index_time])
    c = np.array(series[index_rate])
    sc = np.array(series[index_error_rate])
    
    # Filter out invalid data
    mask = np.where((sc > 0) & (c > 0))[0]
    t, c, sc = t[mask], c[mask], sc[mask]
    
    # Create synthetic data............................................................................
    #print('- create synthetic data')
    #tsim_, simc_, sssimc_ =  _calculate_synthetic_data(t, c, sc,num_simulations)
    print('- done!')
    results = {'dips': [], 'peaks': [], 'sim_dips': [], 'sim_peaks': []}
    
    snr_thresholds = [snr]
    
    #Rebin all data .........................................................................
    for r in snr_thresholds:
    
        # Rebin data and noise
        tb, cb, scb = rebin_snr(t, c, sc, r)
        lcreb = pd.DataFrame({
            't': tb,
            'c': cb,
            'sc': scb
        })

        #tsim, simc, ssimc = rebin_snr(tsim_, simc_, sssimc_, r)
        tsim, simc, ssimc =  _calculate_synthetic_data(lcreb.t, lcreb.c, lcreb.sc,num_simulations)
        tsim = np.array(tsim)
        simc = np.array(simc)
        ssimc = np.array(ssimc)
        
        cb = np.array(cb)
        tb = np.array(tb)
        scb = np.array(scb)

    
    #Collect candidates ......................................................................
    if len(tb) >= 50:
        print('- base calculator')
        lc_dips = _base_calculator(cb)
        lc_sdips = _base_calculator(simc)
        
                
        lc_peaks = _base_calculator(-cb)
        lc_speaks = _base_calculator(-simc)
        print('- done!')
        print('- detecting dips and peaks within light curve')
        dips = _detection(tb, -lc_dips, scb, maxlen, minlen)
        peaks = _detection(tb, -lc_peaks, scb, maxlen, minlen)
        print('- done!')
        
        print('- detecting dips and peaks within synthetic data')
        sdips = _detection(tsim, -lc_sdips, ssimc, maxlen, minlen)
        speaks = _detection(tsim, -lc_speaks, ssimc, maxlen, minlen)
        print('- done!')
        
        dips = dips.dropna().reset_index(drop=True)
        #print('dips')
        peaks = peaks.dropna().reset_index(drop=True)
        #print('peaks')
        sdips = sdips.dropna().reset_index(drop=True)
        #print('simulated dips')
        speaks = speaks.dropna().reset_index(drop=True)
        #print('simulated peaks')
        
        # Store results in the dictionary
        results['dips'].append(dips)
        results['peaks'].append(peaks)
        results['sim_dips'].append(sdips)
        results['sim_peaks'].append(speaks)
        
        #print('candidates collected')
            
        if show_plot:
            
            plt.figure(figsize=(10, 3))
            plt.plot(tb, lc_dips, 'r')
            plt.plot(tb[dips.pos], lc_dips[dips.pos], '*k')
                
            plt.plot(tb, -lc_peaks, 'b')
            plt.plot(tb[peaks.pos], -lc_peaks[peaks.pos], '.k')
            plt.title('Raw result')
            plt.show()
    
    if len(tb) < 50:
        print('The light curve is too short')
        
#########################################################################################
###########################   STEP 2 : TRAINING AUTOENCODERS   ##########################
#########################################################################################
        
    total_dmse_train=[]
    total_dmse_test=[]
    total_pmse_train=[]
    total_pmse_test=[]

    # Extract the dips, peaks, and simulated dips and peaks for this file
    dips_list = results.get('dips', [])
    peaks_list = results.get('peaks', [])
    sim_dips_list = results.get('sim_dips', [])
    sim_peaks_list = results.get('sim_peaks', [])

    ############ COLLECT RESULTS
    # Initialize empty DataFrames for cleaned dips and peaks
    dips_to_clean = pd.DataFrame()
    peaks_to_clean = pd.DataFrame()
    sdips_to_clean = pd.DataFrame()
    speaks_to_clean = pd.DataFrame()

    # Assign the lists of data directly
    dips = pd.concat(dips_list) if isinstance(dips_list, list) else dips_list
    sdips = pd.concat(sim_dips_list) if isinstance(sim_dips_list, list) else sim_dips_list
    peaks = pd.concat(peaks_list) if isinstance(peaks_list, list) else peaks_list
    speaks = pd.concat(sim_peaks_list) if isinstance(sim_peaks_list, list) else sim_peaks_list
    
    #print('evaluating candidates')
    # Process dips if both dips and sdips contain data
    if not dips.empty and not sdips.empty:
        dips_, dmse_test, dmse_train = _clean_autoencoder(dips, sdips, show_plot=show_plot)
        sdips_, sdmse_test, sdmse_train = _clean_autoencoder(sdips, sdips, show_plot=show_plot)
        dips_to_clean = pd.concat([dips_to_clean, dips_])
        sdips_to_clean = pd.concat([sdips_to_clean, sdips_])

    # Process peaks if both peaks and speaks contain data
    if not peaks.empty and not speaks.empty:
        peaks_, pmse_test, pmse_train = _clean_autoencoder(peaks, speaks, show_plot=show_plot)
        speaks_, sdmse_test, sdmse_train = _clean_autoencoder(speaks, speaks, show_plot=show_plot)
        peaks_to_clean = pd.concat([peaks_to_clean, peaks_])
        speaks_to_clean = pd.concat([speaks_to_clean, speaks_])


    ############# SAVE RESULTS
    # Identify potential peaks and dips with a high outlier probability (over 90%)
    pospeaks = peaks_to_clean[(peaks_to_clean.is_outlier==True )] if not peaks_to_clean.empty else None
    posdips = dips_to_clean[(dips_to_clean.is_outlier==True )] if not dips_to_clean.empty else None
    
    spospeaks = speaks_to_clean[(speaks_to_clean.is_outlier==True )] if not speaks_to_clean.empty else None
    sposdips = sdips_to_clean[(sdips_to_clean.is_outlier==True )] if not sdips_to_clean.empty else None
    
    real_rate_peaks = (len(pospeaks) / len(peaks_to_clean)) if len(peaks_to_clean) > 0 else 0
    sim_rate_peaks  = (len(spospeaks) / len(speaks_to_clean))  if len(speaks_to_clean) > 0 else 0

    real_rate_dips  = (len(posdips) / len(dips_to_clean))    if len(dips_to_clean) > 0 else 0
    sim_rate_dips   = (len(sposdips) / len(sdips_to_clean))   if len(sdips_to_clean) > 0 else 0

    
    
    ppeaks = _real_probability(real_rate_peaks, sim_rate_peaks)
    pdips = _real_probability(real_rate_dips, sim_rate_dips)
    

    print("Simulation:")
    print("Peaks per second:", np.round(len(spospeaks)/((max(t)-min(t))*num_simulations),4),
          "percentage of rejected peaks:", np.round((len(speaks_to_clean)-len(spospeaks))/len(speaks_to_clean),4))
          
    print("Dips per second:", np.round(len(sposdips)/((max(t)-min(t))*num_simulations)),
          "percentage of rejected dips:", np.round((len(sdips_to_clean)-len(sposdips))/len(sdips_to_clean),4))
          
    print("Result:")
    print("Peaks per second:", np.round(len(pospeaks)/(max(t)-min(t)),4),
          "percentage of rejected peaks:", np.round((len(peaks_to_clean)-len(pospeaks))/len(peaks_to_clean),4),
          "probability of detected peaks:",ppeaks)
          
    print("Dips per second:", np.round(len(posdips)/(max(t)-min(t)),4),
          "percentage of rejected dips:", np.round((len(dips_to_clean)-len(posdips))/len(dips_to_clean),4),
          "probability of detected dips:",pdips)
          


    if show_plot:
        # Plot dips and peaks, marking positions with 90%+ probability
        plt.figure(figsize=(10, 3))

        # Plot dips with high probability if available
        plt.plot(tb, cb, 'k', label='Dips')
        if posdips is not None and not posdips.empty:
            plt.plot(tb[posdips.pos], cb[posdips.pos], '*r', label='High-prob Dips')
        if pospeaks is not None and not pospeaks.empty:
            plt.plot(tb[pospeaks.pos], cb[pospeaks.pos], '.r', label='High-prob Peaks')

        # Add legend and title
        plt.title('Outliers')
        plt.legend()
        plt.show()

    return peaks_to_clean, dips_to_clean, lcreb
    


#########################################################################################
###########################   COINCIDENCES BETWEEN   ##########################
#########################################################################################
def calculate_overlap_gtp(start1, end1, start2, end2):
    
    start1 = np.array(start1)
    end1 = np.array(end1)
    start2 = np.array(start2)
    end2 = np.array(end2)

    max_start = np.maximum(start1[:, np.newaxis], start2)
    min_end = np.minimum(end1[:, np.newaxis], end2)

    overlap = np.maximum(0, min_end - max_start)
    hoverlap = overlap / (end1[:, np.newaxis] - start1[:, np.newaxis])
    loverlap = overlap / (end2 - start2)

    high_indices, low_indices = np.where(overlap > 0)

    return overlap[high_indices, low_indices], high_indices, low_indices, hoverlap[high_indices, low_indices], loverlap[high_indices, low_indices]



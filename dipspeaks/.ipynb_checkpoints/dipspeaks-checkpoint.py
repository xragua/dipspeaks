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
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

#########################################################################################
pipeline = Pipeline([
    ('normalizer', Normalizer()),
    ('scaler', MinMaxScaler())
])

################################## HELPER  ############################################
#########################################################################################
def rebin_snr(t, x, sy, snr_threshold):
    """
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
    """
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


#########################################################################################
#########################################################################################

def _calculate_synthetic_data(t, c, sc, num_simulations=10):
    """
    Create synthetic data reflecting count errors in the main lightcurve.

    Parameters:
    - t: array-like, time data
    - c: array-like, counts data
    - sc: array-like, count errors
    - num_simulations: int, number of simulations to generate

    Returns:
    - tsim, simc, ssimc: synthetic time series, counts, and errors
    """
    # Detrend the original counts using the Savitzky-Golay filter
    detrend = signal.savgol_filter(c, window_length=100, polyorder=8, mode="nearest")
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



#########################################################################################
#########################################################################################

def _detection(t, x, sy, maxlen, minlen):
    """
    Find peaks in the signal and calculate their properties including SNR.

    Parameters:
    - t: Array of time values.
    - x: Array of signal values.
    - sy: Array of signal uncertainties (standard deviations).
    - maxlen: Maximum allowed length for the peak's duration.
    - minlen: Minimum allowed length for the peak's duration.

    Returns:
    - dips: DataFrame containing detected peak properties including SNR.
    """

    # Initialize lists to collect peak properties
    pddip, pdprominence, pdwidth, pdwidth05 = [], [], [], []
    pdwidth_height, pdleft_ips, pdright_ips = [], [], []
    pdleft_ips05, pdright_ips05 = [], []
    dispersions = []

    # Invert the signal to find dips as peaks
    x_new = x-min(x)

    # Calculate prominence and width thresholds
    prominence_threshold =  0.0000001
    width_threshold = 0.0000001  # Adjust based on data characteristics

    # Find peaks
    peaks, properties = find_peaks(x_new, width=width_threshold, prominence=prominence_threshold,rel_height=0.9)

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

    # Calculate dispersions
    for i in range(len(peaks)):
        left = int(properties['left_ips'][i] - 1)
        right = int(properties['right_ips'][i] + 1)
        dispersion_value = np.median(abs(np.diff(x_new[left:right])))
        dispersions.append(dispersion_value)

    # Concatenate all collected properties
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

    # Convert positions to time values
    dips["t"] = t[dips.pos.astype(int)]

    pose = dips.right_ips.astype(int)
    posi = dips.left_ips.astype(int)
    pose05 = dips.right_ips05.astype(int)
    posi05 = dips.left_ips05.astype(int)
    diff_num = np.insert(np.diff(t), 0, 0, axis=0)
    ti = t[posi] + diff_num[posi] * (dips.left_ips - posi)
    te = t[pose] + diff_num[pose] * (dips.right_ips - pose)
    ti05 = t[posi05] + diff_num[posi05] * (dips.left_ips05 - posi05)
    te05 = t[pose05] + diff_num[pose05] * (dips.right_ips05 - pose05)
    dips["ti"] = ti
    dips["te"] = te
    dips["ti05"] = ti05
    dips["te05"] = te05

    # Calculate additional properties
    dips["duration"] = dips.te - dips.ti
    dips["taua"] = dips.t - dips.ti
    dips["taub"] = dips.te - dips.t
    dips["psi"] = (dips.taub - dips.taua) / (dips.taub + dips.taua)
    dips["duration05"] = dips.te05 - dips.ti05
    dips['rduration'] = abs(dips.duration05 / dips.duration)
    dips['relprominence'] = dips.prominence / abs(dips.width_height)
    dips['density'] = dips.prominence / dips.duration
    dips['dispersion'] = np.array(dispersions) / dips.relprominence

    # Calculate SNR for each peak
    peak_snr_std = []
    peak_snr_uncertainty = []
    
    for i, peak_idx in enumerate(dips['pos']):
        # Peak amplitude (prominence is used as amplitude)
        peak_amplitude = dips['prominence'].iloc[i]

        # Dynamic window size: Use the peak's width
        left_boundary = int(dips['left_ips'].iloc[i])
        right_boundary = int(dips['right_ips'].iloc[i])

        # Define left and right indices for noise estimation, extending beyond the peak's boundaries
        left_idx = max(0, left_boundary - max(int(right_boundary - left_boundary),5))
        right_idx = min(len(x), right_boundary + max(int(right_boundary - left_boundary),5))

        # Exclude the peak itself from noise calculation
        noise_region = np.concatenate((x_new[left_idx:left_boundary], x_new[right_boundary + 1:right_idx]))

        # Noise level estimation for snr_std (standard deviation-based)
        noise_level1 = np.std(noise_region) if len(noise_region) > 0 else np.std(x_new)

        # Noise level estimation for snr_uncertainty (uncertainty-based)
        if len(sy) > 0:
            noise_level2 = np.mean(sy[left_idx:right_idx])  # Use provided uncertainties
        else:
            noise_level2 = noise_level1  # Fallback to standard deviation if uncertainties are not provided

        # Calculate SNR values
        snr_std = peak_amplitude / (noise_level1 + 1e-10)  # Avoid division by zero
        snr_uncertainty = peak_amplitude / (noise_level2 + 1e-10)  # Avoid division by zero

        peak_snr_std.append(snr_std)
        peak_snr_uncertainty.append(snr_uncertainty)

    # Add SNR to the DataFrame
    dips['snr_std'] = peak_snr_std
    dips['snr_uncertainty'] = peak_snr_uncertainty

    # Filter based on duration and SNR threshold
    dips = dips[(dips.duration < maxlen) & (dips.duration > minlen)]
    dips = dips.sort_values(by='t', ascending=False).reset_index(drop=True)

    #print(len(dips))
    return dips.reset_index(drop=True)
    
#########################################################################################
#########################################################################################

def _modified_z_score(errors_train, errors):
    """
    Calculate the modified Z-score for a set of errors.
    It is more robust to outliers because it uses the Median Absolute Deviation (MAD).
    """
    median_error = np.median(errors_train)
    mad_error = np.median(np.abs(errors_train - median_error))
    z_scores = 0.6745 * (errors - median_error) / (mad_error + 1e-10)  # Adding a small value to prevent division by zero
    return z_scores

    # Improved Outlier Probability Calculation
def _calculate_outlier_probability(errors_train, errors, threshold=3.5):
    """
    Calculate the outlier probability based on a modified Z-score.
    Points with higher modified Z-scores are more likely to be outliers.
    """
    z_scores = _modified_z_score(errors_train, errors)
    # Use survival function for both tails (greater than abs(z))
    outlier_probabilities = 2 * norm.sf(np.abs(z_scores))  # 2 times the survival function for two-tailed probability
    outlier_flags = np.abs(z_scores) > threshold  # Flagging potential outliers
    return outlier_probabilities, outlier_flags
    
#########################################################################################
#########################################################################################

def _clean_autoencoder(pd_to_clean, pd_base, show_plot=True):
    """
    Clean a dataset by detecting and flagging outliers using an autoencoder model.

    This function builds an autoencoder to reconstruct data from a baseline dataset (`pd_base`).
    It then uses the reconstruction errors to identify outliers in a separate dataset (`pd_to_clean`)
    based on the specified features. Outlier probabilities and flags are added to the original
    dataframe (`pd_to_clean`).

    Parameters:
    - pd_to_clean (pd.DataFrame): The dataset to be cleaned, containing columns that match those in
                                    `pd_base` for feature selection and comparison.
    - pd_base (pd.DataFrame): The baseline dataset used to train the autoencoder, typically representing
                                normal or expected data patterns.

    Returns:
    - pd_to_clean (pd.DataFrame): The input dataframe with additional columns:
        - "outlier_prob": The calculated outlier probability based on reconstruction error.
        - "is_outlier": A flag indicating whether the row is an outlier (1) or not (0).
    - mse_test (np.array): Mean squared reconstruction errors for the `pd_to_clean` dataset.
    - mse_train (np.array): Mean squared reconstruction errors for the `pd_base` dataset.
    """
    # Select relevant columns
    selected_columns = ['density', 'prominence', 'duration', 'rduration', 'snr_std']
    pd_to_clean_selection = pd_to_clean[selected_columns]
    pd_base_selection = pd_base[selected_columns]

    # Split data into training (baseline) and test (to_clean) sets
    X_train = pd_base_selection
    X_test = pd_to_clean_selection

    # Fit the pipeline on X_train and transform both X_train and X_test
    pipeline.fit(X_train)
    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # Define the autoencoder architecture
    input_dim = X_train_transformed.shape[1]
    encoding_dim = 256

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='elu')(input_layer)
    encoded = Dense(encoding_dim // 2, activation='elu')(encoded)
    encoded = Dense(encoding_dim // 4, activation='elu')(encoded)
    encoded = Dense(encoding_dim // 8, activation='elu')(encoded)
    encoded = Dense(encoding_dim // 16, activation='elu')(encoded)
    encoded = Dense(encoding_dim // 32, activation='elu')(encoded)
    encoded = Dense(encoding_dim // 64, activation='elu')(encoded)
    encoded = Dense(encoding_dim // 128, activation='elu')(encoded)
    encoded = Dense(encoding_dim // 256, activation='elu')(encoded)
        
    decoded = Dense(encoding_dim // 256, activation='elu')(encoded)
    decoded = Dense(encoding_dim // 128, activation='elu')(encoded)
    decoded = Dense(encoding_dim // 64, activation='elu')(encoded)
    decoded = Dense(encoding_dim // 32, activation='elu')(encoded)
    decoded = Dense(encoding_dim // 16, activation='elu')(encoded)
    decoded = Dense(encoding_dim // 8, activation='elu')(decoded)
    decoded = Dense(encoding_dim // 4, activation='elu')(decoded)
    decoded = Dense(encoding_dim // 2, activation='elu')(decoded)
    decoded = Dense(input_dim, activation='elu')(decoded)

    autoencoder = Model(input_layer, decoded)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mae')

    # Define callbacks for early stopping and learning rate adjustment
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)

    # Train the autoencoder on the baseline data
    history = autoencoder.fit(X_train_transformed, X_train_transformed,
                                epochs=20000,
                                batch_size=128,
                                shuffle=True,
                                validation_split=0.1,
                                callbacks=[early_stopping, reduce_lr],
                                verbose=0)
    if show_plot:
        # Plot the learning curve
        plt.figure(figsize=(4, 3))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    # Reconstruct data and compute reconstruction errors
    reconstructed_train = autoencoder.predict(X_train_transformed)
    reconstructed_test = autoencoder.predict(X_test_transformed)

    # Calculate Mean Squared Error (MSE) for reconstruction errors
    mse_train = np.mean(np.power(X_train_transformed - reconstructed_train, 2), axis=1)
    mse_test = np.mean(np.power(X_test_transformed - reconstructed_test, 2), axis=1)

    # Identify outliers using reconstruction error
    outlier_probabilities, outlier_flags = _calculate_outlier_probability(mse_train, mse_test)

    # Append outlier probability and flags to the cleaned dataset
    pd_to_clean["outlier_prob"] = outlier_probabilities
    pd_to_clean["is_outlier"] = outlier_flags
    pd_to_clean = pd_to_clean.sort_values(by='t').reset_index(drop=True)

    return pd_to_clean, mse_test, mse_train

#########################################################################################
#################################   STEP 1 : DETECTION   ################################
#########################################################################################

def detect_dips_and_peaks(lc, snr=0.1 ,index_time=0, index_rate=1, index_error_rate=2, maxlen=10000, minlen=1, filename="lc", show_plot = True):

    print("collecting candidates")
    
    #lOAD LC............................................................................
    series = pd.read_csv(lc, sep=r"\s+", header=None, skiprows=21, comment="!")#lOAD LC

    t = np.array(series[index_time])
    c = np.array(series[index_rate])
    sc = np.array(series[index_error_rate])
    
    # Filter out invalid data
    mask = np.where((sc > 0) & (c > 0))[0]
    t, c, sc = t[mask], c[mask], sc[mask]
    
    # Create synthetic data............................................................................
    #print("- create synthetic data")
    tsim_, simc_, sssimc_ =  _calculate_synthetic_data(t, c, sc)
    #print("- done!")
    results = {"dips": [], "peaks": [], "sim_dips": [], "sim_peaks": []}
    
    snr_thresholds = [0.15]
    
    #Rebin all data .........................................................................
    for r in snr_thresholds:
    
        # Rebin data and noise
        tb, cb, scb = rebin_snr(t, c, sc, r)
        lcreb = pd.DataFrame({
            't': tb,
            'c': cb,
            'sc': scb
        })

        tsim, simc, ssimc = rebin_snr(tsim_, simc_, sssimc_, r)
        
        tsim = np.array(tsim)
        simc = np.array(simc)
        ssimc = np.array(ssimc)
        
        cb = np.array(cb)
        tb = np.array(tb)
        scb = np.array(scb)

    
    #Collect candidates ......................................................................
    if len(tb) >= 50:
        #print("- base calculator")
        lc_dips = _base_calculator(cb)
        lc_sdips = _base_calculator(simc)
        #print("- done!")
                
        lc_peaks = _base_calculator(-cb)
        lc_speaks = _base_calculator(-simc)
        
        #print("- detecting dips and peaks within light curve")
        dips = _detection(tb, -lc_dips, scb, maxlen, minlen)
        peaks = _detection(tb, -lc_peaks, scb, maxlen, minlen)
        #print("- done!")
        
        #print("- detecting dips and peaks within synthetic data")
        sdips = _detection(tsim, -lc_sdips, ssimc, maxlen, minlen)
        speaks = _detection(tsim, -lc_speaks, ssimc, maxlen, minlen)
        #print("- done!")
        
        dips = dips.dropna().reset_index(drop=True)
        #print("dips")
        peaks = peaks.dropna().reset_index(drop=True)
        #print("peaks")
        sdips = sdips.dropna().reset_index(drop=True)
        #print("simulated dips")
        speaks = speaks.dropna().reset_index(drop=True)
        #print("simulated peaks")
        
        # Store results in the dictionary
        results["dips"].append(dips)
        results["peaks"].append(peaks)
        results["sim_dips"].append(sdips)
        results["sim_peaks"].append(speaks)
        
        #print("candidates collected")
            
        if show_plot:
            
            plt.figure(figsize=(10, 3))
            plt.plot(tb, lc_dips, "r")
            plt.plot(tb[dips.pos], lc_dips[dips.pos], "*k")
                
            plt.plot(tb, -lc_peaks, "b")
            plt.plot(tb[peaks.pos], -lc_peaks[peaks.pos], ".k")
            plt.title("Raw result")
            plt.show()
    
    if len(tb) < 50:
        print("The light curve is too short")
        
#########################################################################################
###########################   STEP 2 : TRAINING AUTOENCODERS   ##########################
#########################################################################################
        
    total_dmse_train=[]
    total_dmse_test=[]
    total_pmse_train=[]
    total_pmse_test=[]

    # Extract the dips, peaks, and simulated dips and peaks for this file
    dips_list = results.get("dips", [])
    peaks_list = results.get("peaks", [])
    sim_dips_list = results.get("sim_dips", [])
    sim_peaks_list = results.get("sim_peaks", [])

    ############ COLLECT RESULTS
    # Initialize empty DataFrames for cleaned dips and peaks
    dips_to_clean = pd.DataFrame()
    peaks_to_clean = pd.DataFrame()

    # Assign the lists of data directly
    dips = pd.concat(dips_list) if isinstance(dips_list, list) else dips_list
    sdips = pd.concat(sim_dips_list) if isinstance(sim_dips_list, list) else sim_dips_list
    peaks = pd.concat(peaks_list) if isinstance(peaks_list, list) else peaks_list
    speaks = pd.concat(sim_peaks_list) if isinstance(sim_peaks_list, list) else sim_peaks_list
    
    #print("evaluating candidates")
    # Process dips if both dips and sdips contain data
    if not dips.empty and not sdips.empty:
    
        dips_, dmse_test, dmse_train = _clean_autoencoder(dips, sdips, show_plot=show_plot)
        dips_to_clean = pd.concat([dips_to_clean, dips_])
        total_dmse_train.append(dmse_train)
        total_dmse_test.append(dmse_test)

    # Process peaks if both peaks and speaks contain data
    if not peaks.empty and not speaks.empty:
    
        peaks_, pmse_test, pmse_train = _clean_autoencoder(peaks, speaks, show_plot=show_plot)
        peaks_to_clean = pd.concat([peaks_to_clean, peaks_])
        total_pmse_train.append(pmse_train)
        total_pmse_test.append(pmse_test)

    ############# SAVE RESULTS
    # Identify potential peaks and dips with a high outlier probability (over 90%)
    pospeaks = peaks_to_clean[(peaks_to_clean.is_outlier==True )] if not peaks_to_clean.empty else None
    posdips = dips_to_clean[(dips_to_clean.is_outlier==True )] if not dips_to_clean.empty else None

    if show_plot:
        # Plot dips and peaks, marking positions with 90%+ probability
        plt.figure(figsize=(10, 3))

        # Plot dips with high probability if available
        plt.plot(tb, cb, "k", label="Dips")
        if posdips is not None and not posdips.empty:
            plt.plot(tb[posdips.pos], cb[posdips.pos], "*r", label="High-prob Dips")
        if pospeaks is not None and not pospeaks.empty:
            plt.plot(tb[pospeaks.pos], cb[pospeaks.pos], ".r", label="High-prob Peaks")

        # Add legend and title
        plt.title("Outliers")
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


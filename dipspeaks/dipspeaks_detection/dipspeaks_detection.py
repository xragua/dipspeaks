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
import matplotlib.pyplot as plt

# SciPy for scientific computing

# Scikit-learn for machine learning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer
# TensorFlow and Keras for deep learning

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
    _real_probability,
)


#########################################################################################
#########################################################################################
#########################################################################################

def detect_dips_and_peaks(lc, snr=0.15 ,index_time=0, index_rate=1, index_error_rate=2, num_simulations=1, show_plot = True):
    """    
    Detects dips and peaks in a given light curve using signal-to-noise ratio thresholding,
    synthetic data generation, and autoencoder-based anomaly detection.

    Parameters:
    -----------
    lc : str
        File path to the input light curve (LC) data in text format.

    snr : float, optional (default=0.15)
        The signal-to-noise ratio (SNR) threshold used to rebin the light curve data.

    index_time : int, optional (default=0)
        Column index for time data in the LC file.

    index_rate : int, optional (default=1)
        Column index for count rate data in the LC file.

    index_error_rate : int, optional (default=2)
        Column index for error in count rate data in the LC file.

    num_simulations : int, optional (default=1)
        Number of synthetic data simulations to generate for noise estimation and anomaly detection.

    show_plot : bool, optional (default=True)
        Whether to display plots illustrating the detection process and results.

    Returns:
    --------
    peaks_to_clean : pandas.DataFrame
        Detected peaks from the original LC data, along with calculated anomaly scores.

    dips_to_clean : pandas.DataFrame
        Detected dips from the original LC data, along with calculated anomaly scores.

    lcreb : pandas.DataFrame
        Rebinned LC data after applying the SNR threshold.

    speaks_to_clean : pandas.DataFrame
        Detected peaks from the synthetic (simulated) LC data, for comparative analysis.

    sdips_to_clean : pandas.DataFrame
        Detected dips from the synthetic (simulated) LC data, for comparative analysis.
    """

    num_simulations=num_simulations
    
    #lOAD LC............................................................................
    series = pd.read_csv(lc, sep=r'\s+', header=None, skiprows=21, comment='!')#lOAD LC

    t = np.array(series[index_time])
    c = np.array(series[index_rate])
    sc = np.array(series[index_error_rate])
    
    # Filter out invalid data
    mask = np.where((sc > 0) & (c > 0))[0]
    t, c, sc = t[mask], c[mask], sc[mask]

    print("Creating syntetic data")
    tsim_, simc_, ssimc_ =  _calculate_synthetic_data(t, c, sc,num_simulations)
    tsim = np.array(tsim_)
    simc = np.array(simc_)
    ssimc = np.array(ssimc_)

    mask = np.where((simc > 0) & (ssimc > 0))[0]
    tsim, simc, ssimc = tsim[mask], simc[mask], ssimc[mask]


    duration_lc=max(t)-min(t)
    duration_slc=max(tsim)-min(tsim)
        
    print('- done!')
    results = {'dips': [], 'peaks': [], 'sim_dips': [], 'sim_peaks': []}
    
    snr_thresholds = [snr]
    
    #Rebin all data .........................................................................
    print("Rebin light curve and syntetic lightcurve to the desired sn")
    for r in snr_thresholds:
    
        # Rebin data and noise
        tb, cb, scb = rebin_snr(t, c, sc, r)
        lcreb = pd.DataFrame({
            't': tb,
            'c': cb,
            'sc': scb
        })

        cb = np.array(cb)
        tb = np.array(tb)
        scb = np.array(scb)

        tsim, simc, ssimc = rebin_snr(tsim, simc, ssimc, r)

        tsim = np.array(tsim)
        simc = np.array(simc)
        ssimc = np.array(ssimc)

        plt.show()

        print("Done!")
        #Collect candidates ......................................................................
        if len(tb) >= 50:

            print('Calculate bases for dip/peak detection')
            lc_dips = _base_calculator(cb)
            lc_peaks = _base_calculator(-cb)

            lc_speaks = _base_calculator(-simc)
            lc_sdips = _base_calculator(simc)
            print('- done!')

            print('- detecting dips and peaks within light curve and syntetic lightcurve')
            dips = _detection(tb, -lc_dips, scb)
            peaks = _detection(tb, -lc_peaks, scb)

            sdips = _detection(tsim, -lc_sdips, ssimc)
            speaks = _detection(tsim, -lc_speaks, ssimc)
            
            
            dips = dips.dropna().reset_index(drop=True)
            peaks = peaks.dropna().reset_index(drop=True)
            sdips = sdips.dropna().reset_index(drop=True)
            speaks = speaks.dropna().reset_index(drop=True)
            
            results['dips'].append(dips)
            results['peaks'].append(peaks)
            results['sim_dips'].append(sdips)
            results['sim_peaks'].append(speaks)
            print('- done!')

            
            #print('candidates collected')
                
            if show_plot:
        
                plt.figure(figsize=(20, 3))   
                plt.plot(tb,cb,"k",alpha=0.8) 
                plt.plot(tb, -lc_peaks, 'b:', label="base_for_peaks")
                plt.plot(tb[peaks.pos], -lc_peaks[peaks.pos], '.g', markersize=1)
                plt.xlabel("Time (s)")
                plt.ylabel("Rate")
                plt.title('Raw result light curve')

                plt.plot(tb, lc_dips, 'r:', label="base_for_dips")
                plt.plot(tb[dips.pos], lc_dips[dips.pos], '*g', markersize=1)
                plt.show()

                plt.figure(figsize=(20, 3))
                plt.plot(tsim,simc,"k",alpha=0.8) 
                plt.plot(tsim, -lc_speaks, 'b:',label="simulated base for peaks")
                plt.plot(tsim[speaks.pos], -lc_speaks[speaks.pos], '.g', markersize=1)
                plt.xlabel("Time (s)")
                plt.ylabel("Rate")
                plt.title('Raw result simulated light curve')
                
                plt.plot(tsim, lc_sdips, 'r:',label="simulated base for dips")
                plt.plot(tsim[sdips.pos], lc_sdips[sdips.pos], '*g', markersize=1)
                plt.show()
                    
        if len(tb) < 50:
            # \033[91m turns on bright red, \033[0m resets to default
            print("\033[91mThe light curve is too short, please try a higher sn.\033[0m")

            break
        
#########################################################################################
###########################   STEP 2 : TRAINING AUTOENCODERS   ##########################
#########################################################################################

    print("Train auto-encoders in syntetic data")    
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
    peaks = pd.concat(peaks_list) if isinstance(peaks_list, list) else peaks_list

    sdips = pd.concat(sim_dips_list) if isinstance(sim_dips_list, list) else sim_dips_list
    speaks = pd.concat(sim_peaks_list) if isinstance(sim_peaks_list, list) else sim_peaks_list

    dips=dips.sort_values(by='t', ascending=False).reset_index(drop=True)
    peaks=peaks.sort_values(by='t', ascending=False).reset_index(drop=True)

    sdips=sdips.sort_values(by='t', ascending=False).reset_index(drop=True)
    speaks=speaks.sort_values(by='t', ascending=False).reset_index(drop=True)



    #print('evaluating candidates')
    # Process dips if both dips and sdips contain data
    print("DIPS----------------------------------------------------------------------------------------")
    if not dips.empty and not sdips.empty:
        dips_, dmse_test, dmse_train = _clean_autoencoder(dips, sdips, show_plot=show_plot,show_plot_eval=True)
        sdips_, sdmse_test, sdmse_train = _clean_autoencoder(sdips, sdips, show_plot=show_plot,show_plot_eval=False)
        dips_to_clean = pd.concat([dips_to_clean, dips_])
        sdips_to_clean = pd.concat([sdips_to_clean, sdips_])


    print("PEAKS---------------------------------------------------------------------------------------")
    # Process peaks if both peaks and speaks contain data
    if not peaks.empty and not speaks.empty:
        peaks_, pmse_test, pmse_train = _clean_autoencoder(peaks, speaks, show_plot=show_plot,show_plot_eval=True)
        speaks_, spmse_test, spmse_train = _clean_autoencoder(speaks, speaks, show_plot=show_plot,show_plot_eval=False)
        peaks_to_clean = pd.concat([peaks_to_clean, peaks_])
        speaks_to_clean = pd.concat([speaks_to_clean, speaks_])

    th_zscore=3
    th_error_percentile=0.99


    ############# SAVE RESULTS
    # Identify potential peaks and dips with a high outlier probability (over 90%)
    pospeaks = peaks_to_clean[(peaks_to_clean.zscores>th_zscore) & (peaks_to_clean.error_percentile>th_error_percentile)] if not peaks_to_clean.empty else None
    posdips = dips_to_clean[(dips_to_clean.zscores>th_zscore) & (dips_to_clean.error_percentile>th_error_percentile)] if not dips_to_clean.empty else None
    
    spospeaks = speaks_to_clean[(speaks_to_clean.zscores>th_zscore) & (speaks_to_clean.error_percentile>th_error_percentile)] if not speaks_to_clean.empty else None
    sposdips = sdips_to_clean[(sdips_to_clean.zscores>th_zscore) & (sdips_to_clean.error_percentile>th_error_percentile)] if not sdips_to_clean.empty else None
    
    real_rate_peaks = (len(pospeaks) / len(peaks_to_clean)) if len(peaks_to_clean) > 0 else 0
    sim_rate_peaks  = (len(spospeaks) / len(speaks_to_clean))  if len(speaks_to_clean) > 0 else 0

    real_rate_dips  = (len(posdips) / len(dips_to_clean))    if len(dips_to_clean) > 0 else 0
    sim_rate_dips   = (len(sposdips) / len(sdips_to_clean))   if len(sdips_to_clean) > 0 else 0
    
    ppeaks = _real_probability(real_rate_peaks, sim_rate_peaks)
    pdips = _real_probability(real_rate_dips, sim_rate_dips)


    print("Simulation:")
    print("Peaks per second:", np.round(len(spospeaks)/duration_slc,4),
          "percentage of rejected peaks:", np.round((len(speaks_to_clean)-len(spospeaks))/len(speaks_to_clean),4))
          
    print("Dips per second:", np.round(len(sposdips)/duration_slc,4),
          "percentage of rejected dips:", np.round((len(sdips_to_clean)-len(sposdips))/len(sdips_to_clean),4))
          
    print("Result:")
    print("Peaks per second:", np.round(len(pospeaks)/duration_lc,4),
          "percentage of rejected peaks:", np.round((len(peaks_to_clean)-len(pospeaks))/len(peaks_to_clean),4),
          "probability of detected peaks:",np.round(ppeaks))
          
    print("Dips per second:", np.round(len(posdips)/duration_lc,4),
          "percentage of rejected dips:", np.round((len(dips_to_clean)-len(posdips))/len(dips_to_clean),4),
          "probability of detected dips:",np.round(pdips,2))
          


    if show_plot:
        # Plot dips and peaks, marking positions with 90%+ probability
        plt.figure(figsize=(20, 3))

        # Plot dips with high probability if available
        plt.plot(tb, cb, 'k', label='Dips',alpha=0.2)
        if posdips is not None and not posdips.empty:
            plt.plot(posdips.t,cb[posdips.pos], ".b",label='High-prob Dips')
        if pospeaks is not None and not pospeaks.empty:
            plt.plot(pospeaks.t,cb[pospeaks.pos],  "r.", label='High-prob Peaks')

        # Add legend and title
        plt.title('Outliers')
        plt.legend()
        plt.show()

    return peaks_to_clean, dips_to_clean, lcreb, speaks_to_clean, sdips_to_clean




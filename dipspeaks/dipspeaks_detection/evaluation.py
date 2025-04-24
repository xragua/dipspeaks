#PKGS
##########################################################################################
##########################################################################################
##########################################################################################
# Import Libraries
# Standard libraries
import warnings

# Data manipulation and analysis
import numpy as np

from scipy.stats import norm
# Scikit-learn for machine learning
import matplotlib.pyplot as plt

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


def _outlier_probability(
        errors_train,
        errors,
        times,
        show_plot,
        show_plot_eval
        ):
    
    # ---------- criterio Modified‑Z ----------
    z_scores   = _modified_z_score(errors_train, errors)
    z_scores_train = _modified_z_score(errors_train, errors_train)

    sorted_train = np.sort(errors_train)

    ranks = np.searchsorted(sorted_train, errors, side='right')
    percentiles = ranks / errors_train.size

    ranks_train= np.searchsorted(sorted_train, errors_train, side='right')
    percentiles_train = ranks_train / errors_train.size

    if (show_plot_eval & show_plot):
        plt.figure()
        plt.figure(figsize=(20, 3))
        plt.title("Percentiles")

        plt.plot(times, percentiles,".",markersize=1)

        plt.hlines(y=0.99,xmin=min(times), xmax=(max(times)),color="black", label="99% percentile")
        plt.hlines(y=0.75,xmin=min(times), xmax=(max(times)), color= "blue", label="75% percentile")
        plt.hlines(y=0.50,xmin=min(times), xmax=(max(times)), color="green", label="50% percentile")

        plt.legend()
        plt.show()


    return z_scores, percentiles
    
def _real_probability(real_rate, sim_rate):
    """
    Estimate the probability that a detected event is real,
    based on the real and simulation detection rates.
    """
    if real_rate == 0:
        return 0.0  # No events detected, so probability of being real is zero
    return max(0.0, min(1.0, (real_rate - sim_rate) / real_rate))
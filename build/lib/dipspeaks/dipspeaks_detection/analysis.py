import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from .evaluation import(

    _real_probability,
)

from .helper_functions import(
    scale
)

def overlap(high, low):
    """
    Compute pairwise overlap durations and overlap ratios between two sets features (dips or peaks).
    Useful to analyze the behaviour of the dips/peaks and their presence in different energy ranges.
        
        Parameters
    ----------
    high : pandas.DataFrame
        DataFrame of “high-energy” features (peaks/dips). Must contain columns:
        - **ti** : start time of each feature (array-like of shape (n_high,))
        - **te** : end time of each feature (array-like of shape (n_high,))
    low : pandas.DataFrame
        DataFrame of “low-energy” features. Must contain columns:
        - **ti** : start time of each feature (array-like of shape (n_low,))
        - **te** : end time of each feature (array-like of shape (n_low,))

    Returns
    -------
    overlap_durations : numpy.ndarray, shape (n_overlaps,)
        Duration of each overlapping interval (zero-length overlaps dropped).
    high_indices : numpy.ndarray, shape (n_overlaps,)
        Indices into `high` indicating which high-energy feature is involved.
    low_indices : numpy.ndarray, shape (n_overlaps,)
        Indices into `low` indicating which low-energy feature is involved.
    high_overlap_ratio : numpy.ndarray, shape (n_overlaps,)
        Overlap durations normalized by the duration of the corresponding high-energy feature:
        $$\frac{\text{overlap}}{\text{high.te} - \text{high.ti}}.$$
    low_overlap_ratio : numpy.ndarray, shape (n_overlaps,)
        Overlap durations normalized by the duration of the corresponding low-energy feature:
        $$\frac{\text{overlap}}{\text{low.te} - \text{low.ti}}.$$

    """
    high_start = np.array(high.ti)
    high_end = np.array(high.te)
    low_start = np.array(low.ti)
    low_end = np.array(low.te)

    # Compute the pairwise maximum of start times and minimum of end times
    max_start = np.maximum(high_start[:, np.newaxis], low_start)
    min_end = np.minimum(high_end[:, np.newaxis], low_end)

    # Overlap length (zero if negative)
    overlap = np.maximum(0, min_end - max_start)

    # Normalize by each interval’s length
    hoverlap = overlap / (high_end[:, np.newaxis] - high_start[:, np.newaxis])
    loverlap = overlap / (low_end - low_start)

    # Select only truly overlapping pairs
    high_indices, low_indices = np.where(overlap > 0)

    return (
        overlap[high_indices, low_indices],
        high_indices,
        low_indices,
        hoverlap[high_indices, low_indices],
        loverlap[high_indices, low_indices],
    )



def filter_dip_peak(
    dataset: pd.DataFrame,
    simdataset: pd.DataFrame,
    lc_reb: object,
    error_percentile_threshold: float = 0.99,
    zscore_threshold: float = 4,
    show_plot: bool = True
) -> pd.DataFrame:
    """
    Filter detected dips/peaks by reconstruction‑error percentile and z‑score,
    optionally plot them, and estimate their “real” probability.

    Parameters
    ----------
    dataset : pd.DataFrame
        Detected features on the real light curve. Must contain columns:
        - `error_percentile` (float): percentile ranking of the reconstruction error.
        - `zscores` (float): z‑score of the feature.
        - `pos` (int): index into `lc_reb.t` / `lc_reb.c` for plotting.
        - `t` (float): time coordinate for rate calculation.
    simdataset : pd.DataFrame
        Detected features on a noise‑only (synthetic) light curve. Must contain `t`.
    lc_reb : object
        Re-binned light curve object with attributes:
        - `t` (array‑like): time stamps.
        - `c` (array‑like): measured flux/rate.
    error_percentile_threshold : float, optional
        Minimum error‑percentile (e.g. 0.9 means above the 90th percentile)
        to keep a feature. Default is 0.9.
    zscore_threshold : float, optional
        Minimum z‑score to keep a feature. Default is 4.
    show_plot : bool, optional
        If True, overplot the filtered features on the light curve.
        Default is True.


    Returns
    -------
    pd.DataFrame
        Subset of `dataset` passing both thresholds, with index reset.

    Prints the probability of the dataset based on the rate (features/s) in the filtered light curve vs the rate of the filtered sysntetic light curve.
    """
    # Select features above both thresholds
    filt = dataset[
        (dataset.error_percentile >= error_percentile_threshold) &
        (dataset.zscores >= zscore_threshold)
    ].reset_index(drop=True)

    sfilt = simdataset[
        (simdataset.error_percentile >= error_percentile_threshold) &
        (simdataset.zscores >= zscore_threshold)
    ].reset_index(drop=True)

    if show_plot:
        plt.figure(figsize=(20, 3))
        plt.plot(lc_reb.t, lc_reb.c, label='Light Curve')
        plt.scatter(
            lc_reb.t[filt.pos],
            lc_reb.c[filt.pos],
            marker='*',
            s=100,
            color='red',
            label='Filtered Events'
        )
        plt.xlabel("Time")
        plt.ylabel("Rate")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Compute observation durations
    real_rate = len(filt)/(dataset['t'].max() - dataset['t'].min())
    sim_rate  = len(sfilt)/(simdataset['t'].max() - simdataset['t'].min())

    # Estimate probability
    p = _real_probability(real_rate, sim_rate)

    print(f"The probability of this filtered dataset is {p:.4f}")

    return filt



def gmm_dips_peaks(good_pd, lc, log_scale=False, show_plot=False):
    """
    Perform Gaussian Mixture Model (GMM) clustering on the provided data 
    and compute cluster statistics.

    Parameters:
    - good_pd: DataFrame containing the data to cluster.
    - log_scale: Whether to apply logarithm transformation to the data.

    Returns:
    - cluster_stats_df: DataFrame containing the statistics for each cluster.
    - cluster_labels: Array of cluster labels for each data point.
    """
    
    # Check if the input DataFrame is empty
    if good_pd.empty:
        print("The input DataFrame is empty.")
        return pd.DataFrame(), np.array([])  # Return empty DataFrame and empty array

    # Select columns for clustering
    selected_columns = ['prominence', 'duration']
    data = good_pd[selected_columns]
    data_plot = good_pd[selected_columns]

    # Apply logarithmic transformation if specified
    if log_scale:
        if (data <= 0).any().any():  # Check for non-positive values
            print("Log transformation cannot be applied due to non-positive values in the data.")
            return pd.DataFrame(), np.array([])  # Return empty DataFrame and empty array
        data = np.log(data)

    # Normalize the data
    normalized_data = (data - data.mean(axis=0)) / data.std(axis=0)

    # Range of clusters to test
    k_range = range(2, min(10, max(int(len(data) - 2), 2)))

    # Silhouette Scores for GMM
    silhouette_scores = []

    for k in k_range:
        try:
            gmm = GaussianMixture(n_components=k, n_init=100, random_state=0, tol=1e-5)
            gmm.fit(normalized_data)
            cluster_labels = gmm.predict(normalized_data)
            silhouette_avg = silhouette_score(normalized_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        except:
            continue  # Skip the iteration if clustering fails

    # Check if silhouette scores are available
    if not silhouette_scores:
        print("No valid silhouette scores were calculated.")
        return pd.DataFrame(), np.array([])  # Return empty DataFrame and empty array

    # Determine optimal number of clusters
    optimal_clusters = np.argmax(silhouette_scores) + 2  # Adjust for k_range starting from 2
    
    # Fit GMM with the optimal number of clusters
    gmm = GaussianMixture(n_components=optimal_clusters, n_init=10, random_state=0)
    gmm.fit(normalized_data)
    cluster_labels = gmm.predict(normalized_data)

    
    # Calculate statistics for each cluster
    cluster_stats = []
    optimal_clusters=optimal_clusters
    if log_scale:
        data = np.exp(data)  # Reverse the log transformation for correct statistics
    for i in range(optimal_clusters):
        cluster_data = data.loc[cluster_labels == i]
        mean = cluster_data.mean(axis=0)
        std = cluster_data.std(axis=0)
        number = len(cluster_data)
        cluster_stats.append({
            'Cluster': i,
            'Mean': mean,
            'Standard Deviation': std,
            'Number': number
        })

    # Create a DataFrame with the results
    cluster_stats_df = pd.DataFrame(cluster_stats)
    mean_df = cluster_stats_df['Mean'].apply(pd.Series)
    std_df = cluster_stats_df['Standard Deviation'].apply(pd.Series)
    mean_df.columns = [f'Mean_{col}' for col in mean_df.columns]
    std_df.columns = [f'Std_{col}' for col in std_df.columns]

    # Concatenate mean and std columns with the cluster_stats_df
    cluster_stats_df = pd.concat([cluster_stats_df[['Cluster', 'Number']], mean_df, std_df], axis=1)

    if show_plot:
        # Plotting the Silhouette scores and clustering results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
        axes[0].plot(k_range, silhouette_scores, marker='.', linestyle='-')
        axes[0].set_xlabel('Number of Components (k)')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].set_title('Silhouette Score', fontsize=10)
        
        scatter = axes[1].scatter(data_plot['duration'], data_plot['prominence'], c=cluster_labels, marker=".", cmap='rainbow', alpha=0.7)
        axes[1].set_xlabel('Duration')
        axes[1].set_ylabel('Prominence')
        axes[1].set_title('Clustering', fontsize=10)
        plt.show
        
        if log_scale:
            axes[1].set_xscale("log")
            axes[1].set_yscale("log")
            
        #Plot lightcurve and different clusters
        plt.figure(figsize=(11, 3))
        plt.plot(lc.t, lc.c,"k",alpha=1)
        plt.scatter(good_pd['t'], lc.c[good_pd['pos']], c=cluster_labels, marker="o", cmap='rainbow', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        display(np.round(cluster_stats_df,3))

    return cluster_stats_df, cluster_labels



def clump_candidates(high_dips, low_dips, lc, overlap_threshold=0.75, bin_number=100, show_plot=False):

    """
    Identify clumped dips between high- and low-energy features based on overlap and relative prominence.

    This function finds dip pairs from two energy bands (`high_dips` and `low_dips`) that overlap
    by at least `overlap_threshold` in both directions and where the low-energy dip has greater
    relative prominence than the high-energy dip. Optionally, it can display diagnostic histograms
    of dip times overlaid with the light curve.

    Parameters
    ----------
    high_dips : pandas.DataFrame
        DataFrame of high-energy dip features. Must contain columns:
        - `ti`, `te`: start and end times of each dip
        - `relprominence`: relative prominence of each dip
        - `t`: representative time of the dip (for plotting)
    low_dips : pandas.DataFrame
        DataFrame of low-energy dip features with the same columns as `high_dips`.
    lc : pandas.DataFrame
        Light curve DataFrame used for context in plotting. Must contain:
        - `t`: time array
        - `c`: count (flux) array
    overlap_threshold : float, default=0.75
        Minimum fractional overlap (in both high→low and low→high) required to consider two dips overlapping.
    bin_number : int, default=100
        Number of bins to use when plotting histograms of dip times.
    show_plot : bool, default=False
        If True, display histograms of dip times for both energy bands, overlaid with the scaled light curve.

    Returns
    -------
    high_clump : pandas.DataFrame
        Subset of `high_dips` that meet the clump criteria.
    low_clump : pandas.DataFrame
        Corresponding subset of `low_dips` that pair with `high_clump`."""

    
    _, high_dip_idx_, low_dip_idx_, percentaje_high, percentaje_low = overlap(high_dips,  low_dips)
    
    high_dip_idx = high_dip_idx_[(percentaje_high>overlap_threshold)&(percentaje_low>overlap_threshold)]
    low_dip_idx = low_dip_idx_[(percentaje_high>overlap_threshold)&(percentaje_low>overlap_threshold)]
    
    high_dips_overlap = high_dips.loc[high_dip_idx].reset_index(drop=True)
    low_dips_overlap = low_dips.loc[low_dip_idx].reset_index(drop=True)

    clump_index = low_dips_overlap.relprominence/high_dips_overlap.relprominence >1
    
    high_clump = high_dips_overlap.loc[clump_index].reset_index(drop=True)
    low_clump = low_dips_overlap.loc[clump_index].reset_index(drop=True)


    if show_plot:
        plt.figure(figsize=(20, 3)) 
        
        hcounts, _,_ = plt.hist(high_dips.t, bins=bin_number, alpha=0.5, label='high energy ')
        lcounts, _,_ = plt.hist(low_dips.t, bins=bin_number, alpha=0.5, label='low energy')

        max_count= max(max(hcounts), max(lcounts))
        plt.plot(lc.t, scale(lc.c,[0,max_count]), "k", alpha=0.2)

        plt.legend()
        plt.show()

    return high_clump, low_clump


def overlap_percentaje(high, low, lc, percentaje = 0.5, show_plot=False):

    overlap_duration, high_idx_, low_idx_, percentaje_high, percentaje_low = overlap(high,  low)

    idx = np.where((percentaje_high > percentaje)&(percentaje_low > percentaje))

    high_indices = high_idx_[idx]
    low_indices = low_idx_[idx]

    if show_plot:
        plt.figure(figsize=(20, 3)) 
        plt.plot(lc.t, lc.c, "k", alpha=0.2)
        plt.vlines(x= high.t[high_indices], ymax=max(lc.c), ymin=min(lc.c), color="red", label="High energy dataset" )
        plt.vlines(x= low.t[low_indices], ymax=max(lc.c), ymin=min(lc.c) , color="green", label="Low energy dataset" )
        plt.legend()
        plt.show()


    return (
        overlap_duration[idx], 
        high_indices,
        low_indices,
        percentaje_high[idx], 
        percentaje_low[idx]
    )

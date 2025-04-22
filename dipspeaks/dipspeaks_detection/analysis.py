import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .evaluation import(
    _modified_z_score,
    _outlier_probability,
    _real_probability,
)

def overlap(high_start, high_end, low_start, low_end):
    """
    Compute pairwise overlap durations and overlap ratios between two sets of intervals.

    For each interval in the ‘high’ list and each in the ‘low’ list, this function
    calculates the overlap length and returns only those pairs that actually overlap.

    Parameters
    ----------
    high_start : array_like of shape (N,)
        Start times of the high-interval segments.
    high_end : array_like of shape (N,)
        End times of the high-interval segments.
    low_start : array_like of shape (M,)
        Start times of the low-interval segments.
    low_end : array_like of shape (M,)
        End times of the low-interval segments.

    Returns
    -------
    overlap : ndarray of shape (K,)
        Overlap duration for each overlapping high–low interval pair.
    high_indices : ndarray of shape (K,)
        Indices into `high_start`/`high_end` corresponding to each overlap.
    low_indices : ndarray of shape (K,)
        Indices into `low_start`/`low_end` corresponding to each overlap.
    hoverlap : ndarray of shape (K,)
        Fraction of each high interval that overlaps:
        `overlap / (high_end - high_start)`.
    loverlap : ndarray of shape (K,)
        Fraction of each low interval that overlaps:
        `overlap / (low_end - low_start)`.

    Notes
    -----
    - Any pair with zero or negative overlap is discarded.
    - All inputs are converted to NumPy arrays internally.
    - The returned arrays are flattened so that each entry corresponds
      to one overlapping pair.

    Examples
    --------
    >>> high_s = [0, 5]
    >>> high_e = [3, 8]
    >>> low_s = [1, 6]
    >>> low_e = [4, 10]
    >>> overlap, hi_idx, lo_idx, h_ratio, l_ratio = calculate_overlap_gtp(high_s, high_e, low_s, low_e)
    >>> overlap
    array([2, 2])
    >>> hi_idx
    array([0, 1])
    >>> lo_idx
    array([0, 1])
    >>> h_ratio
    array([2/3, 2/3])
    >>> l_ratio
    array([2/3, 2/4])
    """
    high_start = np.array(high_start)
    high_end = np.array(high_end)
    low_start = np.array(low_start)
    low_end = np.array(low_end)

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
    error_percentile_threshold: float = 0.9,
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

    Side Effects
    ------------
    - If `show_plot`, displays a time‑series plot with detected features.
    - Prints the estimated probability that a detection is real,
      computed by `_real_probability(real_rate, sim_rate)`.

    Notes
    -----
    - `real_rate` and `sim_rate` are each defined as max(t) − min(t).
    - The helper function `_real_probability` should accept these two durations
      and return a float in [0, 1].
    """
    # Select features above both thresholds
    filt = dataset[
        (dataset.error_percentile >= error_percentile_threshold) &
        (dataset.zscores >= zscore_threshold)
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
    real_rate = len(dataset)/(dataset['t'].max() - dataset['t'].min())
    sim_rate  = len(simdataset)/(simdataset['t'].max() - simdataset['t'].min())

    # Estimate probability
    p = _real_probability(real_rate, sim_rate)
    print(f"The probability of this filtered dataset being real is {p:.4f}")

    return filt
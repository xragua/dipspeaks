=======================================================
dipspeaks – Key Public Functions
=======================================================

.. contents::
   :local:
   :depth: 1


clump_candidates
-------------------------------------------------------

Identify clumped dips between high- and low-energy features
based on overlap and relative prominence.

**Parameters**
^^^^^^^^^^^^^^
``high_dips`` : :class:`pandas.DataFrame`
    Columns required:

    * ``ti`` – start time  
    * ``te`` – end time  
    * ``relprominence`` – relative prominence  
    * ``t`` – representative time (for plots)

``low_dips`` : :class:`pandas.DataFrame`
    Same columns as *high_dips*.

``lc`` : :class:`pandas.DataFrame`
    Light-curve context; needs ``t`` (time) and ``c`` (rate).

``overlap_threshold`` : :class:`float`, *default* ``0.75``  
    Minimum fractional overlap (both directions).

``bin_number`` : :class:`int`, *default* ``100``  
    Number of histogram bins when plotting.

``show_plot`` : :class:`bool`, *default* ``False``  
    Show diagnostic histograms if *True*.

**Returns**
^^^^^^^^^^^
``high_clump`` : :class:`pandas.DataFrame`
    Subset of *high_dips* that meet the criteria.

``low_clump`` : :class:`pandas.DataFrame`
    Matching subset from *low_dips*.



filter_dip_peak
------------------------------------------------------

Filter detected dips/peaks by reconstruction-error percentile and z-score,
optionally plot them, and estimate their “real” probability.

**Parameters**
^^^^^^^^^^^^^^
``dataset`` : :class:`pandas.DataFrame`
    Must contain ``error_percentile``, ``zscores``, ``pos``, ``t``.

``simdataset`` : :class:`pandas.DataFrame`
    Synthetic (noise-only) features; needs ``t``.

``lc_reb`` : object  
    Re-binned light curve with attributes ``t`` and ``c``.

``error_percentile_threshold`` : :class:`float`, *default* ``0.9``  
    Keep features above this percentile.

``zscore_threshold`` : :class:`float`, *default* ``4``  
    Keep features with z-score ≥ threshold.

``show_plot`` : :class:`bool`, *default* ``True``  
    Overlay surviving features on the light curve.

**Returns**
^^^^^^^^^^^
:class:`pandas.DataFrame`
    Filtered subset of *dataset* (index reset).

*Side effect –* prints the probability of observing that event rate in noise.



gmm_dips_peaks
-----------------------------------------------------

Perform Gaussian-mixture clustering on dip/peak data.

**Parameters**
^^^^^^^^^^^^^^
``good_pd`` : :class:`pandas.DataFrame`  
    Data to cluster.

``log_scale`` : :class:`bool`  
    If *True*, apply ``log10`` to the features before clustering.

**Returns**
^^^^^^^^^^^
``cluster_stats_df`` : :class:`pandas.DataFrame`  
    Per-cluster statistics.

``cluster_labels`` : :class:`numpy.ndarray`  
    Cluster label for every row in *good_pd*.



overlap
----------------------------------------------

Compute pair-wise overlap durations and ratios between two sets of
features (dips or peaks).

**Parameters**
^^^^^^^^^^^^^^
``high`` : :class:`pandas.DataFrame`  
    Columns ``ti`` and ``te``.

``low`` : :class:`pandas.DataFrame``  
    Same two columns.

**Returns**
^^^^^^^^^^^
``overlap_durations`` : :class:`numpy.ndarray`  
``high_indices``      : :class:`numpy.ndarray`  
``low_indices``       : :class:`numpy.ndarray`  
``high_overlap_ratio``: :class:`numpy.ndarray`  
``low_overlap_ratio`` : :class:`numpy.ndarray`

Math definitions
^^^^^^^^^^^^^^^^
.. math::

   r_\\text{high} \\,=\\,
   \\frac{\\text{overlap}}
        {\\text{high.te} - \\text{high.ti}}
   \\qquad
   r_\\text{low} \\,=\\,
   \\frac{\\text{overlap}}
        {\\text{low.te}  - \\text{low.ti}}



detect_dips_and_peaks
---------------------------------------------------

Detect dips and peaks in a light curve via
S/N thresholding, synthetic-data generation, and
autoencoder-based anomaly detection.

**Parameters**
^^^^^^^^^^^^^^
``lc`` : :class:`str`  
    Path to the input light-curve text file.

``snr`` : :class:`float`, *default* ``0.15``  
``index_time`` : :class:`int`, *default* ``0``  
``index_rate`` : :class:`int`, *default* ``1``  
``index_error_rate`` : :class:`int`, *default* ``2``  
``num_simulations`` : :class:`int`, *default* ``1``  
``show_plot`` : :class:`bool`, *default* ``True``  

**Returns**
^^^^^^^^^^^
``peaks_to_clean``   : :class:`pandas.DataFrame`  
``dips_to_clean``    : :class:`pandas.DataFrame`  
``lcreb``            : :class:`pandas.DataFrame`  
``speaks_to_clean``  : :class:`pandas.DataFrame`  
``sdips_to_clean``   : :class:`pandas.DataFrame`  



rebin_snr
--------------------------------------------------------

Re-bin a signal to achieve a target S/N threshold.

**Parameters**
^^^^^^^^^^^^^^
``t``                : array-like (time)  
``x``                : array-like (signal)  
``sy``               : array-like (uncertainty)  
``snr_threshold``    : :class:`float`

**Returns**
^^^^^^^^^^^
``t_new`` : array-like  
``c_new`` : array-like  
``sc_new``: array-like  


scale
----------------------------------------------------

Linearly scale *x* so that its range matches *y* (useful for overlays).

**Parameters**
^^^^^^^^^^^^^^
``x`` : array-like  
``y`` : array-like

**Returns**
^^^^^^^^^^^
``x_new`` : array-like

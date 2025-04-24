
.. contents::
   :local:
   :depth: 1


The main characteristic of this package is the hability of detecting subtle features from lightcurves. 
The main idea is very simple:

In a light curve we can detect features caused by astrophisical phenoma and features caused by random noise. If we learn how this random noise features look like, we can discard them if we detect them in the light curve.

It is key to obtain good synthetic data to compare properly. This is how we do so:

===============================================================
How dipspeaks builds its synthetic (“noise-only”) light curves
===============================================================


The routine generates many light-curve realisations that *preserve only
the noise properties* of the observation, while intentionally destroying
every real dip, peak, or long-term trend.  
These curves train the auto-encoder, teaching it **what noise looks like** so
that true astrophysical events stand out as anomalies.

Algorithm – step by step
------------------------

1. **Isolate the fast noise**

   * Apply a high-pass Butterworth filter (cut-off = **5000 s** by default).
   * Padding is *reflected* so the filter has no edge artefacts.
   * The result is a *residual* series ``resid`` containing just noise.

2. **Clip crazy samples**

   Any residual with ``|z| > 3`` is replaced by a random “safe” sample with
   ``|z| < 1``.  
   This prevents true dips/peaks from leaking into the noise pool.

3. **Store fractional errors**

   .. math::

      sc_{\mathrm{prop}} \;=\; \frac{sc}{c}

   This relative uncertainty is reused later so the synthetic curve keeps the
   same heteroscedasticity as the data.

4. **Repeat for each simulation**

   * Shuffle ``resid`` ⇒ breaks temporal coherence.  
   * Shuffle the vector of **time differences** so the overall cadence pattern
     is preserved but the order is random.  
   * Re-scale errors::

        ssimc = |sc_prop_shuffled × simc|

     and clip outliers with the same z-score rule.


Why it works
------------

* High-pass filtering decouples slow orbital/instrumental trends.
* Shuffling destroys any real variability but leaves the noise distribution
  untouched.
* Outlier clipping guards against residual real events.
* Re-using **sc / c** keeps the correct error-vs-flux scaling.

The resulting curves are therefore ideal *negative* examples for the
auto-encoder’s anomaly-detection stage.

===============================================================
How the auto-encoder scores dips & peaks
===============================================================

Why an auto-encoder?
--------------------
An **auto-encoder** is a tiny neural network that tries to copy its input back
to itself.  
If it is trained only on *noise-like* examples, it becomes very good at
reproducing noise — and bad at reproducing anything that doesn’t look
like the training set.  
The reconstruction error therefore acts as an *anomaly score*.

In dipspeaks we **train the auto-encoder on the synthetic, noise-only
features** and then ask it to reconstruct the features found in the **real**
light curve.


Workflow of ``_clean_autoencoder``
----------------------------------

1. **Select a compact feature vector**

   The four columns

   * ``prominence`` – depth or prominence  
   * ``duration`` – width in seconds  
   * ``density`` – (depth/prominence) / duration  
   * ``snr`` – local signal-to-noise  

   capture each dip/peak in a 4-D point.

2. **Build a symmetric auto-encoder**

   =========================  =====================
   Encoder                    Decoder
   -------------------------  ---------------------
   256 → 128 → 64 → 32 → 16   32 ← 64 ← 128 ← 256
   =========================  =====================

   * all layers use **ELU** activations  
   * loss = **mean-absolute-error**  
   * early-stopping & LR-plateau callbacks guard against over-fitting

3. **Train only on the *baseline* set**

   ``pd_base`` comes from the **synthetic light curve**, so by definition it is
   “noise”.  After ~hundreds of epochs the AE can reconstruct these vectors
   with tiny error.

4. **Score the real features**

   * Calculate **MSE** between each real vector and its reconstruction.  
   * Compare that distribution with the training error distribution.  
   * Convert to

     * **z-scores** (standard deviations from the training mean)  
     * **percentiles** (how extreme each error is w.r.t. noise)

5. **Augment the DataFrame**

   Two new columns are added:

   * **zscores**    Standard-score of the reconstruction error.
   * **error_percentile**  Position of that error in the cumulative distribution of the syntetic dataset

   High values in *either* column mark a likely real dip/peak.

Typical thresholds
------------------

* ``zscores   > 3``  
* ``percentile > 0.99``

But, in the synthetic light curve we will still find features over a 0.99 percentile error and high zscore, right?
Yes, but to evaluate the probability of the dataset by comparing the **rate** (filterd fetures/s) in the real light curve vs 
in the synthetic light curve.


===============================================================
Probability based on an excess
===============================================================

Once dips or peaks have passed the auto-encoder’s outlier test we still need a
*sanity check*:  
**How often would noise alone deliver the same number of survivors?**

The idea is simple:

1. **Count what survives in the real data**  
   :math:`R_\text{real}`   = “events per second” after all cuts.

2. **Count what survives in a noise-only light curve**  
   :math:`R_\text{sim}`   = the *false-positive* rate our pipeline produces
   when there is, by construction, nothing to detect.

3. **Compare the two**  
   The larger the gap between the real data rate vs the synthetic rate, the more confident we are that
   the events in the real data are **not** random noise.

A linear confidence score
-------------------------

We convert the comparison into a probability-like number

.. math::

   \text{confidence} \;=\;
   \frac{R_\text{real} - R_\text{sim}}{R_\text{real}}
   \quad\in\;[0,1]

* **1.0**  → the synthetic (noise) curve produced **zero** such events.  
* **0.0**  → everything you see in the real curve is equally common in noise.  
* Values in between scale linearly with the “excess” over the noise rate.

By varying the thresholds, we can check the probability of the filtered data set (using the function **filter_dip_peak**).

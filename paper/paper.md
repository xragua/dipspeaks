---
title: 'dipspeaks: A Python Package for Detecting and Analyzing relevant dips and peaks in light curves'
tags:
  - Python
  - X-ray astronomy
  - Timing analysis
  - light curves
authors:
  - given-names: Graciela
    surname: Sanjurjo-Ferrín
    orcid: 0009-0001-0378-7879
    equal-contrib: true
    affiliation: 1
  - given-names: Jessica 
    surname: Planelles Villalva
    orcid: 0009-0007-9988-0202
    equal-contrib: true
    affiliation: 1
  - given-names: Jose Miguel 
    surname: Torrejón
    orcid: 0000-0002-5967-5163
    equal-contrib: true
    affiliation: 1
  - given-names: Jose Joaquín 
    surname: Rodes-Roca
    orcid: 0000-0003-4363-8138
    equal-contrib: true
    affiliation: 1

affiliations:
 - name: Instituto Universitario de Física Aplicada a las Ciencias y las Tecnologías, Universidad de Alicante, 03690 Alicante, Spain
   index: 1

date: 11 November 2024
bibliography: paper.bib

---


# Summary

X-ray astronomy is a young discipline, spanning no more than a few decades. The Earth's atmosphere is opaque to this type of radiation, so observations in this part of the spectrum were delayed until the space era began, when rocket launchers carrying X-ray telescopes revealed the universe from a brand-new point of view [@1962PhRvL.9.439G].

X-ray binaries are tremendously interesting systems. In these extreme environments, a compact object (either a neutron star, a black hole or a white dwarf) draws in matter from a companion star, producing X-ray radiation. These systems offer a unique window into extreme physics, from the effects of strong gravity and relativistic jets to the presence of intense magnetic fields [@Carroll_Ostlie_2017;@1983bhwd.book.S;@lewin1997xray].

The emitted light travels from the source to the telescope, and during this path trespasses over dense-rarified areas, so-called clumps and the circumstellar environment. The accretion processes are not homogeneous. Some spikes/flares/peaks observed were associated to Raighleight-Taylor inestabilities close to the magnetosphere in slow rotating neutron stars, and these are only a couple of very simple examples. All these phenomena leaves a footprint in the x-rays lirght curves, in the form of peaks and dips (features). The correct analysis of these phenomena, combined with phase-resolve spectroscopy have the potential of providing insight in both the stellar wind structure and the acretion processes.

Upcoming high-resolution missions like XRISM [@2022arXiv220205399X] and New Athena [@2016SPIE.9905E.2FB] promise to significantly improve the quality of these analyses. In addition to better resolution, advances in computational power have been crucial. Many of these tools have already been successfully applied to studies using XMM-Newton and Chandra data, enabling analyses that were previously impossible [@2022MNRAS.512.304S;@2021MNRAS.501.5892S;@newcenx3].


# Statement of Need

Several software packages are available for the analysis of X-ray astronomy. To start with, a well-known and widely used comprehensive package for general astronomy computations and data handling is Astropy [@astropy:2022]. Some notable Python-based options focusing on timing analysis of astronomical data is Stingray [@2019ApJ88139H]. On the other hand, Jaxspec [@2024A&A.690A.317D] specializes in spectral fitting using Bayesian inference. Lightkurve [@lightkurve:2018] simplifies the analysis of time-series data from space missions like Kepler and TESS. 

The dipspeaks package specializes in detecting subtle astrophysical features in X-ray light curves, distinguishing genuine phenomena from random noise. The core idea is straightforward: features in light curves may arise either from astrophysical events or random fluctuations. By accurately modeling how noise typically appears, the software can effectively identify and filter out noise-related features.

To achieve this, dipspeaks generates synthetic "noise-only" light curves that replicate the statistical properties of observed noise. This is done through high-pass Butterworth filtering, random shuffling of residual noise, and careful clipping of extreme samples, thereby removing real signals while preserving the intrinsic noise structure.

An auto-encoder neural network is then trained exclusively on these synthetic noise samples. This neural network learns to reconstruct noise features well but struggles to reconstruct genuine astrophysical signals. Consequently, real events appear as anomalies with high reconstruction errors, quantified by z-scores and percentiles.

To further ensure robustness, the package compares the occurrence rates of detected events in real versus synthetic data. A linear confidence score quantifies how significantly real data features exceed typical noise. 

Apart from detection, some functions are also provided to analyse the dataset, as for example a clustering gaussian-mixture base function that classifies the features in different clusters depending on their duration and prominence, overlap analysis to analyse the features along different energy bands or a clump detection algorithm, that provides a list of those dips that overlap and are more profound within the low energy range than in the high energy range.

# Background Theory
Several phenomena can cause short-term irregularities in X-ray light curves, and their interpretation will depend on serveral factors, from the componenets of the system to their orbital configuration, its imposible to cite them all, examples are the following:

In sources with moderate X-ray luminosity (below $\sim 4 \times 10^{34}$ erg s$^{-1}$), a hot convective quasi-spherical shell forms above the NS magnetosphere. The matter enters the magnetosphere due to Rayleigh-Taylor instability (RTI) at the magnetospheric boundary. This accretion processes in HMXRBs has been known and studied for a long time (see [@1976ApJ.207.914A] and [@1977ApJ.215.897E]) and might cause detectable peaks (spikes) [2025A&A...694A.192S].

In Cen X-3 during the transition period, six clump candidates could be detected within the light curve [2024A&A...690A.360S] with an average mass of approximately $1.3\times 10^{20}$ g and $5.8\times 10^{19}$ g per clump, respectively. Similar values were reported for Vela X-1 [@2010A&A.519A.37F;@2014A&A.563A.70M]. 

The switching of the accretion column between the stellar hemispheres in the magnetosphere of a star with the dipole magnetic field aligned with the stellar rotation axis can produce ‘hiccups’ in the light curves, in the form of quasi periodical oscillations [@10.1093.mnras.stz3088].  

# Acknowledgements

This research has been funded by the ASFAE/2022/02 project from the Generalitat Valenciana.


# References

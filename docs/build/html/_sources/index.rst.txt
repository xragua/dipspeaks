.. dipspeaks documentation master file, created by
   sphinx-quickstart on Wed Apr 23 13:49:01 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to dipspeaks’s documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

  
A brief introduction
--------------------
**X-ray binaries are truly fascinating!** In these extreme environments, a compact object—either a neutron star or black hole—draws in matter from a companion star, producing intense X-ray emissions. These systems offer a unique window into extreme physics, from the effects of strong gravity and relativistic jets to the presence of intense magnetic fields.

The emitted light travels from the source to the telescope, and during this path trespasses over dense-rarified areas, so-called clumps and the circumstellar environment. The accretion processes are not homogeneous. Some spikes/flares/peaks observed in some light curves were associated to Raighleight-Taylor inestabilities. All these phenomena leaves a footprint in the x-rays lirght curves, in the form of peaks and dips (features).

The main inconvenient is distinguish real features, caused by astrophysical phenomena from random noise. With **peaksdips** we use an autoencoder approach to flag those events, giving each detection a probability. We also provide some useful analysis tools such as classify the resulting features by prominence and duration using gaussian mixture or calculate overlaps, if we perform the analysis in different energy ranges.

Getting started
===============

Installation
------------

You can install the package directly from PyPI:

.. code-block:: console

   pip install dipspeaks

.. toctree::
   :maxdepth: 1
   :caption: Rationale:

   api/rationale

.. toctree::
   :maxdepth: 1
   :caption: Example Notebooks:

   examples/cenx3/cenx3.ipynb
   examples/noise/noise.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Functions:
   
   api/function_reference
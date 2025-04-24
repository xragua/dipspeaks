[![Docs](https://img.shields.io/badge/docs-gh--pages-brightgreen)](https://xragua.github.io/dipspeaks/)


# dipspeaks: A Python Package for Detecting and Analyzing relevant dips and peaks in light curves

---

## Installation

You can install the package directly from PyPI using pip: **pip install dipspeaks**.

The dipspeaks package specializes in detecting subtle astrophysical features in X-ray light curves, distinguishing genuine phenomena from random noise. The core idea is straightforward: features in light curves may arise either from astrophysical events or random fluctuations. By accurately modeling how noise typically appears, the software can effectively identify and filter out noise-related features.

To achieve this, dipspeaks generates synthetic "noise-only" light curves that replicate the statistical properties of observed noise. This is done through high-pass Butterworth filtering, random shuffling of residual noise, and careful clipping of extreme samples, thereby removing real signals while preserving the intrinsic noise structure.

An auto-encoder neural network is then trained exclusively on these synthetic noise samples. This neural network learns to reconstruct noise features well but struggles to reconstruct genuine astrophysical signals. Consequently, real events appear as anomalies with high reconstruction errors, quantified by z-scores and percentiles.

To further ensure robustness, the package compares the occurrence rates of detected events in real versus synthetic data. A linear confidence score quantifies how significantly real data features exceed typical noise. 

Apart from detection, some functions are also provided to analyse the dataset, as for example a clustering gaussian-mixture base function that classifies the features in different clusters depending on their duration and prominence, overlap analysis to analyse the features along different energy bands or a clump detection algorithm, that provides a list of those dips that overlap and are more profound within the low energy range than in the high energy range.

---

## Dependencies

The following Python libraries are required:

-- numpy
-- pandas
-- scipy
-- matplotlib
-- astropy
-- pytest>=6.0



---

## Documentation

Detailed documentation, including function references and tutorials, can be found at: [dipspeaks Documentation](https://xragua.github.io/dipspeaks/).

---

## Contributing

Contributions are welcome! Please check the [CONTRIBUTING FILE](https://xragua.github.io/dipspeaks/api/contribute.html) for details on how to contribute.

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/xragua/dipspeaks/blob/main/LICENSE) file for details.

---

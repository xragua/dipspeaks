# __init__.py in dips_peaks

from .dipspeaks_detection.dipspeaks_detection import *

########################################################

from .dipspeaks_detection.helper_functions import (
    rebin_snr,
    _moving_average,
    _base_calculator,
)

from .dipspeaks_detection.detection import (
    _detection,
)

from .dipspeaks_detection.synthetic_data_for_train import (
    _calculate_synthetic_data,
)

from .dipspeaks_detection.autoencoder_model import (
    _clean_autoencoder,
)

from .dipspeaks_detection.evaluation import(
    _modified_z_score,
    _outlier_probability,
    _real_probability,
)

from .dipspeaks_detection.analysis import *

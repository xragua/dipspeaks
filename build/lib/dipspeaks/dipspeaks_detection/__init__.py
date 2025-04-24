# __init__.py in dips_peaks
from ..main import *

from .dipspeaks_detection import *
########################################################

from .helper_functions import (
     rebin_snr,
    _moving_average,
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
    _modified_z_score,
    _outlier_probability,
   _real_probability,
)

from .analysis import *


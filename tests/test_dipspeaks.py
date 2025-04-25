import numpy as np
import pandas as pd
from dipspeaks import *

def make_dummy_lc():
    """Return a minimal light-curve dataframe that is *deterministic*."""
    rng = np.random.default_rng(seed=42)          #   ← fixed seed: reproducible
    n = np.arange(0, 20000, 5)
    return pd.DataFrame({
        "time":   n,
        "counts": rng.integers(0, 100, len(n)),
        "srate":  rng.integers(20, 80, len(n)) * 0.10,
    })

def test_pipeline_runs_without_error(tmp_path):
    """Smoke-test the full pipeline on dummy data."""
    df = make_dummy_lc()
    lc_file = tmp_path / "lc.dat"
    df.to_csv(lc_file, index=False, sep=" ")

    low, high, reb, *_ = detect_dips_and_peaks(
        lc_file, index_time=0, index_rate=1, index_error_rate=2,
        show_plot=False
    )

    # simple behavioural checks – tighten as you refine the library
    assert len(low) >= 0
    assert len(high) >= 0
    assert reb.shape[1] == 3            # rebinned LC has 3 columns

def test_gmm_clustering_returns_labels():
    df = make_dummy_lc()
        low, high, reb, *_ = detect_dips_and_peaks(
        lc_file, index_time=0, index_rate=1, index_error_rate=2,
        show_plot=False
    )

    high = filter_dip_peak(high, high, reb, show_plot=False)
    low  = filter_dip_peak(low,  low,  reb, show_plot=False)
    _, _ = clump_candidates(high, low, reb, overlap_threshold=0.6, show_plot=False)

    stats, labels = gmm_dips_peaks(high, reb, log_scale=True, show_plot=False)
    assert len(labels) == len(high)          # one label per candidate
    assert stats.shape[0] >= 1               # at least one cluster returned


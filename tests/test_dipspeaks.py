import dipspeaks
from dipspeaks import *

n = np.arange(0,20000,5)

df = pd.DataFrame({
    "time": n,  # a simple numeric sequence for the time column
    "counts": np.random.randint(0, 100, len(n)),  # random integers between 0 and 99
    "srate": np.random.randint(20, 80, len(n) )*0.10 # random integers between 0 and 99
})

df.to_csv("test_lc", index=False, sep=' ')

lc="./test_lc"
low_dips_to_clean, high_dips_to_clean, lcreb,_,_ = detect_dips_and_peaks(lc, snr=0.5 ,index_time=0, index_rate=1, index_error_rate=2, show_plot = False)

low_dips = filter_dip_peak(low_dips_to_clean,
                       low_dips_to_clean,
                       lcreb,
                       error_percentile_threshold=0.5,
                       zscore_threshold=3,
                       show_plot=False)

high_dips = filter_dip_peak(high_dips_to_clean,
                       high_dips_to_clean,
                       lcreb,
                       error_percentile_threshold=0.5,
                       zscore_threshold=3,
                       show_plot=False)

hclump, lclump = clump_candidates(high_dips,low_dips,lcreb,overlap_threshold=0.6, show_plot=False)

dip_cluster_stats, dip_labels = gmm_dips_peaks(high_dips,lcreb, log_scale=True, show_plot=False)

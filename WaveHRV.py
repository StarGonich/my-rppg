import ampdlib
import numpy as np


def wavehrv(signal, fs) -> dict:
    signal = np.asarray(signal)
    signal = signal - signal.mean()

    """
    В этом промежутке нехватает следующего:
        1) Интерполяция сигнала (увеличение частоты дискретизации) до ближайшей степени 2.
           Реализовать не сложно, но без пункта 2) бесмысленно.
        2) Scattering Transformation (в статье 2.1 и 2.2)
    """

    peaks = ampdlib.ampd(signal)
    ibi = np.diff(peaks) / fs * 1000.0

    # 1. ∀IBI ∈ [400 ms, 1300 ms]
    ibi = ibi[(ibi >= 400.0) & (ibi <= 1300.0)]
    # 2. ∀IBI ∈ mean(IBI) ± 0.4mean(IBI)
    ibi_mean = ibi.mean()
    ibi = ibi[np.abs(ibi - ibi_mean) <= 0.4 * ibi_mean]
    # 3. Non-overlapping window is slid over IBIs with window size 10. IBIs in each window
    # should satisfy ∀IBIwindow ∈ mean(IBIwindow ) ± 0.2mean(IBIwindow ).
    win = 10
    n = len(ibi)
    keep_mask = np.ones(n, dtype=bool)
    for start in range(0, n - win + 1, win):
        seg = ibi[start:start + win]
        mu_w = seg.mean()
        ok = np.abs(seg - mu_w) <= 0.2 * mu_w
        keep_mask[start:start + win] = ok
    ibi = ibi[keep_mask]

    sdnn = (np.sqrt(np.sum((ibi - ibi.mean()) ** 2) / (len(ibi) - 1)))
    rmssd = (np.sqrt(np.sum(np.diff(ibi) ** 2) / (len(ibi) - 1)))

    return {
        "n_ibi": int(len(ibi)),
        "mean_ibi_ms": float(ibi.mean()),
        "sdnn": float(sdnn),
        "rmssd": float(rmssd)
    }
import numpy as np
import utils

def n_sqi(raw_signal, fps):
    try:
        filtered = utils.bandpass_filter(raw_signal, fps)
    except:
        filtered = raw_signal

    sigma_noise = np.var(filtered)
    sigma_signal = np.var(np.abs(filtered))

    return sigma_signal / sigma_noise


def k_sqi(raw_signal):
    sigma = np.std(raw_signal)
    mu = np.mean(raw_signal)
    return np.mean(((raw_signal - mu) / sigma) ** 4)


# def k_sqi(raw_signal):
#     return scipy.stats.kurtosis(raw_signal, fisher=False, bias=True)
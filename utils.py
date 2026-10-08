import scipy

def bandpass_filter(data, fs, lowcut=0.7, highcut=3.5, order=6):
    sos = scipy.signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return scipy.signal.sosfiltfilt(sos, data)
import ampdlib
import numpy as np
import scipy.signal
import scipy.interpolate
from kymatio.numpy import Scattering1D
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def _bandpass_filter(data, fs, lowcut=0.7, highcut=3.5, order=6):
    sos = scipy.signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return scipy.signal.sosfiltfilt(sos, data)


def _interpolate(signal, fs):
    target = 2 ** int(np.ceil(np.log2(fs)))
    if target == fs:
        return signal, fs
    t = np.arange(len(signal)) / fs
    t_new = np.arange(0, t[-1], 1.0 / target)
    interpolate = scipy.interpolate.interp1d(t, signal, kind="cubic", fill_value="extrapolate")
    return interpolate(t_new), target


def _plot_all(signal_orig, fs_orig, signal_interp, fs_interp,
              signal_filtered, S0, S1, n_show=20):
    """
    Строит 24 графика в одном окне (сетка 4x6):
      1) оригинальный сигнал
      2) сигнал после интерполяции
      3) сигнал после bandpass-фильтра Баттерворта
      4) S0 (нулевой порядок scattering)
      5..24) пути S1 (первый порядок scattering)
    """
    n_total = 4 + min(n_show, S1.shape[0])
    n_cols = 6
    n_rows = int(np.ceil(n_total / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 3.2 * n_rows))
    axes = axes.flatten()

    # --- 1. Оригинальный сигнал ---
    axes[0].plot(signal_orig, linewidth=0.8, color="black")
    axes[0].set_title("1. Original signal", fontsize=10)
    axes[0].set_xlabel("Time (s)", fontsize=8)
    axes[0].set_ylabel("Amplitude", fontsize=8)
    axes[0].tick_params(labelsize=7)
    axes[0].grid(True, alpha=0.3)

    # --- 2. Сигнал после интерполяции ---
    axes[1].plot(signal_interp, linewidth=0.8, color="tab:blue")
    axes[1].set_title(f"2. After interpolation (fs={fs_interp})", fontsize=10)
    axes[1].set_xlabel("Time (s)", fontsize=8)
    axes[1].set_ylabel("Amplitude", fontsize=8)
    axes[1].tick_params(labelsize=7)
    axes[1].grid(True, alpha=0.3)

    # --- 3. Сигнал после bandpass-фильтра ---
    axes[2].plot(signal_filtered, linewidth=0.8, color="tab:orange")
    axes[2].set_title("3. After Butterworth bandpass", fontsize=10)
    axes[2].set_xlabel("Time (s)", fontsize=8)
    axes[2].set_ylabel("Amplitude", fontsize=8)
    axes[2].tick_params(labelsize=7)
    axes[2].grid(True, alpha=0.3)

    # --- 4. S0 (нулевой порядок scattering) ---
    # S0 обычно имеет ту же длину, что и входной сигнал (при T=1)
    axes[3].plot(S0, linewidth=0.8, color="tab:green")
    axes[3].set_title("4. S0 (scattering order 0)", fontsize=10)
    axes[3].set_xlabel("Time (s)", fontsize=8)
    axes[3].set_ylabel("Amplitude", fontsize=8)
    axes[3].tick_params(labelsize=7)
    axes[3].grid(True, alpha=0.3)

    # --- 5..24. Пути S1 ---
    n_paths = min(n_show, S1.shape[0])
    for i in range(n_paths):
        ax = axes[4 + i]
        ax.plot(S1[i], linewidth=0.8)
        ax.set_title(f"{5 + i}. S1 path {i}", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Скрываем пустые подграфики
    for j in range(4 + n_paths, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Signal processing pipeline & Scattering paths "
                 f"(4 + {n_paths} = {4 + n_paths} plots)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("scattering_paths.png", dpi=100)
    plt.close(fig)


def _scattering(signal, fs):
    T_pool = int(fs * 16)  # pooling size = 16 секунд
    J = int(np.log2(T_pool))
    Q = 20  # число вейвлетов в filter bank

    scattering = Scattering1D(J=J, Q=Q, shape=len(signal),
                              T=T_pool, max_order=1)
    S = scattering(signal)
    meta = scattering.meta()
    xi_hz = meta['xi'].flatten() * fs

    ind_start_J = scattering.ind_start[scattering.J]
    t_samples_padded = (ind_start_J + np.arange(S.shape[1]) + 0.5) * T_pool
    t_samples = t_samples_padded - scattering.pad_left

    return S[1:], xi_hz[1:], t_samples


def _amplify_edges(reconstructed, w, s):
    """
    Algorithm 1: peak amplification of the two ends of the signal.
    Усиливает края сигнала, чтобы пики на границах не терялись
    из-за скользящего окна.
    """
    n = len(reconstructed)
    R = reconstructed.copy()
    n_edges = int(w / s)

    for j in range(n_edges):
        c = w / (s * (j + 1))  # убрана 2
        # Левый край: R[s*j : s*(j+1)]
        R[s*j:s*(j+1)] *= c
        # Правый край: R[n - s*(j+1) : n - s*j]
        R[n-s*(j+1):n-s*j] *= c

    return R


def wavehrv(signal, fs) -> dict:
    signal = np.asarray(signal)
    signal = signal - signal.mean()

    signal, fs = _interpolate(signal, fs)

    signal = _bandpass_filter(signal, fs, 0.7, 5, 7)

    S1, xi_hz, time = _scattering(signal, fs)

    mask = (xi_hz >= 0.75) & (xi_hz <= 2.5)
    S1 = S1[mask]
    xi_hz = xi_hz[mask]

    window_size = int(14.5 * fs)  # w в отсчётах
    step = int(2 * fs)  # s в отсчётах

    reconstructed = np.zeros(len(signal))
    counts = np.zeros(len(signal))

    start, end = 0, window_size
    while end <= len(signal):
        i, time_i = None, None
        for el, t_el in enumerate(time):
            if start <= t_el <= end:
                i, time_i = el, t_el
                break
        if i is None and time_i is None:
            # print(start)
            start += step
            end += step
            continue
        i_prev = max(i - 1, 0)

        x = end - time_i
        d1 = (window_size - x) / window_size
        d2 = x / window_size
        E = []
        for e_hz in range(S1.shape[0]):
            E.append(d1*S1[e_hz, i_prev] + d2*S1[e_hz, i])
        if len(E) != len(xi_hz):
            raise ValueError("len(i_candidate) != S1.shape()[1]")

        points = np.column_stack([xi_hz, E])
        kmeans = KMeans(n_clusters=3, init='k-means++',
                        n_init=10, random_state=0)
        kmeans.fit(points)
        centers = kmeans.cluster_centers_
        centers = centers[np.argsort(centers[:, 0])]  # сортируем по частоте
        left_centroid = centers[0, 0]
        right_centroid = centers[-1, 0]

        # --- График points с центроидами для текущего окна ---
        # plt.figure(figsize=(8, 5))
        # plt.scatter(points[:, 0], points[:, 1], c='blue', s=30, label='points')
        # plt.scatter(centers[:, 0], centers[:, 1], c='red', s=30, label='centroids')
        # plt.axvline(left_centroid, color='red', linestyle='--', alpha=0.5)
        # plt.axvline(right_centroid, color='red', linestyle='--', alpha=0.5)
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Energy')
        # plt.title(f'Window: [{start}, {end}], '
        #           f'band = [{left_centroid:.3f}, {right_centroid:.3f}] Hz')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.show()

        seg = signal[start:end]
        seg_filt = _bandpass_filter(seg, fs, left_centroid, right_centroid, 7)
        seg_filt = seg_filt - seg_filt.mean()

        reconstructed[start:end] += seg_filt
        counts[start:end] += 1

        start += step
        end += step

    reconstructed = reconstructed[counts > 0]
    reconstructed = _amplify_edges(reconstructed, window_size, step)

    t = np.arange(len(reconstructed)) / fs
    # plt.figure(figsize=(14, 4))
    # plt.plot(t, reconstructed, linewidth=0.8)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Reconstructed signal (after windowing)')
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()

    signal = reconstructed

    peaks = ampdlib.ampd(signal)
    ibi = np.diff(peaks) / fs * 1000.0

    # 1. ∀IBI ∈ [400 ms, 1300 ms]
    ibi = ibi[(ibi >= 400.0) & (ibi <= 1300.0)]
    # 2. ∀IBI ∈ mean(IBI) ± 0.4mean(IBI)
    ibi_mean = ibi.mean()
    ibi = ibi[np.abs(ibi - ibi_mean) <= 0.4 * ibi_mean]
    # 3. Non-overlapping window is slid over IBIs with window size 10.
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


def wavehrv_simple(signal, fs) -> dict:
    peaks = ampdlib.ampd(signal)
    ibi = np.diff(peaks) / fs * 1000.0

    # 1. ∀IBI ∈ [400 ms, 1300 ms]
    ibi = ibi[(ibi >= 400.0) & (ibi <= 1300.0)]
    # 2. ∀IBI ∈ mean(IBI) ± 0.4mean(IBI)
    ibi_mean = ibi.mean()
    ibi = ibi[np.abs(ibi - ibi_mean) <= 0.4 * ibi_mean]
    # 3. Non-overlapping window is slid over IBIs with window size 10.
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


if __name__ == "__main__":
    signal = np.load('../label_inputs.npy', allow_pickle=False)
    wavehrv(signal, 30)
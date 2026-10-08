import numpy as np
import matplotlib.pyplot as plt

def plot_rppg_fft(signal_data, fps=30):
    """Строит график rPPG-сигнала и его амплитудный спектр после FFT"""
    x = np.asarray(signal_data, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"Ожидался 1D-сигнал длиной >= 2, получено shape={x.shape}")

    n = x.size
    t = np.arange(n) / fps

    # ---- FFT ----
    # Убираем постоянную составляющую, чтобы пик на 0 Гц не забивал график
    x_detrended = x - np.mean(x)

    # Окно Ханна снижает утечку спектра (leakage)
    window = np.hanning(n)
    x_win = x_detrended * window

    fft_vals = np.fft.rfft(x_win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)

    # Нормировка амплитуды с учётом окна
    amplitude = np.abs(fft_vals) * (2.0 / np.sum(window))

    # ---- Пик в физиологическом диапазоне 0.7–3.5 Гц (42–210 bpm) ----
    band = (freqs >= 0.7) & (freqs <= 3.5)
    if np.any(band):
        idx_band = np.where(band)[0]
        idx_peak = idx_band[np.argmax(amplitude[idx_band])]
        f_peak = freqs[idx_peak]
        bpm_peak = f_peak * 60.0
    else:
        f_peak, bpm_peak = None, None

    # ---- Графики ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    # 1) Временной ряд
    axes[0].plot(t, x, color="tab:blue", linewidth=1.2)
    axes[0].set_xlabel("Время, с")
    axes[0].set_ylabel("Амплитуда (усл. ед.)")
    axes[0].set_title(f"rPPG-сигнал: {n} отсчётов, {n / fps:.2f} с @ {fps} Гц")
    axes[0].grid(alpha=0.3)

    # 2) Спектр
    axes[1].plot(freqs, amplitude, color="tab:red", linewidth=1.2)
    axes[1].set_xlabel("Частота, Гц")
    axes[1].set_ylabel("Амплитуда")
    axes[1].set_title("Амплитудный спектр (FFT)")
    axes[1].set_xlim(0, min(5.0, freqs[-1]))   # фокус на физиологическом диапазоне
    axes[1].grid(alpha=0.3)

    # Подсветка полосы 0.7–3.5 Гц
    axes[1].axvspan(0.7, 3.5, color="green", alpha=0.08, label="0.7–3.5 Гц (42–210 bpm)")

    if f_peak is not None:
        axes[1].axvline(f_peak, color="black", linestyle="--", linewidth=1,
                        label=f"Пик: {f_peak:.3f} Гц ≈ {bpm_peak:.1f} bpm")
        axes[1].plot(f_peak, amplitude[idx_peak], "ko", markersize=5)

    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()
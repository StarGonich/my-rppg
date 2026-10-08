import os.path
import numpy as np
import pandas as pd

from vitallens import VitalLens
import time

import SQI as SQI
from WaveHRV import wavehrv

FPS = 30

metrics = {
    "chrom": {"sdnn": [], "rmssd": []},
    "pos":   {"sdnn": [], "rmssd": []},
}
gt_metrics = {"sdnn": [], "rmssd": []}

def iter_ground_truth(root_folder):
    """
    Генератор, который последовательно читает ground truth из всех
    подпапок датасета UBFC-RPPG.

    Для каждой подпапки (испытуемого) возвращает словарь:
        {
            'name':   str,          # имя подпапки
            'path':   str,          # полный путь к папке
            'trace':  np.ndarray,   # PPG-сигнал (нормализованный)
            'time':   np.ndarray,   # временная ось (сек)
            'hr':     np.ndarray,   # ЧСС от сенсора (может быть None)
            'format': str,          # 'DATASET_1' или 'DATASET_2'
        }

    Если в папке нет ground truth или произошла ошибка чтения —
    элемент пропускается. Память не накапливается: каждая итерация
    отдаёт только один массив, который затем может быть собран GC.
    """
    try:
        all_entries = os.listdir(root_folder)
    except FileNotFoundError:
        print(f"Error: Dataset root folder '{root_folder}' not found.")
        return

    dirs = [d for d in all_entries
            if os.path.isdir(os.path.join(root_folder, d))
            and d not in ['.', '..', 'desktop.ini']]
    dirs.sort()

    if not dirs:
        print(f"No subdirectories found in '{root_folder}'.")
        return

    for i, dir_name in enumerate(dirs):
        vid_folder = os.path.join(root_folder, dir_name)

        gt_trace = None
        gt_time = None
        gt_hr = None
        fmt = None

        # --- DATASET_1: gtdump.xmp ---
        gt_filename_1 = os.path.join(vid_folder, 'gtdump.xmp')
        if os.path.exists(gt_filename_1):
            try:
                gt_data = pd.read_csv(gt_filename_1, header=None).values
                gt_trace = gt_data[:, 3].astype(float)
                gt_time = gt_data[:, 0].astype(float) / 1000.0
                gt_hr = gt_data[:, 1].astype(float)
                fmt = 'DATASET_1'
            except Exception as e:
                print(f"[{dir_name}] Error reading gtdump.xmp: {e}")
                continue
        else:
            # --- DATASET_2: ground_truth.txt ---
            gt_filename_2 = os.path.join(vid_folder, 'ground_truth.txt')
            if os.path.exists(gt_filename_2):
                try:
                    gt_data = np.loadtxt(gt_filename_2)
                    gt_trace = gt_data[0, :].astype(float)
                    gt_hr = gt_data[1, :].astype(float)
                    gt_time = gt_data[2, :].astype(float)
                    fmt = 'DATASET_2'
                except Exception as e:
                    print(f"[{dir_name}] Error reading ground_truth.txt: {e}")
                    continue
            else:
                print(f"[{dir_name}] No ground truth file found, skipping.")
                continue

        # Нормализация PPG (z-score)
        std = np.std(gt_trace)
        gt_trace = (gt_trace - np.mean(gt_trace)) / std

        yield {
            'name': dir_name,
            'path': vid_folder,
            'trace': gt_trace,
            'time': gt_time,
            'hr': gt_hr,
            'format': fmt,
        }


for video in iter_ground_truth('../rPPG-Toolbox/data/UBFC-rPPG'):
    print(f"\n=== {video["name"]} ===")
    hrv_results = wavehrv(video["trace"], FPS)
    print(f"GT SDNN:             {round(hrv_results["sdnn"])} ms")
    print(f"GT RMSSD:            {round(hrv_results["rmssd"])} ms")

    print("-" * 40)
    for method in ["chrom", "pos"]:
        start_time = time.time()
        vl = VitalLens(method=method, export_to_json=False)

        # ================================================
        # frames, FPS = read_video_frames(video)
        # video_array = np.array(frames, dtype=np.uint8)
        # results = vl(video=video_array, fps=FPS)
        # ================================================
        results = vl(os.path.join(video["path"], 'vid.avi'))
        # ================================================

        processing_time = time.time() - start_time

        if results and len(results) > 0:
            vitals = results[0]['vitals']
            hr = vitals['heart_rate']['value']
            print(f"{method.upper()} Heart Rate:       {hr:.1f} bpm")
            print(f"{method.upper()} Processing time:  {processing_time:.2f} seconds")

            signal_data = results[0]['waveforms']['ppg_waveform']['data']
            hrv_results = wavehrv(signal_data, FPS)
            print(f"{method.upper()} SDNN:             {round(hrv_results["sdnn"])} ms")
            print(f"{method.upper()} RMSSD:            {round(hrv_results["rmssd"])} ms")
            if signal_data is not None:
                print(f"{method.upper()} Signal length:    {len(signal_data)} samples")
                N_SQI = SQI.n_sqi(signal_data, FPS)
                K_SQI = SQI.k_sqi(signal_data)
                print(f"{method.upper()} N_SQI:            {N_SQI:.4f}")
                print(f"{method.upper()} K_SQI:            {K_SQI:.4f}")
                if N_SQI < 0.293:
                    quality = "Excellent"
                else:
                    quality = "Acceptable/Unfit"
                print(f"{method.upper()} Signal Quality:   {quality}")
                # graphics.plot_rppg_fft(signal_data, fps=30)
            else:
                print(f"{method.upper()} No signal available")
        else:
            print(f"{method.upper()} No results")

        print("-" * 40)
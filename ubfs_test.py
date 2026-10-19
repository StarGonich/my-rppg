import math
import os.path
import numpy as np
import pandas as pd

from vitallens import VitalLens
import time

import SQI as SQI
from WaveHRV import wavehrv, wavehrv_simple

def is_valid(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))

def fmt(x, nd=0):
    return f"{round(x, nd)}" if is_valid(x) else "n/a"

pairs = {
    "chrom": {"sdnn": [], "rmssd": []},
    "pos":   {"sdnn": [], "rmssd": []},
}

def calc_metrics(y_true, y_pred):
    """
    y_true, y_pred: array-like
    Возвращает dict с MAE, SD (std разностей), RMSE.
    Пары с NaN исключаются.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {"n": 0, "MAE": np.nan, "SD": np.nan, "RMSE": np.nan}

    diff = y_pred - y_true
    mae  = np.mean(np.abs(diff))
    sd   = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    rmse = np.sqrt(np.mean(diff ** 2))

    return {"n": len(y_true), "MAE": mae, "SD": sd, "RMSE": rmse}


def iter_ground_truth(root_folder):
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
        fmt_type = None

        gt_filename_1 = os.path.join(vid_folder, 'gtdump.xmp')
        if os.path.exists(gt_filename_1):
            try:
                gt_data = pd.read_csv(gt_filename_1, header=None).values
                gt_trace = gt_data[:, 3].astype(float)
                gt_time = gt_data[:, 0].astype(float) / 1000.0
                gt_hr = gt_data[:, 1].astype(float)
                fmt_type = 'DATASET_1'
            except Exception as e:
                print(f"[{dir_name}] Error reading gtdump.xmp: {e}")
                continue
        else:
            gt_filename_2 = os.path.join(vid_folder, 'ground_truth.txt')
            if os.path.exists(gt_filename_2):
                try:
                    gt_data = np.loadtxt(gt_filename_2)
                    gt_trace = gt_data[0, :].astype(float)
                    gt_hr = gt_data[1, :].astype(float)
                    gt_time = gt_data[2, :].astype(float)
                    fmt_type = 'DATASET_2'
                except Exception as e:
                    print(f"[{dir_name}] Error reading ground_truth.txt: {e}")
                    continue
            else:
                print(f"[{dir_name}] No ground truth file found, skipping.")
                continue

        std = np.std(gt_trace)
        gt_trace = (gt_trace - np.mean(gt_trace)) / std

        yield {
            'name': dir_name,
            'path': vid_folder,
            'trace': gt_trace,
            'time': gt_time,
            'hr': gt_hr,
            'format': fmt_type,
        }


def iter_ubfc_phys(root_folder):
    try:
        entries = os.listdir(root_folder)
    except FileNotFoundError:
        print(f"Error: UBFC-Phys root folder '{root_folder}' not found.")
        return

    subject_dirs = []
    for d in entries:
        full = os.path.join(root_folder, d)
        if os.path.isdir(full) and d.startswith('s') and d[1:].isdigit():
            subject_dirs.append(d)
    subject_dirs.sort(key=lambda x: int(x[1:]))

    if not subject_dirs:
        print(f"No subject folders (s1, s2, ...) found in '{root_folder}'.")
        return

    tasks = ['T1', 'T2', 'T3']

    for subj in subject_dirs:
        subj_path = os.path.join(root_folder, subj)
        for task in tasks:
            bvp_file = os.path.join(subj_path, f'bvp_{subj}_{task}.csv')
            vid_file = os.path.join(subj_path, f'vid_{subj}_{task}.avi')

            if not os.path.exists(bvp_file):
                print(f"[{subj}_{task}] BVP file not found: {bvp_file}, skipping.")
                continue

            try:
                bvp = pd.read_csv(bvp_file, header=None).values.squeeze().astype(float)
            except Exception as e:
                print(f"[{subj}_{task}] Error reading BVP: {e}")
                continue

            std = np.std(bvp)
            if std == 0:
                print(f"[{subj}_{task}] Zero std in BVP, skipping.")
                continue
            trace = (bvp - np.mean(bvp)) / std

            n = len(trace)
            time_axis = np.arange(n) / 64

            yield {
                'subject': subj,
                'task': task,
                'name': f'{subj}_{task}',
                'path': subj_path,
                'video': vid_file if os.path.exists(vid_file) else None,
                'trace': trace,
                'time': time_axis,
                'fs': 64,
            }

FPS = 30
for video in iter_ground_truth('../rPPG-Toolbox/data/UBFC-rPPG'):
# FPS = 30
# for video in iter_ground_truth('../UBFC-rPPG-1'):
    print(f"\n=== {video['name']} ===")
    hrv_results = wavehrv_simple(video["trace"], FPS)

    gt_sdnn  = hrv_results["sdnn"]
    gt_rmssd = hrv_results["rmssd"]

    if is_valid(gt_sdnn):
        print(f"GT SDNN:             {fmt(gt_sdnn)} ms")
    else:
        print(f"GT SDNN:             n/a (n_ibi={hrv_results['n_ibi']})")
    if is_valid(gt_rmssd):
        print(f"GT RMSSD:            {fmt(gt_rmssd)} ms")
    else:
        print(f"GT RMSSD:            n/a (n_ibi={hrv_results['n_ibi']})")

    print("-" * 40)
    for method in ["chrom", "pos"]:
        start_time = time.time()
        vl = VitalLens(method=method, export_to_json=False)

        results = vl(os.path.join(video["path"], 'vid.avi'))

        processing_time = time.time() - start_time

        if results and len(results) > 0:
            vitals = results[0]['vitals']
            hr = vitals['heart_rate']['value']
            print(f"{method.upper()} Heart Rate:       {hr:.1f} bpm")
            print(f"{method.upper()} Processing time:  {processing_time:.2f} seconds")

            signal_data = results[0]['waveforms']['ppg_waveform']['data']
            hrv_results = wavehrv_simple(signal_data, FPS)

            pred_sdnn  = hrv_results["sdnn"]
            pred_rmssd = hrv_results["rmssd"]

            if is_valid(pred_sdnn):
                print(f"{method.upper()} SDNN:             {fmt(pred_sdnn)} ms")
            else:
                print(f"{method.upper()} SDNN:             n/a (n_ibi={hrv_results['n_ibi']})")
            if is_valid(pred_rmssd):
                print(f"{method.upper()} RMSSD:            {fmt(pred_rmssd)} ms")
            else:
                print(f"{method.upper()} RMSSD:            n/a (n_ibi={hrv_results['n_ibi']})")

            if is_valid(gt_sdnn) and is_valid(pred_sdnn):
                pairs[method]["sdnn"].append((gt_sdnn, pred_sdnn))
            if is_valid(gt_rmssd) and is_valid(pred_rmssd):
                pairs[method]["rmssd"].append((gt_rmssd, pred_rmssd))

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
            else:
                print(f"{method.upper()} No signal available")
        else:
            print(f"{method.upper()} No results")

        print("-" * 40)

print("\n" + "=" * 55)
print("ИТОГОВЫЕ МЕТРИКИ (ms)")
print("=" * 55)

for method in pairs:
    for metric_name in ["sdnn", "rmssd"]:
        data = pairs[method][metric_name]
        if not data:
            print(f"{method.upper():>5} | {metric_name.upper():>5} | n=0")
            continue
        y_true, y_pred = zip(*data)
        res = calc_metrics(y_true, y_pred)
        print(f"{method.upper():>5} | {metric_name.upper():>5} | "
              f"n={res['n']:>3} | "
              f"MAE={res['MAE']:>7.2f} | "
              f"SD={res['SD']:>7.2f} | "
              f"RMSE={res['RMSE']:>7.2f}")
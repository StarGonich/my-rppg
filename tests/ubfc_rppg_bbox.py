import math
import numpy as np

from vitallens import VitalLens
import time

import SQI as SQI
from WaveHRV import wavehrv, wavehrv_simple
from tests import iters


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


FPS = 30
for video in iters.iter_ubfc_rppg('../../rPPG-Toolbox/data/UBFC-rPPG'):
    print(f"\n=== {video['name']} ===")
    hrv_results = wavehrv_simple(video["trace"], FPS)
    # if hrv_results['n_ibi'] == 0:
    #     hrv_results = wavehrv_simple(video["trace"], FPS)

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

        results = vl(video["video"])

        processing_time = time.time() - start_time

        if results and len(results) > 0:
            vitals = results[0]['vitals']
            hr = vitals['heart_rate']['value']
            print(f"{method.upper()} Heart Rate:       {hr:.1f} bpm")
            print(f"{method.upper()} Processing time:  {processing_time:.2f} seconds")

            signal_data = results[0]['waveforms']['ppg_waveform']['data']
            signal_data = np.asarray(signal_data)
            hrv_results = wavehrv(signal_data, FPS)
            if hrv_results['n_ibi'] == 0:
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
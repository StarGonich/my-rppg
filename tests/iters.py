import os.path
import numpy as np
import pandas as pd


def iter_ubfc_rppg(root_folder):
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
        # if dir_name != 'subject37':
        #     continue
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
            'video': os.path.join(vid_folder, 'vid.avi'),
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
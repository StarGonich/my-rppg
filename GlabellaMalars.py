import math
import time

import cv2
import numpy as np
import scipy
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

GLABELLA_INDICES = [151, 108, 107, 55, 8, 285, 336, 337]
LEFT_MALAR_INDICES = [126, 100, 118, 117, 116, 123, 147, 187, 205, 203, 129, 209]
RIGHT_MALAR_INDICES = [355, 429, 358, 423, 425, 411, 376, 352, 345, 346, 347, 329]

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    # running_mode=vision.RunningMode.VIDEO,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.75,
    min_face_presence_confidence=0.75,
    min_tracking_confidence=0.75,
)
DETECTOR = vision.FaceLandmarker.create_from_options(options)

def _detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = scipy.sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


def POS(RGB_mean, fs):
    """
    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
    Algorithmic principles of remote PPG.
    IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
    """
    WinSec = 1.6
    N = RGB_mean.shape[0]
    H = np.zeros(N)
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            win = RGB_mean[m:n, :]
            Cn = win / np.mean(win, axis=0)
            S = np.array([[0, 1, -1],
                          [-2, 1, 1]]) @ Cn.T
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            h = h - np.mean(h)
            H[m:n] += h

    BVP = H
    BVP = _detrend(BVP.reshape(-1, 1), 100)
    BVP = np.asarray(BVP).ravel()

    b, a = scipy.signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = scipy.signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP


def CHROM(RGB_mean, fs):
    """
    The Chrominance Method from: De Haan, G., & Jeanne, V. (2013).
    Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
    DOI: 10.1109/TBME.2013.2266196
    """
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6

    FN = RGB_mean.shape[0]
    NyquistF = 1 / 2 * fs
    B, A = scipy.signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')

    WinL = math.ceil(WinSec * fs)
    if (WinL % 2):
        WinL = WinL + 1
    NWin = math.floor((FN - WinL // 2) / (WinL // 2))
    WinS = 0
    WinM = int(WinS + WinL // 2)
    WinE = WinS + WinL
    totallen = (WinL // 2) * (NWin + 1)
    S = np.zeros(totallen)

    for i in range(NWin):
        RGBBase = np.mean(RGB_mean[WinS:WinE, :], axis=0)
        RGBNorm = np.zeros((WinE - WinS, 3))
        for temp in range(WinS, WinE):
            RGBNorm[temp - WinS] = np.true_divide(RGB_mean[temp], RGBBase)
        Xs = np.squeeze(3 * RGBNorm[:, 0] - 2 * RGBNorm[:, 1])
        Ys = np.squeeze(1.5 * RGBNorm[:, 0] + RGBNorm[:, 1] - 1.5 * RGBNorm[:, 2])
        Xf = scipy.signal.filtfilt(B, A, Xs, axis=0)
        Yf = scipy.signal.filtfilt(B, A, Ys)

        Alpha = np.std(Xf) / np.std(Yf)
        SWin = Xf - Alpha * Yf
        SWin = np.multiply(SWin, scipy.signal.windows.hann(WinL))

        temp = SWin[:int(WinL // 2)]
        S[WinS:WinM] = S[WinS:WinM] + SWin[:int(WinL // 2)]
        S[WinM:WinE] = SWin[int(WinL // 2):]
        WinS = WinM
        WinM = WinS + WinL // 2
        WinE = WinS + WinL
    BVP = S
    return BVP


def _get_points(face_landmarks, indices, w, h):
    points = []
    for idx in indices:
        lm = face_landmarks[idx]
        points.append((int(lm.x * w), int(lm.y * h)))
    return np.array(points, dtype=np.int32)


def _pixels_in_polygon(image, polygon):
    x, y, w, h = cv2.boundingRect(polygon)
    roi = image[y:y+h, x:x+w]
    poly_local = polygon - np.array([x, y], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_local], 255)
    return roi[mask == 255]


def getRGB_mean(video):
    cap = cv2.VideoCapture(video)
    RGB_mean = []
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        h, w, _ = image.shape

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # plt.imshow(rgb)
        # plt.show()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # start_time = time.time()
        result = DETECTOR.detect(mp_image)
        # result = DETECTOR.detect_for_video(mp_image, timestamp_ms)
        # print(time.time() - start_time)
        if result.face_landmarks:
            lm = result.face_landmarks[0]

            glabella = _get_points(lm, GLABELLA_INDICES, w, h)
            left_cheek = _get_points(lm, LEFT_MALAR_INDICES, w, h)
            right_cheek = _get_points(lm, RIGHT_MALAR_INDICES, w, h)

            glabella_pixels = _pixels_in_polygon(rgb, glabella)
            left_cheek_pixels = _pixels_in_polygon(rgb, left_cheek)
            right_cheek_pixels = _pixels_in_polygon(rgb, right_cheek)

            ROI_pixels = np.vstack([glabella_pixels, left_cheek_pixels, right_cheek_pixels])
            total = np.sum(ROI_pixels, axis=0)
            ROI_mean = total / ROI_pixels.shape[0]
            RGB_mean.append(ROI_mean)
    return np.asarray(RGB_mean)


def get_bvp(RGB_mean, fps, method: str) -> np.ndarray:
    match method:
        case "chrom":
            bvp = CHROM(RGB_mean, fps)
        case "pos":
            bvp = POS(RGB_mean, fps)
        case _:
            raise ValueError(f"Неизвестный method: {method}")
    return bvp


def get_hr(BVP: np.ndarray, FPS) -> float:
    freqs = scipy.fft.rfftfreq(len(BVP), d=1 / FPS)
    spec = np.abs(scipy.fft.rfft(BVP - BVP.mean()))
    m = (freqs >= 0.7) & (freqs <= 3.5)
    hr = freqs[m][np.argmax(spec[m])] * 60
    return hr


def get_fps(video):
    return cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS)
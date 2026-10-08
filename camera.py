import cv2
import numpy as np
import time
import os
from datetime import datetime
from vitallens import VitalLens
import logging
from SQI import n_sqi
from WaveHRV import wavehrv

logging.disable(logging.WARNING)  # отключение предупреждения: 'WARNING:root:No faces found'

vl = VitalLens(method="pos", export_to_json=False)
face_detector = vl.face_detector

# Настройки
window_name = "VitalLens - Camera"
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("http://10.133.232.17:4747/video")  # для записи со смартфона
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Переменные состояния
is_recording = False
recording_start_time = None
recording_duration = 30  # секунд
frames = []
face_boxes = []
current_bbox = None
FPS = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS камеры = {FPS}")
FPS = 30
TARGET_FRAMES = recording_duration * FPS
face_lost = False


def detect_face(frame):
    """Детекция лица с использованием встроенного детектора VitalLens"""
    if face_detector is None:
        return None
    try:
        # Подготовка кадра для детектора
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.expand_dims(frame_rgb, axis=0)

        # Запуск детекции
        faces, _ = face_detector(
            inputs=frame_rgb,
            n_frames=1,
            fps=FPS
        )

        if len(faces) > 0:
            h, w = frame.shape[:2]
            # Берем первое лицо (уверенное)
            face = faces[0][0]
            x0, y0, x1, y1 = (face * [w, h, w, h]).astype(np.int64)
            return int(x0), int(y0), int(x1), int(y1)
    except Exception as e:
        print(f"Ошибка детекции: {e}")
    return None


def draw_bbox(frame, bbox, color=(0, 255, 0)):
    """Функция рисования рамки"""
    if bbox is None:
        return frame

    x0, y0, x1, y1 = bbox
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    return frame


def process_video(frames, face_boxes, fps):
    """Обработка видео через VitalLens"""
    if not frames:
        return None

    print(f"\nОбработка видео ({len(frames)} кадров)...")

    # Проверяем, есть ли детекции лиц на всех кадрах
    all_detected = all(b is not None for b in face_boxes)

    if not all_detected:
        print("ОШИБКА: Лицо было потеряно во время записи!")
        print("Пожалуйста, повторите замер, держа лицо в кадре.")
        return None

    # Подготовка массива с координатами лиц
    n_frames = len(frames)
    faces_array = np.zeros((n_frames, 4), dtype=np.int64)

    for i, bbox in enumerate(face_boxes):
        if bbox is not None:
            faces_array[i] = bbox

    # Конвертация в numpy
    video_array = np.array(frames, dtype=np.uint8)

    try:
        results = vl(video=video_array, fps=fps)

        if results and len(results) > 0:
            vitals = results[0]['vitals']
            rppg_signal = results[0]['waveforms']['ppg_waveform']['data']
            N_SQI = n_sqi(rppg_signal, fps)

            # ========== ВЫВОД В КОНСОЛЬ ==========
            print("\n" + "=" * 50)
            print("РЕЗУЛЬТАТЫ ИЗМЕРЕНИЯ:")
            for name, data in vitals.items():
                if data.get('value') is not None:
                    print(f"{name.upper()}: {data['value']:.1f} {data.get('unit', '')}")
            hrv_results = wavehrv(rppg_signal, fps)
            print(f"SDNN: {hrv_results["sdnn"]}")
            print(f"RMSSD: {hrv_results["rmssd"]}")

            # ========== ВЫВОД N_SQI ==========
            print("-" * 50)
            print("КАЧЕСТВО СИГНАЛА (SQI):")
            if N_SQI is not None:
                print(f"N_SQI: {N_SQI:.4f}")
                print(f"Оценка: {"Excellent" if N_SQI < 0.293 else "Acceptable/Unfit"}")
            else:
                print(f"N_SQI: не удалось вычислить")
            print("=" * 50)
            return results
        else:
            print("Не удалось получить результаты анализа")
            return None

    except Exception as e:
        print(f"Ошибка обработки: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_video(frames, fps):
    """Сохранение видео рядом с программой в .avi файл, без сжатия"""
    if not frames:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vitallens_recording_{timestamp}.avi"
    video_path = os.path.join(os.getcwd(), filename)

    try:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        # fourcc = cv2.VideoWriter_fourcc(*'HFYU')
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        for frame in frames:
            out.write(frame)

        out.release()
        print(f"Видео сохранено: {video_path} | {os.path.getsize(video_path) / (1024 * 1024):.1f} MB")
        return video_path
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")
        return None


# Основной цикл
print("\nУПРАВЛЕНИЕ:")
print("  SPACE - начать/остановить запись")
print("  ESC   - выход")
print("\nВАЖНО: Держите лицо в кадре на протяжении всего замера!")
print("   Если лицо пропадет - запись прервется.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Детекция лица
    current_bbox = detect_face(frame)

    # Запись видео
    if is_recording:
        # Проверяем потерю лица ДО сохранения кадра
        if current_bbox is None:
            print("\nЛИЦО ПОТЕРЯНО! Запись ПРЕРВАНА.")
            is_recording = False
            frames = []
            face_boxes = []
            continue  # Переходим к следующему кадру

        # Сохраняем кадр только если лицо есть
        frames.append(frame.copy())
        face_boxes.append(current_bbox)

        elapsed = time.time() - recording_start_time
        print(len(frames))

        # Проверка окончания записи
        if elapsed >= recording_duration:
            is_recording = False
            print(f"Запись завершена успешно! Записано {len(frames)} кадров")
            # defacto_fps = len(frames) / recording_duration
            if len(frames) >= 299:  # минимум 10 секунд
                save_video(frames, FPS)
                process_video(frames, face_boxes, FPS)
            else:
                print(f"Недостаточно кадров: {len(frames)} (нужно минимум 299)")
            frames = []
            face_boxes = []

    # Выбор цвета рамки
    if is_recording:
        color = (0, 255, 255)  # Желтая
    else:
        color = (0, 255, 0)  # Зеленая
    draw_bbox(frame, current_bbox, color)

    # Таймер во время записи
    if is_recording:
        elapsed = time.time() - recording_start_time
        remaining = max(0, recording_duration - elapsed)
        cv2.putText(frame, f"{int(remaining)}s", (frame.shape[1] - 80, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, "RECORDING", (frame.shape[1] // 2 - 60, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow(window_name, frame)

    # Обработка клавиш
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        if not is_recording:
            if current_bbox is not None:
                is_recording = True
                recording_start_time = time.time()
                frames = []
                face_boxes = []
                print(f"\nНачата запись на {recording_duration} секунд...")
                print("Держите лицо в кадре!")
            else:
                print("Лицо не обнаружено!")
        else:
            # Ручная остановка
            is_recording = False
            print("\nЗапись остановлена пользователем")
            if len(frames) >= 299:
                print(f"Записано {len(frames)} кадров")
                save_video(frames, FPS)
                process_video(frames, face_boxes, FPS)
            else:
                print(f"Недостаточно кадров: {len(frames)}")
            frames = []
            face_boxes = []

cap.release()
cv2.destroyAllWindows()
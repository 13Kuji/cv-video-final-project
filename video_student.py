"""Чтение видео и анализ движения. Здесь нет обучения или разметки игроков."""

import cv2
import numpy as np


def read_frames(path, stride=2, width=640, max_frames=None):
    """Последовательно читать видео, не загружая его целиком в память."""
    if stride is None or int(stride) != stride or stride <= 0:
        raise ValueError("stride должен быть положительным целым")
    if width is None or int(width) != width or width <= 0:
        raise ValueError("width должен быть положительным числом пикселей")
    stride = int(stride)
    width = int(width)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Видео недоступно: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_h = max(1, int(round(source_h * width / max(1, source_w))))
        target, emitted, frame_no = 0, 0, 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if frame_no == target:
                resized = cv2.resize(img, (width, new_h))
                yield dict(frame=frame_no, time_s=frame_no / fps, image=resized)
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
                target += stride
            frame_no += 1
    finally:
        cap.release()


def motion_mask(previous, current, threshold=25, min_area=25):
    """Выделить изменившиеся пиксели между двумя кадрами."""
    if previous.shape != current.shape or previous.ndim != 3:
        raise ValueError("Кадры должны быть одинаковой BGR-формы")
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    if min_area > 1:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = (stats[1:, cv2.CC_STAT_AREA] >= min_area).astype(np.uint8)
        indices = np.arange(1, num, dtype=np.int32)[keep == 1]
        clean = np.zeros_like(mask)
        if indices.size:
            clean = np.isin(labels, indices).astype(np.uint8) * 255
        mask = clean
    return mask.astype(np.uint8)


def dense_flow(previous, current):
    """Рассчитать плотный оптический поток между двумя кадрами."""
    if previous.shape != current.shape or previous.ndim != 3:
        raise ValueError("Кадры должны быть одинаковой BGR-формы")
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def summarize_motion(mask, flow, delta_s):
    """Перевести смещения в скорость и кратко описать движение кадра."""
    if (
        mask.ndim != 2
        or flow.ndim != 3
        or flow.shape[2] != 2
        or flow.shape[:2] != mask.shape
    ):
        raise ValueError("Неверная форма маски или потока")
    if not np.isfinite(flow).all():
        raise ValueError("Поток содержит NaN/Inf")
    if not hasattr(delta_s, "__float__") or float(delta_s) <= 0:
        raise ValueError("delta_s должен быть положительным")
    if int((mask != 0).sum()) == 0:
        return dict(moving_fraction=0.0, median_speed_px_s=0.0)
    moving = mask != 0
    fraction = float(moving.mean())
    speed = np.linalg.norm(flow, axis=-1)[moving] / float(delta_s)
    return dict(moving_fraction=fraction, median_speed_px_s=float(np.median(speed)))
"""Часть 2: сравните признаки внешности и подключите DeepSORT.

Реализуйте три функции. Загрузка данных, метрики и видео уже готовы.
После базового запуска исследуйте ошибки и проверьте своё улучшение.
"""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from deep_sort_realtime.deepsort_tracker import DeepSort


def _validate_vectors(queries, gallery):
    q = np.asarray(queries, float)
    g = np.asarray(gallery, float)
    if q.ndim != 2 or g.ndim != 2:
        raise ValueError("Запросы и образцы должны быть матрицами [N, D]")
    if q.shape[0] and g.shape[1] != q.shape[1] and q.shape[1] != 0:
        raise ValueError("Число признаков запросов и образцов не совпадает")
    return q, g


def cosine_distances(queries, gallery):
    """Вычислить косинусные расстояния между запросами и образцами."""
    q, g = _validate_vectors(queries, gallery)
    if q.ndim != 2 or g.ndim != 2:
        raise ValueError("Требуются матрицы [N, D]")
    if q.shape[1] != g.shape[1]:
        raise ValueError("Размерности признаков не совпадают")
    for name, mat in (("запросы", q), ("образцы", g)):
        if not np.isfinite(mat).all():
            raise ValueError(f"{name} содержат NaN/Inf")
        if mat.shape[0]:
            norms = np.linalg.norm(mat, axis=1)
            if (norms <= 1e-12).any():
                raise ValueError(f"Нулевой вектор в {name}")
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    gn = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    cos = np.einsum("ij,kj->ik", qn, gn)
    cos = np.clip(cos, -1.0, 1.0)
    return 1.0 - cos


def identify_people(queries, gallery, gallery_ids, threshold=0.3):
    """Найти ближайший известный образец или ответить «неизвестно»."""
    gallery_ids = np.asarray(gallery_ids, int)
    if gallery.shape[0] == 0:
        return np.full(queries.shape[0], -1, int)
    dist = cosine_distances(queries, gallery)
    best = np.argmin(dist, axis=1)
    best_dist = dist[np.arange(dist.shape[0]), best]
    ids = gallery_ids[best]
    ids[best_dist > threshold] = -1
    return ids.astype(int)


def track_with_deepsort(data, fps, max_age_s=1.0, cosine_threshold=0.3):
    """Сопроводить людей по готовым детекциям и признакам внешности."""
    frames = np.asarray(data["frames"], int)
    step = int(frames[1] - frames[0]) if len(frames) > 1 else 2
    max_age = int(math.ceil(max_age_s * fps / step))
    tracker = DeepSort(
        max_iou_distance=0.7,
        max_age=max_age,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=float(cosine_threshold),
        nn_budget=50,
        embedder=None,
    )
    rows = []
    for f in frames:
        idx = np.flatnonzero(data["det_frame"] == f)
        raw, embeds, indexes = [], [], []
        for i in idx:
            x1, y1, x2, y2 = (float(v) for v in data["bbox"][i])
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            raw.append(([x1, y1, w, h], float(data["confidence"][i]), "person"))
            embeds.append(data["embeddings"][int(i)])
            indexes.append(int(i))
        if len(raw):
            current = tracker.update_tracks(raw, embeds=embeds, others=indexes)
        else:
            current = tracker.update_tracks([], embeds=[])
        tracks = []
        for t in current:
            observed = int(t.time_since_update) == 0
            supplementary = t.get_det_supplementary()
            detection_index = supplementary if observed else None
            if observed:
                bb = t.to_ltrb(orig=True)
                if bb is None:
                    bb = t.to_ltrb()
            else:
                bb = t.to_ltrb()
            if bb is not None:
                bb = tuple(float(v) for v in bb)
                if bb[2] <= bb[0] or bb[3] <= bb[1]:
                    bb = None
            tracks.append(
                dict(
                    track_id=int(t.track_id),
                    bbox=bb,
                    confirmed=bool(t.is_confirmed()),
                    observed=observed,
                    missed_steps=int(t.time_since_update),
                    detection_index=detection_index,
                )
            )
        rows.append(dict(frame=int(f), time_s=float(f) / fps, tracks=tracks))
    return rows


def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def track_with_iou(data, fps, max_age_s=0.5, iou_threshold=0.3):
    """Реализовать простой трекер без признаков внешности и фильтра Калмана."""
    frames = np.asarray(data["frames"], int)
    step = int(frames[1] - frames[0]) if len(frames) > 1 else 2
    life = int(math.ceil(max_age_s * fps / step))
    active = {}
    next_id = 1
    rows = []
    for si, f in enumerate(frames):
        idx = np.flatnonzero(data["det_frame"] == f)
        current_boxes = [tuple(float(v) for v in data["bbox"][i]) for i in idx]
        track_keys = [k for k in active if si - active[k]["last_i"] < life]
        boxes_a = [active[k]["bbox"] for k in track_keys]
        cost = np.full((len(boxes_a), len(current_boxes)), 1e6, float)
        if boxes_a and current_boxes:
            for r in range(len(boxes_a)):
                for c in range(len(current_boxes)):
                    value = _iou(boxes_a[r], current_boxes[c])
                    if value >= iou_threshold:
                        cost[r, c] = -value
        r_matched = set()
        c_matched = set()
        if cost.size:
            rows_a, cols_a = linear_sum_assignment(cost)
            for r, c in zip(rows_a, cols_a):
                if cost[r, c] > 1e3:
                    continue
                r_matched.add(r)
                c_matched.add(c)
        emitted = []
        assigned_det = {}
        for r, key in enumerate(track_keys):
            if r in r_matched:
                c = int(cols_a[list(rows_a).index(r)])
                assigned_det[key] = c
        for key in track_keys:
            if key in assigned_det:
                c = assigned_det[key]
                active[key]["bbox"] = current_boxes[c]
                active[key]["last_i"] = si
                emitted.append(
                    dict(
                        track_id=key,
                        bbox=list(current_boxes[c]),
                        confirmed=True,
                        observed=True,
                        missed_steps=0,
                        detection_index=int(idx[c]),
                    )
                )
            else:
                emitted.append(
                    dict(
                        track_id=key,
                        bbox=list(active[key]["bbox"]),
                        confirmed=True,
                        observed=False,
                        missed_steps=si - active[key]["last_i"],
                        detection_index=None,
                    )
                )
        for ci in range(len(current_boxes)):
            if ci in c_matched:
                continue
            while next_id in active:
                next_id += 1
            active[next_id] = dict(bbox=current_boxes[ci], last_i=si)
            emitted.append(
                dict(
                    track_id=next_id,
                    bbox=list(current_boxes[ci]),
                    confirmed=True,
                    observed=True,
                    missed_steps=0,
                    detection_index=int(idx[ci]),
                )
            )
            next_id += 1
        for key in [k for k in active if si - active[k]["last_i"] >= life]:
            del active[key]
        rows.append(dict(frame=int(f), time_s=float(f) / fps, tracks=emitted))
    return rows


def player_statistics(rows, fps, image_size, grid_size=(12, 8)):
    """Рассчитать движение и карту посещений отдельно для каждого ID."""
    grid_cols, grid_rows = grid_size
    width, height = image_size
    result = {}
    last_seen = {}
    for i, row in enumerate(rows):
        for t in row["tracks"]:
            if not (t["confirmed"] and t["observed"] and t["bbox"] is not None):
                continue
            key = str(t["track_id"])
            bb = t["bbox"]
            x = (bb[0] + bb[2]) / 2.0
            y = float(bb[3]) if bb[3] > bb[1] else float(bb[1])
            time_s = float(row["time_s"])
            entry = result.setdefault(
                key,
                dict(
                    observed_frames=0,
                    observed_time_s=0.0,
                    distance_px=0.0,
                    mean_speed_px_s=0.0,
                    max_speed_px_s=0.0,
                ),
            )
            pairs = []
            if key in last_seen:
                prev_row_i, prev_x, prev_y, prev_t = last_seen[key]
                if prev_row_i == i - 1:
                    dist = math.hypot(x - prev_x, y - prev_y)
                    dt = max(time_s - prev_t, 1e-9)
                    entry["distance_px"] += dist
                    entry["_total_dt"] = entry.get("_total_dt", 0.0) + dt
                    pairs.append((dist, dt))
            last_seen[key] = (i, x, y, time_s)
            entry["observed_frames"] += 1
            entry["observed_time_s"] = entry["observed_frames"] / fps
            if pairs:
                entry["mean_speed_px_s"] = entry["distance_px"] / entry["_total_dt"]
                entry["max_speed_px_s"] = max(
                    entry["max_speed_px_s"], max(d / dt for d, dt in pairs)
                )
    for key, entry in result.items():
        if entry["observed_frames"] == 0:
            entry["mean_speed_px_s"] = 0.0
        hm = np.zeros((grid_rows, grid_cols), int)
        for i, row in enumerate(rows):
            for t in row["tracks"]:
                if not (t["confirmed"] and t["observed"] and t["bbox"] is not None):
                    continue
                if str(t["track_id"]) != key:
                    continue
                bb = t["bbox"]
                x = (bb[0] + bb[2]) / 2.0
                y = float(bb[3]) if bb[3] > bb[1] else float(bb[1])
                col = min(grid_cols - 1, max(0, int(x / width * grid_cols)))
                rrow = min(grid_rows - 1, max(0, int(y / height * grid_rows)))
                hm[rrow, col] += 1
        entry["heatmap"] = hm.astype(int).tolist()
    return result
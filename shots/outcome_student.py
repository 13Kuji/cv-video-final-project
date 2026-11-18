"""Часть 1: реализуйте функции для мяча, бросков и обучения модели.

Начните с track_ball и find_candidates, затем перейдите к обучению.
Загрузка данных, сохранение весов, оценка и видео уже готовы.
"""

import random

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class FrameBaseline(nn.Module):
    """Готовый пример: модель исхода по последнему кадру окна."""

    def __init__(self):
        """Создать небольшую сеть: 70 признаков → 32 числа → один logit."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(70, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features):
        """Преобразовать тензор [пачка, время, 70] в logits [пачка]."""
        last_frame = features[:, -1]
        return self.net(last_frame).squeeze(-1)


class _BallKalman:
    """Постоянноскоростной фильтр Калмана для центра мяча в плоскости."""

    def __init__(self, x, y):
        self.x = np.array([float(x), float(y), 0.0, 0.0], float)
        self.P = np.eye(4) * 1e2
        self.P[2:, 2:] = 0.0
        self.set = True
        self.Q = np.diag([20.0, 20.0, 2500.0, 2500.0])
        self.R = np.diag([9.0, 9.0])

    def predict(self, dt):
        F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt
        return self.x.copy()

    def update(self, z):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = np.asarray(z, float) - H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return self.x.copy()

    def innov_dist(self, z):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        S = H @ self.P @ H.T + self.R
        y = np.asarray(z, float) - H @ self.x
        try:
            return float(np.sqrt(y @ np.linalg.inv(S) @ y))
        except np.linalg.LinAlgError:
            return float("inf")


class TemporalOutcomeModel(nn.Module):
    """Модель исхода броска: последний кадр, среднее или GRU."""

    def __init__(self, kind="gru"):
        """Создать слои выбранной модели."""
        super().__init__()
        if kind not in ("last", "mean", "gru"):
            raise ValueError("kind должен быть last, mean или gru")
        self.kind = kind
        if kind == "last":
            self.backbone = FrameBaseline()
        elif kind == "mean":
            self.net = nn.Sequential(
                nn.Linear(70, 32), nn.ReLU(), nn.Linear(32, 1)
            )
        else:
            self.gru = nn.GRU(
                70, 32, batch_first=True, bidirectional=False
            )
            self.out = nn.Linear(32, 1)

    def forward(self, features):
        """Вычислить исход для каждого окна."""
        if self.kind == "last":
            return self.backbone(features)
        if self.kind == "mean":
            return self.net(features.mean(dim=1)).squeeze(-1)
        hidden, _ = self.gru(features)
        return self.out(hidden[:, -1]).squeeze(-1)


def build_training_windows(features, time_s, attempts, length):
    """Собрать окна с известными исходами для обучения."""
    features = np.asarray(features, np.float32)
    time_s = np.asarray(time_s, float)
    rows = []
    for end in range(length - 1, len(features), 2):
        t_end = float(time_s[end])
        for attempt in attempts:
            outcome = attempt["outcome"]
            if outcome not in ("made", "missed"):
                continue
            ref = float(attempt["reference_time_s"])
            if ref + 0.8 <= t_end <= ref + 1.8:
                rows.append((features[end - length + 1 : end + 1], float(outcome == "made")))
    if not rows:
        return np.zeros((0, length, features.shape[1]), np.float32), np.zeros((0,), np.float32)
    xs = np.stack([r[0] for r in rows]).astype(np.float32)
    ys = np.asarray([r[1] for r in rows], np.float32)
    return xs, ys


def train_model(model, train_data, validation_data, config):
    """Обучить переданную модель на CPU и восстановить её лучшие веса."""
    epochs = int(config["epochs"])
    lr = float(config["lr"])
    seed = int(config.get("seed", 17))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    (tx, ty), (vx, vy) = train_data, validation_data
    X = torch.from_numpy(np.asarray(tx, np.float32))
    Y = torch.from_numpy(np.asarray(ty, np.float32))
    Vx = torch.from_numpy(np.asarray(vx, np.float32))
    Vy = torch.from_numpy(np.asarray(vy, np.float32))

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    batch = int(config.get("batch_size", 64))
    patience = int(config.get("early_stopping", 20))

    best_brier, best_epoch, wait = float("inf"), 1, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    curve = []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X))
        total, count = 0.0, 0
        for i in range(0, len(X), batch):
            idx = perm[i : i + batch]
            if idx.numel() < 2:
                continue
            xb, yb = X[idx], Y[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * yb.numel()
            count += yb.numel()
        train_loss = total / max(count, 1)
        model.eval()
        with torch.inference_mode():
            prob = torch.sigmoid(model(Vx))
        brier = float((prob - Vy).pow(2).mean())
        curve.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_brier": brier,
            }
        )
        if brier < best_brier:
            best_brier = brier
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return {"curve": curve, "best_epoch": best_epoch}


def _pick_ball(detections, predicted, rim):
    """Выбрать лучшее наблюдение мяча рядом с прогнозом.

    Игнорируем низкоуверенные детекции (шум/мусор на фоне площадки) и
    выбираем уверенное наблюдение, ближайшее к прогнозу фильтра. У самого
    кольца ограничение на уверенность ослабляем: там мусор встречается
    реже, а настоящий мяч может быть детектирован слабее.
    """
    rim_width = rim[2] - rim[0]
    center_x = (rim[0] + rim[2]) / 2
    balls = []
    for d in detections:
        if d.get("class_name") != "ball":
            continue
        box = np.asarray(d["box"], float)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        near_rim = (
            abs(cx - center_x) <= 1.5 * rim_width
            and rim[1] - 1.2 * rim_width <= cy <= rim[3] + 0.5 * rim_width
        )
        floor = 0.25 if near_rim else 0.35
        if d.get("confidence", 0.0) < floor:
            continue
        balls.append((cx, cy, d))
    if not balls:
        return None
    if predicted is not None:
        best, best_dist = None, float("inf")
        for cx, cy, d in balls:
            dist = (cx - predicted[0]) ** 2 + (cy - predicted[1]) ** 2
            if dist < best_dist:
                best, best_dist = d, dist
        return best
    return max((d for _, _, d in balls), key=lambda d: d.get("confidence", 0.0))


def track_ball(detections, rim):
    """Сопроводить мяч фильтром Калмана, используя только доступное прошлое."""
    # pylint: disable=unused-argument
    kf = None
    last_time = None
    rows = []
    for frame, time_s, dets in detections:
        if kf is not None and last_time is not None:
            dt = float(time_s - last_time)
            if dt > 0:
                kf.predict(dt)
        predicted = kf.x[:2].copy() if kf is not None else None
        chosen = _pick_ball(dets, predicted, rim)
        accepted = None
        if chosen is not None:
            box = np.asarray(chosen["box"], float)
            z = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], float)
            if kf is None:
                kf = _BallKalman(z[0], z[1])
                accepted = z.copy()
            elif kf.innov_dist(z) < 1e9:
                kf.update(z)
                accepted = z.copy()
        if kf is None:
            rows.append(
                dict(frame=frame, time_s=time_s, observed=False, x=None, y=None, vx=None, vy=None)
            )
        else:
            observed = accepted is not None
            rows.append(
                dict(
                    frame=frame,
                    time_s=time_s,
                    observed=observed,
                    x=float(accepted[0]) if observed else float(kf.x[0]),
                    y=float(accepted[1]) if observed else float(kf.x[1]),
                    vx=float(kf.x[2]),
                    vy=float(kf.x[3]),
                )
            )
        last_time = time_s
    return rows


def find_candidates(tracks, rim):
    """Найти попытки броска по движению мяча около кольца.

    Кандидат — наблюдаемый кадр, в котором мяч только что пересёк линию
    подлёта сверху вниз в горизонтальном коридоре у кольца. Пересечение
    ищется между последним «верхним» и текущим «нижним» наблюдениями:
    если между ними были пропуски (нет непрерывных наблюдений), это не
    мешает — время кандидата берём у текущего наблюдаемого кадра, без
    интерполяции назад. Повторные срабатывания подавляются интервалом 1.2 с.
    """
    rim = np.asarray(rim, float)
    rim_width = rim[2] - rim[0]
    line_y = rim[1] - 0.5 * rim_width
    center_x = (rim[0] + rim[2]) / 2
    candidates = []
    last_candidate = -float("inf")
    last_above = None
    for tr in tracks:
        if not tr["observed"] or tr["x"] is None or tr["y"] is None:
            continue
        y, x, t = tr["y"], tr["x"], float(tr["time_s"])
        if y <= line_y:
            last_above = t
            continue
        if not tr["vy"] or tr["vy"] <= 0:
            continue
        if abs(x - center_x) > 1.5 * rim_width:
            continue
        if last_above is None or t <= last_above:
            continue
        if t - last_candidate >= 1.2:
            candidates.append(t)
            last_candidate = t
        last_above = None
    return candidates
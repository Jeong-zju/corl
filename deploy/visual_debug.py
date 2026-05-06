from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _array(value: Any, *, dtype=None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _even(value: int) -> int:
    value = int(value)
    return value if value % 2 == 0 else value + 1


def _fit_bgr_to_canvas(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = np.full((height, width, 3), pad_color, dtype=np.uint8)
    src_h, src_w = int(frame.shape[0]), int(frame.shape[1])
    if src_h <= 0 or src_w <= 0:
        return canvas, (0, 0, width, height)
    scale = min(width / float(src_w), height / float(src_h))
    fit_w = max(1, min(width, int(round(src_w * scale))))
    fit_h = max(1, min(height, int(round(src_h * scale))))
    x0 = (width - fit_w) // 2
    y0 = (height - fit_h) // 2
    resized = cv2.resize(frame, (fit_w, fit_h), interpolation=cv2.INTER_LINEAR)
    canvas[y0 : y0 + fit_h, x0 : x0 + fit_w] = resized
    return canvas, (x0, y0, fit_w, fit_h)


def _image_to_bgr(
    image: np.ndarray | None,
    *,
    color_order: str,
    width: int,
    height: int,
    preserve_aspect: bool = True,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8), (0, 0, width, height)
    frame = np.asarray(image, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        return np.zeros((height, width, 3), dtype=np.uint8), (0, 0, width, height)
    if color_order == "rgb":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if preserve_aspect:
        frame, rect = _fit_bgr_to_canvas(frame, width=width, height=height)
        return np.ascontiguousarray(frame), rect
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(frame), (0, 0, width, height)


def _draw_label(image: np.ndarray, text: str, x: int, y: int, scale: float = 0.5) -> None:
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _normalize_heatmap(values: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(values, dtype=np.float32)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    vmin = float(heatmap.min()) if heatmap.size else 0.0
    vmax = float(heatmap.max()) if heatmap.size else 0.0
    if vmax <= vmin + 1e-12:
        return np.zeros_like(heatmap, dtype=np.uint8)
    normalized = (heatmap - vmin) / (vmax - vmin)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _attention_lift_heatmap(
    values: np.ndarray,
    *,
    uniform_token_mass: float,
    red_lift: float = 3.0,
) -> np.ndarray:
    heatmap = np.asarray(values, dtype=np.float32)
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    if heatmap.size == 0 or uniform_token_mass <= 1e-12:
        return np.zeros_like(heatmap, dtype=np.uint8)
    lift = heatmap / float(uniform_token_mass)
    # Show attention above a uniform image-token prior. This avoids turning tiny
    # per-frame min/max differences into saturated red borders.
    normalized = (lift - 1.0) / max(1e-6, red_lift - 1.0)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _edge_mass_ratio(values: np.ndarray) -> float:
    heatmap = np.asarray(values, dtype=np.float32)
    if heatmap.ndim != 2 or heatmap.size == 0:
        return 0.0
    total = float(np.nan_to_num(heatmap, nan=0.0).sum())
    if total <= 1e-12:
        return 0.0
    edge_mask = np.zeros(heatmap.shape, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    return float(heatmap[edge_mask].sum() / total)


def _draw_bar_row(
    canvas: np.ndarray,
    *,
    title: str,
    values: np.ndarray | None,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
    symmetric: bool = False,
) -> None:
    cv2.putText(canvas, title, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (35, 35, 35), 1, cv2.LINE_AA)
    if values is None or values.size == 0:
        cv2.putText(canvas, "n/a", (x + 130, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (120, 120, 120), 1, cv2.LINE_AA)
        return
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    n = int(values.shape[0])
    gap = 5
    slot_w = max(1, int((width - gap * max(0, n - 1)) / max(1, n)))
    if symmetric:
        max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        denom = max(max_abs, 1e-6)
        center_y = y + height // 2
        cv2.line(canvas, (x, center_y), (x + width, center_y), (180, 180, 180), 1)
        for idx, value in enumerate(values):
            bar_h = int((abs(float(value)) / denom) * (height // 2 - 2))
            x0 = x + idx * (slot_w + gap)
            if value >= 0:
                y0, y1 = center_y - bar_h, center_y
                bar_color = color
            else:
                y0, y1 = center_y, center_y + bar_h
                bar_color = (80, 80, 210)
            cv2.rectangle(canvas, (x0, y0), (x0 + slot_w, y1), bar_color, -1)
            cv2.putText(canvas, str(idx), (x0 + 2, y + height + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1, cv2.LINE_AA)
        return

    vmax = float(np.max(values)) if values.size else 0.0
    denom = max(vmax, 1e-6)
    for idx, value in enumerate(values):
        bar_h = int((max(float(value), 0.0) / denom) * (height - 2))
        x0 = x + idx * (slot_w + gap)
        y0 = y + height - bar_h
        cv2.rectangle(canvas, (x0, y), (x0 + slot_w, y + height), (230, 230, 230), 1)
        cv2.rectangle(canvas, (x0 + 1, y0), (x0 + slot_w - 1, y + height - 1), color, -1)
        cv2.putText(canvas, str(idx), (x0 + 2, y + height + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1, cv2.LINE_AA)


def _slot_vector(value: Any, *, num_slots: int | None = None) -> np.ndarray | None:
    arr = _array(value, dtype=np.float32)
    if arr is None:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr.reshape(-1)
    if arr.ndim == 2:
        if num_slots is not None and arr.shape[0] == num_slots:
            return arr.reshape(num_slots, -1).mean(axis=1)
        return arr[0].reshape(-1)
    if arr.ndim >= 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        if num_slots is not None and arr.shape[0] == num_slots:
            return arr.reshape(num_slots, -1).mean(axis=1)
        return arr.reshape(arr.shape[0], -1).mean(axis=1)
    return arr.reshape(-1)


def _infer_num_slots(
    *,
    slot: dict[str, Any],
    stats: dict[str, Any],
    history: list[dict[str, Any]],
) -> int:
    try:
        num_slots = int(stats.get("num_slots", 0) or slot.get("num_slots", 0) or 0)
    except (TypeError, ValueError):
        num_slots = 0
    if num_slots > 0:
        return num_slots
    for payload in [slot, *history]:
        for key in (
            "routing_weights",
            "routing_distribution",
            "routing_weights_zero_signature",
            "write_strength",
            "readout_weights",
            "memory_next_norm",
            "memory_delta_norm",
        ):
            values = _slot_vector(payload.get(key), num_slots=None)
            if values is not None and values.size:
                return int(values.size)
    return 0


def _slot_colors(num_slots: int) -> list[tuple[int, int, int]]:
    if num_slots <= 0:
        return []
    colors = []
    for idx in range(num_slots):
        hue = int(round((idx / max(1, num_slots)) * 179))
        hsv = np.array([[[hue, 175, 215]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def _draw_slot_legend(
    canvas: np.ndarray,
    *,
    colors: list[tuple[int, int, int]],
    x: int,
    y: int,
    width: int,
) -> int:
    if not colors:
        return y
    col_w = 64
    cols = max(1, width // col_w)
    for idx, color in enumerate(colors):
        col = idx % cols
        row = idx // cols
        x0 = x + col * col_w
        y0 = y + row * 18
        cv2.line(canvas, (x0, y0), (x0 + 20, y0), color, 3, cv2.LINE_AA)
        cv2.putText(canvas, f"s{idx}", (x0 + 25, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (55, 55, 55), 1, cv2.LINE_AA)
    return y + ((len(colors) - 1) // cols + 1) * 18


def _history_matrix(
    history: list[dict[str, Any]],
    *,
    key: str,
    num_slots: int,
) -> np.ndarray | None:
    if not history or num_slots <= 0:
        return None
    matrix = np.full((len(history), num_slots), np.nan, dtype=np.float32)
    found = False
    for row_idx, payload in enumerate(history):
        values = _slot_vector(payload.get(key), num_slots=num_slots)
        if values is None or values.size == 0:
            continue
        count = min(num_slots, int(values.size))
        matrix[row_idx, :count] = values[:count]
        found = True
    return matrix if found else None


def _draw_multislot_history_plot(
    canvas: np.ndarray,
    *,
    title: str,
    matrix: np.ndarray | None,
    colors: list[tuple[int, int, int]],
    x: int,
    y: int,
    width: int,
    height: int,
    current_step: int | None,
    total_steps: int | None,
    fixed_min: float | None = None,
    fixed_max: float | None = None,
) -> None:
    cv2.putText(canvas, title, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (205, 205, 205), 1)
    if matrix is None or matrix.size == 0 or np.all(np.isnan(matrix)):
        cv2.putText(canvas, "n/a", (x + 12, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1, cv2.LINE_AA)
        return

    values = matrix[np.isfinite(matrix)]
    y_min = float(values.min()) if fixed_min is None else float(fixed_min)
    y_max = float(values.max()) if fixed_max is None else float(fixed_max)
    if y_max <= y_min + 1e-9:
        y_max = y_min + 1.0
    cv2.line(canvas, (x, y + height // 2), (x + width, y + height // 2), (232, 232, 232), 1)

    total = max(int(total_steps or matrix.shape[0]), int(matrix.shape[0]), 1)
    denom = max(1, total - 1)

    def point(step_idx: int, value: float) -> tuple[int, int]:
        px = x + int(round((step_idx / denom) * width))
        py = y + height - int(round(((float(value) - y_min) / (y_max - y_min)) * height))
        return px, py

    for slot_idx in range(matrix.shape[1]):
        color = colors[slot_idx % len(colors)] if colors else (50, 145, 215)
        last_point: tuple[int, int] | None = None
        last_valid = False
        for step_idx, value in enumerate(matrix[:, slot_idx]):
            valid = bool(np.isfinite(value))
            if valid:
                current_point = point(step_idx, float(value))
                if last_valid and last_point is not None:
                    cv2.line(canvas, last_point, current_point, color, 2, cv2.LINE_AA)
                cv2.circle(canvas, current_point, 2, color, -1, cv2.LINE_AA)
                last_point = current_point
            last_valid = valid

    if current_step is not None:
        current_step = max(0, min(int(current_step), total - 1))
        current_x = x + int(round((current_step / denom) * width))
        cv2.line(canvas, (current_x, y - 2), (current_x, y + height + 2), (25, 25, 25), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"t={current_step}", (min(current_x + 6, x + width - 58), y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (25, 25, 25), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"{y_min:.3g}", (x + 4, y + height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (95, 95, 95), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{y_max:.3g}", (x + 4, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (95, 95, 95), 1, cv2.LINE_AA)


def _slot_vector_or_zeros(
    value: Any,
    *,
    num_slots: int,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    values = _slot_vector(value, num_slots=num_slots)
    if values is None or values.size == 0:
        if fallback is not None:
            return np.asarray(fallback, dtype=np.float32).reshape(num_slots)
        return np.zeros((num_slots,), dtype=np.float32)
    result = np.zeros((num_slots,), dtype=np.float32)
    count = min(num_slots, int(values.size))
    result[:count] = values[:count]
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _blend_bgr(
    base: tuple[int, int, int],
    color: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    amount = float(np.clip(amount, 0.0, 1.0))
    return tuple(
        int(round(float(base[idx]) * (1.0 - amount) + float(color[idx]) * amount))
        for idx in range(3)
    )


def _draw_metric_cell(
    canvas: np.ndarray,
    *,
    value: float,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
    denom: float,
    signed: bool = False,
    text: str | None = None,
) -> None:
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (226, 226, 226), 1)
    inner_x = x + 2
    inner_y = y + 2
    inner_w = max(1, width - 4)
    inner_h = max(1, height - 4)
    if signed:
        center_x = inner_x + inner_w // 2
        cv2.line(canvas, (center_x, inner_y), (center_x, inner_y + inner_h), (184, 184, 184), 1)
        intensity = min(1.0, abs(float(value)) / max(float(denom), 1e-6))
        bar_w = int(round(intensity * (inner_w // 2)))
        if value >= 0.0:
            fill_color = _blend_bgr((242, 242, 242), color, 0.35 + 0.65 * intensity)
            cv2.rectangle(
                canvas,
                (center_x, inner_y),
                (center_x + bar_w, inner_y + inner_h),
                fill_color,
                -1,
            )
        else:
            fill_color = _blend_bgr((242, 242, 242), (95, 80, 205), 0.35 + 0.65 * intensity)
            cv2.rectangle(
                canvas,
                (center_x - bar_w, inner_y),
                (center_x, inner_y + inner_h),
                fill_color,
                -1,
            )
    else:
        intensity = min(1.0, max(0.0, float(value)) / max(float(denom), 1e-6))
        fill_color = _blend_bgr((242, 242, 242), color, 0.25 + 0.75 * intensity)
        bar_w = int(round(intensity * inner_w))
        cv2.rectangle(
            canvas,
            (inner_x, inner_y),
            (inner_x + bar_w, inner_y + inner_h),
            fill_color,
            -1,
        )
    label = text if text is not None else f"{float(value):.2f}"
    cv2.putText(
        canvas,
        label,
        (x + 5, y + max(14, height - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )


def _draw_video_mosaic(
    canvas: np.ndarray,
    *,
    images: dict[str, np.ndarray],
    color_order: str,
    camera_labels: dict[str, str],
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    cv2.putText(canvas, "video stream", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 45, 45), 1, cv2.LINE_AA)
    keys = list(images.keys())
    if not keys:
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (220, 220, 220), 1)
        cv2.putText(canvas, "no image payload", (x + 18, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (110, 110, 110), 1, cv2.LINE_AA)
        return

    cols = 1 if len(keys) == 1 else 2
    rows = int(np.ceil(len(keys) / float(cols)))
    gap = 8
    tile_w = max(1, (width - gap * (cols - 1)) // cols)
    tile_h = max(1, (height - gap * (rows - 1)) // rows)
    for idx, camera_key in enumerate(keys):
        row = idx // cols
        col = idx % cols
        tile_x = x + col * (tile_w + gap)
        tile_y = y + row * (tile_h + gap)
        frame, _ = _image_to_bgr(
            images.get(camera_key),
            color_order=color_order,
            width=tile_w,
            height=tile_h,
        )
        label = camera_labels.get(camera_key, camera_key.rsplit(".", 1)[-1])
        _draw_label(frame, label, 8, 22, scale=0.48)
        canvas[tile_y : tile_y + tile_h, tile_x : tile_x + tile_w] = frame


def _draw_slot_metric_matrix(
    canvas: np.ndarray,
    *,
    slot: dict[str, Any],
    stats: dict[str, Any],
    colors: list[tuple[int, int, int]],
    x: int,
    y: int,
    width: int,
    row_height: int,
) -> int:
    num_slots = len(colors)
    routing = _slot_vector_or_zeros(slot.get("routing_weights"), num_slots=num_slots)
    zero_routing_values = _slot_vector(slot.get("routing_weights_zero_signature"), num_slots=num_slots)
    zero_routing = _slot_vector_or_zeros(
        slot.get("routing_weights_zero_signature"),
        num_slots=num_slots,
        fallback=np.zeros((num_slots,), dtype=np.float32),
    )
    delta_values = _slot_vector(slot.get("routing_delta_from_zero_signature"), num_slots=num_slots)
    routing_delta = (
        routing - zero_routing
        if delta_values is None and zero_routing_values is not None
        else _slot_vector_or_zeros(slot.get("routing_delta_from_zero_signature"), num_slots=num_slots)
    )
    write_strength = _slot_vector_or_zeros(slot.get("write_strength"), num_slots=num_slots)
    readout = _slot_vector_or_zeros(slot.get("readout_weights"), num_slots=num_slots)
    memory_delta = _slot_vector_or_zeros(slot.get("memory_delta_norm"), num_slots=num_slots)
    memory_next = _slot_vector_or_zeros(slot.get("memory_next_norm"), num_slots=num_slots)

    uniform = 1.0 / float(max(1, num_slots))
    route_scale = max(float(np.max(routing)), uniform * 2.0, 1e-6)
    read_scale = max(float(np.max(readout)), uniform * 2.0, 1e-6)
    write_scale = max(float(np.max(write_strength)), uniform, 1e-6)
    mem_delta_scale = max(float(np.max(memory_delta)), 1e-6)
    mem_scale = max(float(np.max(memory_next)), 1e-6)
    sig_scale = max(float(np.max(np.abs(routing_delta))), uniform * 0.5, 1e-6)

    top_route = int(np.argmax(routing)) if num_slots else -1
    top_boost = int(np.argmax(routing_delta)) if num_slots else -1
    top_read = int(np.argmax(readout)) if num_slots else -1
    top_write = int(np.argmax(write_strength)) if num_slots else -1

    headline = (
        f"route=s{top_route} {routing[top_route]:.3f}   "
        f"sig_boost=s{top_boost} {routing_delta[top_boost]:+.3f}   "
        f"read=s{top_read} {readout[top_read]:.3f}   "
        f"write=s{top_write} {write_strength[top_write]:.3f}"
        if num_slots
        else "slot metrics unavailable"
    )
    cv2.putText(canvas, "slot difference dashboard", (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(canvas, headline, (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1, cv2.LINE_AA)

    header_y = y + 30
    col_gap = 7
    slot_w = 52
    badge_w = 76
    usable_w = max(1, width - slot_w - badge_w - col_gap * 6)
    metric_w = max(52, usable_w // 6)
    columns = [
        ("route", metric_w),
        ("sig +/-", metric_w),
        ("write", metric_w),
        ("read", metric_w),
        ("dmem", metric_w),
        ("mem", metric_w),
    ]
    x_positions = []
    cursor = x + slot_w
    for _, col_w in columns:
        x_positions.append(cursor)
        cursor += col_w + col_gap
    badge_x = min(x + width - badge_w, cursor)

    cv2.putText(canvas, "slot", (x + 4, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (85, 85, 85), 1, cv2.LINE_AA)
    for (label, _), col_x in zip(columns, x_positions, strict=False):
        cv2.putText(canvas, label, (col_x + 4, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (85, 85, 85), 1, cv2.LINE_AA)

    first_row_y = header_y + 10
    for slot_idx in range(num_slots):
        row_y = first_row_y + slot_idx * row_height
        if slot_idx == top_route:
            cv2.rectangle(canvas, (x, row_y - 3), (x + width, row_y + row_height - 2), (238, 246, 252), -1)
        cv2.rectangle(canvas, (x, row_y - 3), (x + 8, row_y + row_height - 2), colors[slot_idx], -1)
        cv2.putText(canvas, f"s{slot_idx}", (x + 14, row_y + row_height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)

        cell_h = max(18, row_height - 9)
        cell_y = row_y
        _draw_metric_cell(canvas, value=float(routing[slot_idx]), x=x_positions[0], y=cell_y, width=metric_w, height=cell_h, color=(50, 145, 215), denom=route_scale, text=f"{routing[slot_idx]:.2f}")
        _draw_metric_cell(canvas, value=float(routing_delta[slot_idx]), x=x_positions[1], y=cell_y, width=metric_w, height=cell_h, color=(45, 170, 90), denom=sig_scale, signed=True, text=f"{routing_delta[slot_idx]:+.2f}")
        _draw_metric_cell(canvas, value=float(write_strength[slot_idx]), x=x_positions[2], y=cell_y, width=metric_w, height=cell_h, color=(70, 170, 150), denom=write_scale, text=f"{write_strength[slot_idx]:.2f}")
        _draw_metric_cell(canvas, value=float(readout[slot_idx]), x=x_positions[3], y=cell_y, width=metric_w, height=cell_h, color=(215, 130, 55), denom=read_scale, text=f"{readout[slot_idx]:.2f}")
        _draw_metric_cell(canvas, value=float(memory_delta[slot_idx]), x=x_positions[4], y=cell_y, width=metric_w, height=cell_h, color=(155, 100, 190), denom=mem_delta_scale, text=f"{memory_delta[slot_idx]:.2f}")
        _draw_metric_cell(canvas, value=float(memory_next[slot_idx]), x=x_positions[5], y=cell_y, width=metric_w, height=cell_h, color=(110, 110, 110), denom=mem_scale, text=f"{memory_next[slot_idx]:.1f}")

        badges = []
        if slot_idx == top_route:
            badges.append("R")
        if slot_idx == top_boost:
            badges.append("S+")
        if slot_idx == top_read:
            badges.append("O")
        if slot_idx == top_write:
            badges.append("W")
        cv2.putText(canvas, " ".join(badges), (badge_x, row_y + row_height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (35, 35, 35), 1, cv2.LINE_AA)

    footer_y = first_row_y + num_slots * row_height + 18
    summary = (
        f"updates={stats.get('update_count', 0)} state_norm={float(stats.get('state_norm', 0.0)):.2f} "
        f"slot_std={float(slot.get('memory_next_slot_std', 0.0)):.3f} "
        f"path_sig_norm={float(slot.get('signature_embedding_norm', 0.0)):.2f} "
        f"delta_sig_norm={float(slot.get('delta_signature_embedding_norm', 0.0)):.2f}"
    )
    cv2.putText(canvas, summary, (x, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (75, 75, 75), 1, cv2.LINE_AA)
    return footer_y + 12


def _draw_slot_history_heatmap(
    canvas: np.ndarray,
    *,
    title: str,
    matrix: np.ndarray | None,
    x: int,
    y: int,
    width: int,
    height: int,
    current_step: int | None,
    total_steps: int | None,
    colors: list[tuple[int, int, int]],
    symmetric: bool = False,
    fixed_min: float | None = None,
    fixed_max: float | None = None,
) -> None:
    cv2.putText(canvas, title, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (205, 205, 205), 1)
    if matrix is None or matrix.size == 0 or np.all(np.isnan(matrix)):
        cv2.putText(canvas, "n/a", (x + 12, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1, cv2.LINE_AA)
        return

    values = np.nan_to_num(np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    num_steps, num_slots = values.shape
    label_w = 44
    heat_x = x + label_w
    heat_w = max(1, width - label_w - 6)
    heat_h = max(1, height - 8)
    heat_y = y + 4

    image = np.full((num_slots, num_steps, 3), 242, dtype=np.uint8)
    if symmetric:
        max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        denom = max(max_abs, abs(float(fixed_min or 0.0)), abs(float(fixed_max or 0.0)), 1e-6)
        for row in range(num_slots):
            for col in range(num_steps):
                value = float(values[col, row])
                amount = min(1.0, abs(value) / denom)
                color = (45, 170, 90) if value >= 0.0 else (95, 80, 205)
                image[row, col] = _blend_bgr((242, 242, 242), color, 0.2 + 0.8 * amount)
        min_label = f"-{denom:.2g}"
        max_label = f"+{denom:.2g}"
    else:
        finite_values = values[np.isfinite(values)]
        vmin = float(finite_values.min()) if fixed_min is None and finite_values.size else float(fixed_min or 0.0)
        vmax = float(finite_values.max()) if fixed_max is None and finite_values.size else float(fixed_max or 1.0)
        if vmax <= vmin + 1e-9:
            vmax = vmin + 1.0
        for row in range(num_slots):
            row_color = colors[row % len(colors)] if colors else (50, 145, 215)
            for col in range(num_steps):
                amount = np.clip((float(values[col, row]) - vmin) / (vmax - vmin), 0.0, 1.0)
                image[row, col] = _blend_bgr((242, 242, 242), row_color, 0.2 + 0.8 * amount)
        min_label = f"{vmin:.2g}"
        max_label = f"{vmax:.2g}"

    heat = cv2.resize(image, (heat_w, heat_h), interpolation=cv2.INTER_NEAREST)
    canvas[heat_y : heat_y + heat_h, heat_x : heat_x + heat_w] = heat
    row_h = heat_h / float(max(1, num_slots))
    for slot_idx in range(num_slots):
        label_y = int(round(heat_y + (slot_idx + 0.65) * row_h))
        if row_h >= 8:
            cv2.putText(canvas, f"s{slot_idx}", (x + 8, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (65, 65, 65), 1, cv2.LINE_AA)
    if current_step is not None:
        total = max(int(total_steps or num_steps), num_steps, 1)
        denom = max(1, total - 1)
        current_x = heat_x + int(round(max(0, min(int(current_step), total - 1)) / denom * heat_w))
        cv2.line(canvas, (current_x, heat_y - 2), (current_x, heat_y + heat_h + 2), (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, min_label, (heat_x + 4, y + height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(canvas, max_label, (heat_x + heat_w - 52, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (70, 70, 70), 1, cv2.LINE_AA)


def _render_slot_memory_video_panel(
    *,
    images: dict[str, np.ndarray],
    debug: dict[str, Any],
    history: list[dict[str, Any]],
    current_step: int | None,
    total_steps: int | None,
    color_order: str,
    camera_labels: dict[str, str],
    width: int,
    height: int,
) -> np.ndarray:
    slot = dict(debug.get("slot_memory") or {})
    stats = dict(debug.get("visual_memory_stats") or {})
    num_slots = _infer_num_slots(slot=slot, stats=stats, history=history)
    colors = _slot_colors(num_slots)

    row_height = 38 if num_slots <= 8 else 30 if num_slots <= 16 else 22
    matrix_height = 56 + max(1, num_slots) * row_height + 36
    history_height = max(78, min(220, max(1, num_slots) * 10))
    top_height = max(380, matrix_height)
    width = _even(max(int(width), 1280))
    height = _even(max(int(height), 94 + top_height + 44 + history_height * 2 + 42 + 20))
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)

    cv2.putText(canvas, "slot memory + video debug", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"enabled={stats.get('enabled', False)} initialized={stats.get('initialized', False)} "
        f"slots={num_slots} updates={stats.get('update_count', 0)} state_norm={float(stats.get('state_norm', 0.0)):.3f}",
        (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )

    left_x = 18
    top_y = 94
    gap = 24
    left_w = min(560, max(420, width // 2 - 80))
    right_x = left_x + left_w + gap
    right_w = width - right_x - 18
    _draw_video_mosaic(
        canvas,
        images=images,
        color_order=color_order,
        camera_labels=camera_labels,
        x=left_x,
        y=top_y,
        width=left_w,
        height=top_height,
    )

    if not slot or num_slots <= 0:
        cv2.putText(canvas, "No slot-memory debug payload yet.", (right_x, top_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (95, 95, 95), 1, cv2.LINE_AA)
        return canvas

    _draw_slot_metric_matrix(
        canvas,
        slot=slot,
        stats=stats,
        colors=colors,
        x=right_x,
        y=top_y,
        width=right_w,
        row_height=row_height,
    )

    history_y = top_y + top_height + 44
    routing_history = _history_matrix(history, key="routing_weights", num_slots=num_slots)
    routing_delta_history = _history_matrix(history, key="routing_delta_from_zero_signature", num_slots=num_slots)
    routing_vmax = None
    if routing_history is not None and not np.all(np.isnan(routing_history)):
        routing_vmax = max(
            float(np.nanmax(routing_history)),
            2.0 / float(max(1, num_slots)),
            1e-6,
        )
    _draw_slot_history_heatmap(
        canvas,
        title="routing focus over time",
        matrix=routing_history,
        x=left_x,
        y=history_y,
        width=width - 36,
        height=history_height,
        current_step=current_step,
        total_steps=total_steps,
        colors=colors,
        fixed_min=0.0,
        fixed_max=routing_vmax,
    )
    _draw_slot_history_heatmap(
        canvas,
        title="signature boost over time  (green=boost, purple=suppressed)",
        matrix=routing_delta_history,
        x=left_x,
        y=history_y + history_height + 42,
        width=width - 36,
        height=history_height,
        current_step=current_step,
        total_steps=total_steps,
        colors=colors,
        symmetric=True,
    )
    return canvas


def make_slot_memory_history_sample(debug: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(debug, dict):
        return {}
    slot = dict(debug.get("slot_memory") or {})
    stats = dict(debug.get("visual_memory_stats") or {})
    sample: dict[str, Any] = {}
    if "num_slots" in stats:
        sample["num_slots"] = stats["num_slots"]
    for key in (
        "routing_weights",
        "routing_distribution",
        "routing_weights_zero_signature",
        "routing_delta_from_zero_signature",
        "write_strength",
        "readout_weights",
        "memory_next_norm",
        "memory_delta_norm",
    ):
        if key in slot:
            value = _array(slot.get(key), dtype=np.float32)
            sample[key] = None if value is None else value.copy()
    return sample


def render_attention_panel(
    *,
    images: dict[str, np.ndarray],
    debug: dict[str, Any] | None,
    color_order: str,
    camera_labels: dict[str, str],
    overlay_alpha: float,
    query_step: int,
    tile_width: int = 300,
    tile_height: int = 300,
) -> np.ndarray:
    debug = debug or {}
    layout = dict(debug.get("decoder_token_layout") or {})
    attn = _array(debug.get("decoder_cross_attention"), dtype=np.float32)
    camera_keys = list(layout.get("camera_keys") or images.keys())
    num_cameras = int(layout.get("num_cameras", len(camera_keys)) or len(camera_keys))
    feature_h = int(layout.get("feature_h", 0) or 0)
    feature_w = int(layout.get("feature_w", 0) or 0)
    image_token_start = int(layout.get("image_token_start", 0) or 0)
    image_token_count = int(layout.get("image_token_count", 0) or 0)
    query_step = max(0, int(query_step))

    maps: list[np.ndarray | None] = [None for _ in range(max(num_cameras, len(camera_keys)))]
    token_attention_sum = 0.0
    uniform_image_token_mass = 0.0
    if attn is not None and attn.ndim == 2 and feature_h > 0 and feature_w > 0 and num_cameras > 0:
        query_idx = min(query_step, attn.shape[0] - 1)
        source = attn[query_idx]
        end = image_token_start + image_token_count
        if image_token_count == num_cameras * feature_h * feature_w and end <= source.shape[0]:
            image_attention = source[image_token_start:end].reshape(num_cameras, feature_h, feature_w)
            token_attention_sum = float(image_attention.sum())
            uniform_image_token_mass = token_attention_sum / float(max(1, image_token_count))
            maps = [image_attention[idx] for idx in range(num_cameras)]

    width = max(1, len(camera_keys)) * tile_width
    extra_labels = list(layout.get("extra_memory_token_labels") or [])
    extra_label_cols = max(1, width // 158)
    extra_label_rows = (
        0 if not extra_labels else ((len(extra_labels) - 1) // extra_label_cols) + 1
    )
    header_h = 44
    footer_h = 82 + max(0, extra_label_rows - 1) * 38
    canvas = np.full((header_h + tile_height + footer_h, width, 3), 245, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"decoder cross-attention overlay | query_step={query_step}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    for idx, camera_key in enumerate(camera_keys):
        x0 = idx * tile_width
        frame, content_rect = _image_to_bgr(
            images.get(camera_key),
            color_order=color_order,
            width=tile_width,
            height=tile_height,
        )
        heat = maps[idx] if idx < len(maps) else None
        if heat is not None:
            heat_u8 = _attention_lift_heatmap(
                heat,
                uniform_token_mass=uniform_image_token_mass,
            )
            rect_x, rect_y, rect_w, rect_h = content_rect
            heat_u8 = cv2.resize(heat_u8, (rect_w, rect_h), interpolation=cv2.INTER_LINEAR)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            content = frame[rect_y : rect_y + rect_h, rect_x : rect_x + rect_w]
            frame[rect_y : rect_y + rect_h, rect_x : rect_x + rect_w] = cv2.addWeighted(
                content,
                1.0 - overlay_alpha,
                heat_color,
                overlay_alpha,
                0.0,
            )
            heat_sum = float(np.asarray(heat, dtype=np.float32).sum())
            peak_lift = (
                float(np.asarray(heat, dtype=np.float32).max() / uniform_image_token_mass)
                if uniform_image_token_mass > 1e-12
                else 0.0
            )
            edge_mass = _edge_mass_ratio(heat)
        else:
            heat_sum = 0.0
            peak_lift = 0.0
            edge_mass = 0.0
        label = camera_labels.get(camera_key, camera_key.rsplit(".", 1)[-1])
        _draw_label(
            frame,
            f"{label} sum={heat_sum:.3f} peak={peak_lift:.1f}x edge={edge_mass:.0%}",
            8,
            22,
        )
        canvas[header_h : header_h + tile_height, x0 : x0 + tile_width] = frame

    footer_y = header_h + tile_height + 24
    cv2.putText(
        canvas,
        f"image-token attention total={token_attention_sum:.3f} | overlay shows >uniform lift, red ~= >=3x uniform",
        (12, footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )
    if attn is not None and attn.ndim == 2 and extra_labels:
        query_idx = min(query_step, attn.shape[0] - 1)
        values = attn[query_idx, : len(extra_labels)]
        max_v = max(float(np.max(values)), 1e-6)
        for idx, (label, value) in enumerate(zip(extra_labels, values, strict=False)):
            col = idx % extra_label_cols
            row = idx // extra_label_cols
            x = 12 + col * 158
            y = footer_y + 22 + row * 38
            bar_w = int(130 * float(value) / max_v)
            cv2.rectangle(canvas, (x, y - 12), (x + 130, y + 2), (222, 222, 222), 1)
            cv2.rectangle(canvas, (x, y - 11), (x + bar_w, y + 1), (70, 145, 210), -1)
            cv2.putText(canvas, f"{label}:{float(value):.3f}", (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (55, 55, 55), 1, cv2.LINE_AA)
    return canvas


def render_slot_memory_panel(
    *,
    debug: dict[str, Any] | None,
    images: dict[str, np.ndarray] | None = None,
    color_order: str = "rgb",
    camera_labels: dict[str, str] | None = None,
    history: list[dict[str, Any]] | None = None,
    current_step: int | None = None,
    total_steps: int | None = None,
    width: int = 760,
    height: int = 560,
) -> np.ndarray:
    debug = debug or {}
    slot = dict(debug.get("slot_memory") or {})
    stats = dict(debug.get("visual_memory_stats") or {})
    history = list(history or [])
    if images is not None:
        return _render_slot_memory_video_panel(
            images=dict(images),
            debug=debug,
            history=history,
            current_step=current_step,
            total_steps=total_steps,
            color_order=color_order,
            camera_labels=dict(camera_labels or {}),
            width=width,
            height=height,
        )
    num_slots = _infer_num_slots(slot=slot, stats=stats, history=history)
    if num_slots > 0:
        width = max(int(width), 260 + num_slots * 28)
    has_history = bool(history)
    if has_history:
        legend_cols = max(1, (int(width) - 48) // 64)
        legend_rows = ((max(1, num_slots) - 1) // legend_cols) + 1
        height = max(int(height), 1010 + max(0, legend_rows - 1) * 18)
    width = _even(width)
    height = _even(height)
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, "signature-indexed slot memory routing", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"enabled={stats.get('enabled', False)} initialized={stats.get('initialized', False)} "
        f"updates={stats.get('update_count', 0)} state_norm={float(stats.get('state_norm', 0.0)):.3f}",
        (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )
    if not slot:
        cv2.putText(canvas, "No slot-memory debug payload yet.", (18, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (95, 95, 95), 1, cv2.LINE_AA)
        return canvas

    routing = _slot_vector(slot.get("routing_weights"), num_slots=num_slots or None)
    zero_routing = _slot_vector(slot.get("routing_weights_zero_signature"), num_slots=num_slots or None)
    routing_delta = _slot_vector(slot.get("routing_delta_from_zero_signature"), num_slots=num_slots or None)
    gate = _slot_vector(slot.get("gate"), num_slots=num_slots or None)
    write_strength = _slot_vector(slot.get("write_strength"), num_slots=num_slots or None)
    readout = _slot_vector(slot.get("readout_weights"), num_slots=num_slots or None)
    memory_delta = _slot_vector(slot.get("memory_delta_norm"), num_slots=num_slots or None)
    memory_next = _slot_vector(slot.get("memory_next_norm"), num_slots=num_slots or None)

    cv2.putText(
        canvas,
        "routing is computed from path signature (+ delta signature if enabled) against previous memory slots",
        (18, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (75, 75, 75),
        1,
        cv2.LINE_AA,
    )
    left_x = 24
    bar_w = width - 48
    bar_y = 135
    colors = _slot_colors(num_slots)
    if has_history:
        legend_bottom = _draw_slot_legend(canvas, colors=colors, x=left_x, y=122, width=bar_w)
        plot_y = max(150, legend_bottom + 24)
        routing_history = _history_matrix(history, key="routing_weights", num_slots=num_slots)
        memory_history = _history_matrix(history, key="memory_next_norm", num_slots=num_slots)
        _draw_multislot_history_plot(
            canvas,
            title="routing weights over time",
            matrix=routing_history,
            colors=colors,
            x=left_x,
            y=plot_y,
            width=bar_w,
            height=140,
            current_step=current_step,
            total_steps=total_steps,
            fixed_min=0.0,
            fixed_max=max(1.0, float(np.nanmax(routing_history)) if routing_history is not None and not np.all(np.isnan(routing_history)) else 1.0),
        )
        _draw_multislot_history_plot(
            canvas,
            title="slot memory norm over time",
            matrix=memory_history,
            colors=colors,
            x=left_x,
            y=plot_y + 188,
            width=bar_w,
            height=140,
            current_step=current_step,
            total_steps=total_steps,
            fixed_min=0.0,
        )
        bar_y = plot_y + 376

    _draw_bar_row(canvas, title="routing weights", values=routing, x=left_x, y=bar_y, width=bar_w, height=46, color=(50, 145, 215))
    _draw_bar_row(canvas, title="zero-signature routing baseline", values=zero_routing, x=left_x, y=bar_y + 86, width=bar_w, height=46, color=(145, 145, 145))
    _draw_bar_row(canvas, title="routing delta caused by signature", values=routing_delta, x=left_x, y=bar_y + 172, width=bar_w, height=52, color=(40, 170, 90), symmetric=True)
    _draw_bar_row(canvas, title="slot write gate", values=gate, x=left_x, y=bar_y + 272, width=bar_w // 2 - 20, height=40, color=(90, 120, 210))
    _draw_bar_row(canvas, title="write strength", values=write_strength, x=left_x + bar_w // 2, y=bar_y + 272, width=bar_w // 2 - 10, height=40, color=(70, 170, 150))
    _draw_bar_row(canvas, title="readout weights", values=readout, x=left_x, y=bar_y + 361, width=bar_w // 2 - 20, height=36, color=(215, 130, 55))
    _draw_bar_row(canvas, title="memory delta norm", values=memory_delta, x=left_x + bar_w // 2, y=bar_y + 361, width=bar_w // 2 - 10, height=36, color=(155, 100, 190))

    summary = (
        f"emb_norms visual={float(slot.get('visual_embedding_norm', 0.0)):.2f} "
        f"state={float(slot.get('state_embedding_norm', 0.0)):.2f} "
        f"path_sig={float(slot.get('signature_embedding_norm', 0.0)):.2f} "
        f"delta_sig={float(slot.get('delta_signature_embedding_norm', 0.0)):.2f} "
        f"slot_id_scale={float(slot.get('slot_identity_scale', 0.0)):.3f} "
        f"slot_std={float(slot.get('memory_next_slot_std', 0.0)):.3f}"
    )
    cv2.putText(canvas, summary, (18, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 55, 55), 1, cv2.LINE_AA)
    if memory_next is not None and memory_next.size:
        cv2.putText(canvas, f"next_slot_norm_mean={float(memory_next.mean()):.3f}", (width - 235, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 55, 55), 1, cv2.LINE_AA)
    return canvas


def _stats_vector(stats: dict[str, Any] | None, key: str) -> np.ndarray | None:
    if not isinstance(stats, dict):
        return None
    value = stats.get(key)
    arr = _array(value, dtype=np.float32)
    if arr is None or arr.size == 0:
        return None
    return arr.reshape(-1)


def render_signature_panel(
    *,
    signature_debug: dict[str, Any] | None,
    width: int = 920,
    height: int = 540,
) -> np.ndarray:
    signature_debug = signature_debug or {}
    signature = _array(signature_debug.get("signature"), dtype=np.float32)
    delta = _array(signature_debug.get("delta_signature"), dtype=np.float32)
    stats = signature_debug.get("signature_stats")
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.putText(canvas, "path signature vs dataset distribution", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"backend={signature_debug.get('backend', 'unknown')} window={signature_debug.get('window_length', 0)}/"
        f"{signature_debug.get('history_length', 'full')} dataset={signature_debug.get('dataset_root') or 'unknown'}",
        (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (65, 65, 65),
        1,
        cv2.LINE_AA,
    )
    if signature is None:
        cv2.putText(canvas, "No path signature has been computed yet.", (18, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (95, 95, 95), 1, cv2.LINE_AA)
        return canvas

    signature = signature.reshape(-1)
    delta = np.zeros_like(signature) if delta is None else delta.reshape(-1)
    mean = _stats_vector(stats, "mean")
    std = _stats_vector(stats, "std")
    q10 = _stats_vector(stats, "q10")
    q90 = _stats_vector(stats, "q90")
    vmin = _stats_vector(stats, "min")
    vmax = _stats_vector(stats, "max")

    sig_norm = float(np.linalg.norm(signature))
    delta_norm = float(np.linalg.norm(delta))
    cv2.putText(canvas, f"dim={signature.size} | ||sig||={sig_norm:.3f} | ||delta||={delta_norm:.3f}", (18, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 55, 55), 1, cv2.LINE_AA)

    if mean is None or std is None or mean.size != signature.size:
        cv2.putText(
            canvas,
            "Dataset signature stats are missing. Run dataset signature-stat generation or provide meta/stats.json entries for observation.path_signature.",
            (18, 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (55, 55, 160),
            1,
            cv2.LINE_AA,
        )
        sample = signature[: min(96, signature.size)]
        mean = np.zeros_like(signature)
        std = np.ones_like(signature)
        selected = np.arange(sample.size)
    else:
        z = (signature - mean) / np.maximum(std, 1e-6)
        selected = np.argsort(-np.abs(z))[: min(96, signature.size)]
        out_of_range = 0.0
        if vmin is not None and vmax is not None and vmin.size == signature.size:
            out_of_range = float(np.mean((signature < vmin) | (signature > vmax)))
        cv2.putText(
            canvas,
            f"z_abs_mean={float(np.mean(np.abs(z))):.3f} z_abs_max={float(np.max(np.abs(z))):.3f} out_of_minmax={out_of_range * 100:.1f}%",
            (18, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )

    plot_x, plot_y = 48, 170
    plot_w, plot_h = width - 90, height - 210
    cv2.rectangle(canvas, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (210, 210, 210), 1)
    current = signature[selected]
    center = mean[selected] if mean is not None and mean.size == signature.size else np.zeros_like(current)
    spread = std[selected] if std is not None and std.size == signature.size else np.ones_like(current)
    low = q10[selected] if q10 is not None and q10.size == signature.size else center - spread
    high = q90[selected] if q90 is not None and q90.size == signature.size else center + spread
    y_values = np.concatenate([current, low, high, center])
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    if y_max <= y_min + 1e-9:
        y_max = y_min + 1.0

    def xy(idx: int, value: float) -> tuple[int, int]:
        x = plot_x + int((idx / max(1, len(selected) - 1)) * plot_w)
        y = plot_y + plot_h - int(((float(value) - y_min) / (y_max - y_min)) * plot_h)
        return x, y

    for idx in range(len(selected) - 1):
        cv2.line(canvas, xy(idx, low[idx]), xy(idx + 1, low[idx + 1]), (185, 205, 235), 1)
        cv2.line(canvas, xy(idx, high[idx]), xy(idx + 1, high[idx + 1]), (185, 205, 235), 1)
        cv2.line(canvas, xy(idx, center[idx]), xy(idx + 1, center[idx + 1]), (150, 150, 150), 1)
        cv2.line(canvas, xy(idx, current[idx]), xy(idx + 1, current[idx + 1]), (30, 90, 220), 2)
    cv2.putText(canvas, "blue=current | gray=dataset mean | pale band=q10/q90 or +/-1std", (plot_x, plot_y + plot_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (60, 60, 60), 1, cv2.LINE_AA)
    return canvas

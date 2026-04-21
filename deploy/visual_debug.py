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


def _first_row(value: Any) -> np.ndarray | None:
    arr = _array(value, dtype=np.float32)
    if arr is None:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 2:
        return arr[0]
    return arr.reshape(-1)


def _image_to_bgr(image: np.ndarray | None, *, color_order: str, width: int, height: int) -> np.ndarray:
    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    frame = np.asarray(image, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    if color_order == "rgb":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(frame)


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
    slot_w = max(14, int((width - gap * max(0, n - 1)) / max(1, n)))
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

    header_h = 44
    footer_h = 82
    width = max(1, len(camera_keys)) * tile_width
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
        frame = _image_to_bgr(
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
            heat_u8 = cv2.resize(heat_u8, (tile_width, tile_height), interpolation=cv2.INTER_LINEAR)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 1.0 - overlay_alpha, heat_color, overlay_alpha, 0.0)
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
    extra_labels = list(layout.get("extra_memory_token_labels") or [])
    if attn is not None and attn.ndim == 2 and extra_labels:
        query_idx = min(query_step, attn.shape[0] - 1)
        values = attn[query_idx, : len(extra_labels)]
        x = 12
        y = footer_y + 22
        max_v = max(float(np.max(values)), 1e-6)
        for label, value in zip(extra_labels, values, strict=False):
            bar_w = int(130 * float(value) / max_v)
            cv2.rectangle(canvas, (x, y - 12), (x + 130, y + 2), (222, 222, 222), 1)
            cv2.rectangle(canvas, (x, y - 11), (x + bar_w, y + 1), (70, 145, 210), -1)
            cv2.putText(canvas, f"{label}:{float(value):.3f}", (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (55, 55, 55), 1, cv2.LINE_AA)
            x += 158
            if x + 150 > width:
                break
    return canvas


def render_slot_memory_panel(
    *,
    debug: dict[str, Any] | None,
    width: int = 760,
    height: int = 560,
) -> np.ndarray:
    debug = debug or {}
    slot = dict(debug.get("slot_memory") or {})
    stats = dict(debug.get("visual_memory_stats") or {})
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

    routing = _first_row(slot.get("routing_weights"))
    zero_routing = _first_row(slot.get("routing_weights_zero_signature"))
    routing_delta = _first_row(slot.get("routing_delta_from_zero_signature"))
    gate = _first_row(slot.get("gate"))
    write_strength = _first_row(slot.get("write_strength"))
    readout = _first_row(slot.get("readout_weights"))
    memory_delta = _first_row(slot.get("memory_delta_norm"))
    memory_next = _first_row(slot.get("memory_next_norm"))

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
    _draw_bar_row(canvas, title="routing weights", values=routing, x=left_x, y=135, width=bar_w, height=46, color=(50, 145, 215))
    _draw_bar_row(canvas, title="zero-signature routing baseline", values=zero_routing, x=left_x, y=221, width=bar_w, height=46, color=(145, 145, 145))
    _draw_bar_row(canvas, title="routing delta caused by signature", values=routing_delta, x=left_x, y=307, width=bar_w, height=52, color=(40, 170, 90), symmetric=True)
    _draw_bar_row(canvas, title="slot write gate", values=gate, x=left_x, y=407, width=bar_w // 2 - 20, height=40, color=(90, 120, 210))
    _draw_bar_row(canvas, title="write strength", values=write_strength, x=left_x + bar_w // 2, y=407, width=bar_w // 2 - 10, height=40, color=(70, 170, 150))
    _draw_bar_row(canvas, title="readout weights", values=readout, x=left_x, y=496, width=bar_w // 2 - 20, height=36, color=(215, 130, 55))
    _draw_bar_row(canvas, title="memory delta norm", values=memory_delta, x=left_x + bar_w // 2, y=496, width=bar_w // 2 - 10, height=36, color=(155, 100, 190))

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

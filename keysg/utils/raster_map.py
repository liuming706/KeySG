from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger

UNKNOWN_VALUE = 205
FREE_VALUE = 255
OCCUPIED_VALUE = 0
CELL_CODES = {
    OCCUPIED_VALUE: 0,
    FREE_VALUE: 1,
    UNKNOWN_VALUE: 2,
}


def save_scene_raster_map(
    floor_rooms: Iterable[Tuple[object, Sequence[object]]],
    output_dir: str,
    *,
    up_axis: str = "y",
    resolution: float = 0.05,
    scene_height: Optional[float] = None,
    floor_clearance: float = 0.2,
    obstacle_min_height: float = 0.25,
    free_dilation_radius: int = 1,
    occupied_dilation_radius: int = 0,
) -> Optional[str]:
    if resolution <= 0:
        raise ValueError(f"scene_map.resolution must be positive, got {resolution}")

    up_idx, horiz_idx = _axis_indices(up_axis)
    layers = []
    for floor, rooms in floor_rooms:
        points = _floor_points(floor, rooms)
        if points.size == 0:
            continue

        zero_level = _zero_level(floor, rooms, points, up_idx)
        crop_height = _crop_height(
            floor, rooms, points, up_idx, zero_level, scene_height
        )
        if crop_height <= 0:
            logger.warning(
                "Skipping floor {} with non-positive map height {}",
                getattr(floor, "floor_id", "?"),
                crop_height,
            )
            continue

        vertical = points[:, up_idx]
        crop_mask = (vertical >= zero_level - floor_clearance) & (
            vertical <= zero_level + crop_height
        )
        if not np.any(crop_mask):
            continue

        cropped = points[crop_mask]
        cropped_vertical = cropped[:, up_idx]
        free_mask = cropped_vertical <= zero_level + floor_clearance
        occupied_mask = cropped_vertical >= zero_level + obstacle_min_height
        layers.append((cropped[:, horiz_idx], free_mask, occupied_mask))

    if not layers:
        logger.warning("No points available for scene raster map")
        return None

    all_xy = np.vstack([xy for xy, _, _ in layers])
    finite = np.isfinite(all_xy).all(axis=1)
    all_xy = all_xy[finite]
    if all_xy.size == 0:
        logger.warning("No finite points available for scene raster map")
        return None

    min_xy = np.floor(all_xy.min(axis=0) / resolution) * resolution
    max_xy = np.ceil(all_xy.max(axis=0) / resolution) * resolution
    width = max(1, int(np.ceil((max_xy[0] - min_xy[0]) / resolution)) + 1)
    height = max(1, int(np.ceil((max_xy[1] - min_xy[1]) / resolution)) + 1)

    raster = np.full((height, width), UNKNOWN_VALUE, dtype=np.uint8)
    for xy, free_mask, occupied_mask in layers:
        valid = np.isfinite(xy).all(axis=1)
        if not np.any(valid):
            continue
        cells = np.floor((xy[valid] - min_xy) / resolution).astype(np.int64)
        cols = np.clip(cells[:, 0], 0, width - 1)
        rows = np.clip(height - 1 - cells[:, 1], 0, height - 1)

        free = free_mask[valid]
        if np.any(free):
            raster[rows[free], cols[free]] = FREE_VALUE

        occupied = occupied_mask[valid]
        if np.any(occupied):
            raster[rows[occupied], cols[occupied]] = OCCUPIED_VALUE

    raster = _dilate_value(raster, FREE_VALUE, free_dilation_radius)
    raster = _dilate_value(raster, OCCUPIED_VALUE, occupied_dilation_radius)

    map_dir = os.path.join(output_dir, "umap")
    os.makedirs(map_dir, exist_ok=True)
    map_path = os.path.join(map_dir, "map.png")
    cv2.imwrite(map_path, raster)
    umap_path = os.path.join(map_dir, "umap.map")
    _save_umap_map(umap_path, raster, resolution, min_xy)
    logger.info("Saved scene raster map to {}", map_path)
    return map_path


def _axis_indices(up_axis: str) -> Tuple[int, Sequence[int]]:
    axis = (up_axis or "y").lower()
    if axis == "y":
        return 1, [0, 2]
    if axis == "z":
        return 2, [0, 1]
    raise ValueError(f"up_axis must be 'y' or 'z', got '{up_axis}'")


def _floor_points(floor: object, rooms: Sequence[object]) -> np.ndarray:
    pcd = getattr(floor, "pcd", None)
    if pcd is not None and len(pcd.points) > 0:
        return np.asarray(pcd.points)

    room_points = []
    for room in rooms:
        room_pcd = getattr(room, "pcd", None)
        if room_pcd is not None and len(room_pcd.points) > 0:
            room_points.append(np.asarray(room_pcd.points))
    if room_points:
        return np.vstack(room_points)
    return np.empty((0, 3), dtype=float)


def _zero_level(
    floor: object,
    rooms: Sequence[object],
    points: np.ndarray,
    up_idx: int,
) -> float:
    floor_zero = getattr(floor, "floor_zero_level", None)
    if floor_zero is not None:
        return float(floor_zero)

    room_zeros = [getattr(room, "zero_level", None) for room in rooms]
    room_zeros = [float(z) for z in room_zeros if z is not None]
    if room_zeros:
        return min(room_zeros)
    return float(np.nanmin(points[:, up_idx]))


def _crop_height(
    floor: object,
    rooms: Sequence[object],
    points: np.ndarray,
    up_idx: int,
    zero_level: float,
    configured_height: Optional[float],
) -> float:
    floor_height = getattr(floor, "floor_height", None)

    if configured_height is not None and configured_height > 0:
        return float(configured_height)

    if floor_height is not None and floor_height > 0:
        return float(floor_height)

    room_heights = [getattr(room, "height", None) for room in rooms]
    room_heights = [float(h) for h in room_heights if h is not None and h > 0]
    if room_heights:
        return max(room_heights)

    vertical = points[:, up_idx]
    return float(np.nanpercentile(vertical, 99.0) - zero_level)


def _dilate_value(raster: np.ndarray, value: int, radius: int) -> np.ndarray:
    if radius <= 0:
        return raster
    mask = (raster == value).astype(np.uint8)
    kernel_size = radius * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1).astype(bool)
    out = raster.copy()
    if value == FREE_VALUE:
        dilated &= out != OCCUPIED_VALUE
    out[dilated] = value
    return out


def _save_umap_map(
    path: str,
    raster: np.ndarray,
    resolution: float,
    min_xy: np.ndarray,
) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "map_data": _pack_raster_cells(raster),
        "meta_data": {
            "create_time": stamp,
            "free_thresh": None,
            "height": int(raster.shape[0]),
            "modify_time": stamp,
            "negate": None,
            "occupied_thresh": None,
            "origin": [float(min_xy[0]), float(min_xy[1]), 0.0],
            "resolution": float(resolution),
            "width": int(raster.shape[1]),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=3)
    logger.info("Saved umap metadata to {}", path)


def _pack_raster_cells(raster: np.ndarray) -> list[list[int]]:
    packed_rows = []
    for row in raster:
        packed_row = []
        for start in range(0, row.shape[0], 16):
            value = 0
            for offset, pixel in enumerate(row[start : start + 16]):
                code = CELL_CODES.get(int(pixel), CELL_CODES[UNKNOWN_VALUE])
                value |= code << (offset * 2)
            if value >= 2**31:
                value -= 2**32
            packed_row.append(value)
        packed_rows.append(packed_row)
    return packed_rows

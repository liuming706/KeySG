"""Dataset compatibility helpers.

These helpers keep KeySG pipeline code agnostic to dataset-specific frame
metadata.  In particular, the updated Replica dataloader may contain multiple
cameras, where intrinsics/depth scale depend on the RGB filename/camera key.
For datasets that do not expose these APIs, helpers fall back to the historical
single-camera attributes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def get_frame_rgb_path(dataset: Any, frame_idx: Optional[int]) -> Optional[str]:
    """Return RGB path for a dataset frame when available."""
    if frame_idx is None:
        return None
    data_list = getattr(dataset, "data_list", None)
    if data_list is None or frame_idx < 0 or frame_idx >= len(data_list):
        return None
    entry = data_list[frame_idx]
    if isinstance(entry, (list, tuple)) and entry:
        return entry[0]
    return None


def get_frame_camera_key(dataset: Any, frame_idx: Optional[int]) -> Optional[str]:
    """Infer a frame camera key for multi-camera datasets when supported."""
    rgb_path = get_frame_rgb_path(dataset, frame_idx)
    infer = getattr(dataset, "infer_camera_key", None)
    if callable(infer) and rgb_path:
        try:
            return infer(rgb_path)
        except Exception:
            return None
    return None


def get_frame_camera_context(
    dataset: Any,
    frame_idx: Optional[int],
    rgb: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Return frame-specific path, camera key, intrinsics and depth scale.

    The returned dict is safe for non-Replica datasets too.  Keys:
    ``rgb_path``, ``camera_key``, ``intrinsics``, ``depth_scale``.
    """
    rgb_path = get_frame_rgb_path(dataset, frame_idx)
    camera_key = get_frame_camera_key(dataset, frame_idx)

    if rgb is not None:
        h, w = rgb.shape[:2]
    else:
        h = int(getattr(dataset, "rgb_H", getattr(dataset, "depth_H", 0)))
        w = int(getattr(dataset, "rgb_W", getattr(dataset, "depth_W", 0)))

    intrinsics = None
    matrix_for_shape = getattr(dataset, "_camera_matrix_for_rgb_shape", None)
    if callable(matrix_for_shape) and w > 0 and h > 0:
        try:
            intrinsics = matrix_for_shape(w, h, camera_key=camera_key)
        except TypeError:
            intrinsics = matrix_for_shape(w, h)
        except Exception:
            intrinsics = None

    if intrinsics is None:
        intrinsics = getattr(dataset, "rgb_intrinsics", None)
    if intrinsics is None:
        intrinsics = getattr(dataset, "depth_intrinsics", None)
    intrinsics = intrinsics.copy() if hasattr(intrinsics, "copy") else intrinsics

    depth_scale = getattr(dataset, "depth_scale", 1000.0)
    rec_for_key = getattr(dataset, "_intrinsics_record_for_key", None)
    if callable(rec_for_key):
        try:
            depth_scale = rec_for_key(camera_key)[3]
        except Exception:
            pass

    return {
        "rgb_path": rgb_path,
        "camera_key": camera_key,
        "intrinsics": intrinsics,
        "depth_scale": depth_scale,
    }


def create_pcd_for_frame(
    dataset: Any,
    frame_idx: Optional[int],
    rgb: np.ndarray,
    depth: np.ndarray,
    pose: np.ndarray,
):
    """Create a point cloud using frame-specific camera metadata if supported."""
    ctx = get_frame_camera_context(dataset, frame_idx, rgb)
    try:
        return dataset.create_pcd(
            rgb,
            depth,
            pose,
            rgb_path=ctx["rgb_path"],
            camera_key=ctx["camera_key"],
        )
    except TypeError:
        return dataset.create_pcd(rgb, depth, pose)


def project_2d_mask_to_3d_for_frame(
    dataset: Any,
    frame_idx: Optional[int],
    mask_2d: np.ndarray,
    depth_image: np.ndarray,
    rgb_image: np.ndarray,
    camera_pose: np.ndarray,
):
    """Project a mask to 3D using frame-specific camera metadata if supported."""
    ctx = get_frame_camera_context(dataset, frame_idx, rgb_image)
    try:
        return dataset.project_2d_mask_to_3d(
            mask_2d,
            depth_image,
            rgb_image,
            camera_pose,
            camera_key=ctx["camera_key"],
        )
    except TypeError:
        return dataset.project_2d_mask_to_3d(
            mask_2d, depth_image, rgb_image, camera_pose
        )
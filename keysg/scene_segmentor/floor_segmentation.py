"""Floor segmentation algorithm using height histogram analysis."""

from __future__ import annotations
import os
from typing import List, Optional

import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .floor import Floor


class FloorSegmentation:
    """
    Segments a scene point cloud into floors using height histogram analysis.

    Algorithm:
    1. Create height histogram of the point cloud
    2. Find peaks corresponding to floor surfaces
    3. Cluster peaks and extract floor boundaries
    4. Create Floor objects for each detected floor
    """

    def __init__(
        self,
        full_pcd: o3d.geometry.PointCloud,
        save_intermediate: bool = False,
    ):
        self.full_pcd = full_pcd
        self.save_intermediate = save_intermediate
        self.floors: List[Floor] = []

    def segment_floors(
        self,
        output_path: Optional[str] = None,
        flip_zy: bool = False,
        up_idx: int = 1,
        resolution: float = 0.01,
        peak_distance: float = 0.2,
        peak_percentile: float = 90,
    ) -> List[Floor]:
        """
        Segment floors from the point cloud.

        Args:
            output_path: Path to save intermediate visualizations
            flip_zy: Deprecated. Use up_idx instead. If True, treated as up_idx=2.
            up_idx: Axis index representing the vertical direction (1 for Y-up, 2 for Z-up)
            resolution: Histogram bin size in meters
            peak_distance: Minimum distance between peaks in meters
            peak_percentile: Percentile threshold for peak detection

        Returns:
            List of Floor objects
        """
        logger.info("Starting floor segmentation...")

        if flip_zy and up_idx == 1:
            logger.warning("flip_zy is deprecated; use up_idx=2 instead. Treating flip_zy=True as up_idx=2.")
            up_idx = 2

        horiz_idx = [i for i in range(3) if i != up_idx]

        # Downsample and prepare points
        downpcd = self.full_pcd.voxel_down_sample(voxel_size=0.05)
        points = np.asarray(downpcd.points)
        logger.info(f"Processing {len(points)} points")

        # Create height histogram
        up_range = np.max(points[:, up_idx]) - np.min(points[:, up_idx])
        bins = max(int(up_range / resolution), 1)
        hist_counts, hist_edges = np.histogram(points[:, up_idx], bins=bins)
        hist_smooth = gaussian_filter1d(hist_counts, sigma=2)

        # Find peaks
        min_peak_dist = peak_distance / resolution
        min_height = np.percentile(hist_smooth, peak_percentile)
        peaks, _ = find_peaks(hist_smooth, distance=min_peak_dist, height=min_height)
        logger.info(f"Found {len(peaks)} peaks at heights: {hist_edges[peaks]}")

        if self.save_intermediate and output_path:
            self._save_histogram(hist_edges, hist_smooth, peaks, min_height, output_path)

        # Cluster peaks and create floors
        if len(peaks) > 0:
            boundaries = self._cluster_peaks_to_boundaries(peaks, hist_edges, hist_smooth, points, up_idx)
        else:
            logger.warning("No peaks found - creating single floor")
            boundaries = [[np.min(points[:, up_idx]), np.max(points[:, up_idx])]]

        self.floors = self._create_floors(boundaries, up_idx, horiz_idx)
        logger.info(f"Created {len(self.floors)} floors")
        return self.floors

    def _cluster_peaks_to_boundaries(
        self,
        peaks: np.ndarray,
        hist_edges: np.ndarray,
        hist_smooth: np.ndarray,
        points: np.ndarray,
        up_idx: int = 1,
    ) -> List[List[float]]:
        """Cluster peaks and derive floor boundaries."""
        peak_heights = hist_edges[peaks]
        clustering = DBSCAN(eps=1, min_samples=1).fit(peak_heights.reshape(-1, 1))
        labels = clustering.labels_

        # Extract representative peaks from each cluster
        clustered_peaks = []
        for label in np.unique(labels):
            cluster_peaks = peaks[labels == label]
            # Take the highest peak in each cluster
            best_idx = np.argmax(hist_smooth[cluster_peaks])
            clustered_peaks.append(hist_edges[cluster_peaks[best_idx]])

        clustered_peaks = sorted(clustered_peaks)

        # Create floor boundaries from consecutive peak pairs
        boundaries = []
        for i in range(0, len(clustered_peaks) - 1, 2):
            if i + 1 < len(clustered_peaks):
                boundaries.append([clustered_peaks[i], clustered_peaks[i + 1]])

        # Handle odd number of peaks
        if len(clustered_peaks) % 2 == 1:
            boundaries.append([clustered_peaks[-1], clustered_peaks[-1] + 2.5])

        # Extend boundaries to cover full range
        if boundaries:
            boundaries[0][0] = (boundaries[0][0] + np.min(points[:, up_idx])) / 2
            boundaries[-1][1] = (boundaries[-1][1] + np.max(points[:, up_idx])) / 2

        return boundaries

    def _create_floors(self, boundaries: List[List[float]], up_idx: int = 1, horiz_idx: List[int] = None) -> List[Floor]:
        """Create Floor objects from boundaries."""
        if horiz_idx is None:
            horiz_idx = [0, 2] if up_idx == 1 else [0, 1]

        floors = []
        for i, (up_min, up_max) in enumerate(boundaries):
            floor = Floor(str(i), name=f"floor_{i}")

            # Build AABB: unconstrained on horizontal axes, bounded on vertical axis
            min_bound = [np.inf, np.inf, np.inf]
            max_bound = [-np.inf, -np.inf, -np.inf]
            for ax in horiz_idx:
                min_bound[ax] = -np.inf
                max_bound[ax] = np.inf
            min_bound[up_idx] = up_min
            max_bound[up_idx] = up_max

            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=min_bound,
                max_bound=max_bound,
            )
            floor_pcd = self.full_pcd.crop(bbox)

            if len(floor_pcd.points) > 0:
                floor.pcd = floor_pcd
                floor.vertices = np.asarray(floor_pcd.get_axis_aligned_bounding_box().get_box_points())
                pts = np.asarray(floor_pcd.points)
                floor.floor_zero_level = float(np.min(pts[:, up_idx]))
                floor.floor_height = up_max - floor.floor_zero_level
                floors.append(floor)
                logger.info(f"Floor {i}: height={floor.floor_height:.2f}m, points={len(floor_pcd.points)}")

        return floors

    def _save_histogram(
        self,
        hist_edges: np.ndarray,
        hist_smooth: np.ndarray,
        peaks: np.ndarray,
        min_height: float,
        output_path: str,
    ) -> None:
        """Save histogram visualization."""
        try:
            import matplotlib.pyplot as plt
            os.makedirs(output_path, exist_ok=True)

            plt.figure(figsize=(10, 6))
            plt.plot(hist_edges[:-1], hist_smooth, label="Smoothed histogram")
            plt.plot(hist_edges[peaks], hist_smooth[peaks], "x", markersize=10, label="Peaks")
            plt.axhline(y=min_height, color="r", linestyle="--", label="Threshold")
            plt.xlabel("Height (m)")
            plt.ylabel("Point count")
            plt.title("Floor Detection - Height Histogram")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "floor_histogram.png"), dpi=150)
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to save histogram: {e}")

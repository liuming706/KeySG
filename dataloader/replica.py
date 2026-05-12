import sys
import os
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
import open3d as o3d
from loguru import logger

# merged 输出与 visualize_merged_replica_style 一致时，可选用 merge 脚本的同名函数；
# 未安装时 replica 内置 stem->tag 规则，仍可按 traj_imgname 配对（勿依赖「排序+zip」）。
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
try:
    import merge_session_images_mono_depth as ms
except ImportError:
    ms = None


def _format_timestamp_tag(ts: int, min_digits: int) -> str:
    """与 merge_session_images_mono_depth._format_timestamp_tag 一致。"""
    s = str(int(ts))
    if len(s) < min_digits:
        return s.zfill(min_digits)
    return s


def _timestamp_id_from_stem(stem: str, min_digits: int) -> str:
    """
    与 merge_session_images_mono_depth._timestamp_id_from_stem 一致：
    cam0_35300 → cam0_00035300（末段数字按 min_digits 补零）。
    """
    if "_" in stem:
        head, tail = stem.rsplit("_", 1)
        if not tail.isdigit():
            raise ValueError(
                f"无法从 stem 解析时间戳: {stem!r}（最后一个 '_' 之后须为纯数字）"
            )
        ts = int(tail)
        return f"{head}_{_format_timestamp_tag(ts, min_digits)}"
    if stem.isdigit():
        return _format_timestamp_tag(int(stem), min_digits)
    raise ValueError(
        f"无法从 stem 解析时间戳: {stem!r}（需含 '_' 且末段为数字，或整段为数字）"
    )


def _stem_normalize_for_merge_tag(stem: str, rgb_prefix: str) -> str:
    """若 traj 中误带 rgb_ 前缀（如 rgb_cam0_xxx），去掉一层以便与 id 推导一致。"""
    pfx = f"{rgb_prefix}_"
    if stem.startswith(pfx):
        return stem[len(pfx) :]
    return stem


def _merge_index_field_width(n_total: int, min_width: int) -> int:
    n = max(0, int(n_total) - 1)
    return max(len(str(n)), int(min_width))


def _read_depth_uint16_write_scale(root_dir: str) -> float:
    """JSON 无 camera.scale 时，与 merge 输出根目录下 depth_uint16_write_scale.txt 一致。"""
    p = Path(root_dir) / "depth_uint16_write_scale.txt"
    if not p.is_file():
        raise FileNotFoundError(str(p))
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            return float(line.split()[0])
        except ValueError:
            continue
    raise ValueError(f"无法解析: {p}")


def _intrinsics_for_image_size(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    w0: int,
    h0: int,
    w1: int,
    h1: int,
) -> np.ndarray:
    """标定分辨率 (w0,h0) 与当前图 (w1,h1) 不同时缩放 K（同 visualize_merged_replica_style）。"""
    if w1 == w0 and h1 == h0:
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
    sx = w1 / float(w0)
    sy = h1 / float(h0)
    return np.array(
        [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


class ReplicaDataset:
    """
    Dataset class for the Replica dataset, matching the interface of HM3DSemDataset.

    在保持原有接口的前提下，对 ``merged_from_gs_mesh`` 目录做了与
    ``visualize_merged_replica_style.py`` 对齐的补充（见 ``__init__`` / ``_get_data_list`` /
    ``create_pcd`` 内注释）。
    """

    def __init__(self, cfg):
        """
        Args:
            root_dir: Path to the root directory containing the dataset.
            transforms: Optional transformations to apply to the data.
        """
        self.root_dir = cfg.get("root_dir", "")
        self.depth_scale = cfg.get("depth_scale", 1000.0)
        self.depth_min = cfg.get("depth_min", 0.5)
        self.depth_max = cfg.get("depth_max", 4.0)
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root directory {self.root_dir} does not exist.")
        self.transforms = cfg.get("transforms", None)

        # 与 merge 默认输出一致：优先当前目录 cam_params，再父目录（KeySG Replica 原版为父目录优先）
        parent_dir = os.path.split(self.root_dir)[0]
        cam_params_path_parent = os.path.join(parent_dir, "cam_params.json")
        cam_params_path_current = os.path.join(self.root_dir, "cam_params.json")
        if os.path.exists(cam_params_path_current):
            cam_params_path = cam_params_path_current
        elif os.path.exists(cam_params_path_parent):
            cam_params_path = cam_params_path_parent
        else:
            raise FileNotFoundError(
                "cam_params.json not found in current or parent directory"
            )

        # merge 配对参数（与 merge_session_from_gs_renders 默认一致）
        self._output_indexing = cfg.get("output_indexing", "timestamp")
        self._timestamp_min_digits = int(cfg.get("timestamp_min_digits", 8))
        self._index_min_width = int(cfg.get("index_min_width", 4))
        self._rgb_prefix_cfg = cfg.get("rgb_prefix", None)
        self._depth_prefix_cfg = cfg.get("depth_prefix", "depth")

        sm_path = Path(self.root_dir) / "merge_from_gs_renders_summary.json"
        if sm_path.is_file():
            try:
                smj = json.loads(sm_path.read_text(encoding="utf-8"))
                if (
                    "timestamp_min_digits" not in cfg
                    and smj.get("timestamp_min_digits") is not None
                ):
                    self._timestamp_min_digits = int(smj["timestamp_min_digits"])
                if "output_indexing" not in cfg and smj.get("output_indexing"):
                    self._output_indexing = str(smj["output_indexing"])
                if (
                    "index_min_width" not in cfg
                    and smj.get("index_min_width") is not None
                ):
                    self._index_min_width = int(smj["index_min_width"])
                if "rgb_prefix" not in cfg and smj.get("results_rgb_prefix"):
                    self._rgb_prefix_cfg = str(smj["results_rgb_prefix"]).rstrip("_")
                if "depth_prefix" not in cfg and smj.get("results_depth_prefix"):
                    self._depth_prefix_cfg = str(smj["results_depth_prefix"]).rstrip(
                        "_"
                    )
            except Exception as e:
                logger.warning(f"读取 merge_from_gs_renders_summary.json 失败: {e}")

        (
            self.depth_intrinsics,
            self.depth_scale,
            self._calib_w,
            self._calib_h,
        ) = self._load_depth_intrinsics(cam_params_path, self.root_dir)
        logger.info(f"depth_intrinsics=\n{self.depth_intrinsics}")
        logger.info(f"depth_scale={self.depth_scale}")

        self.data_list = self._get_data_list()
        print(f"Loaded {len(self.data_list)} images from {self.root_dir}")

        sample_rgb, sample_dep, _, _ = self.data_list[0]
        sample_image = self._load_image(sample_rgb)
        self.rgb_H = sample_image.shape[0]
        self.rgb_W = sample_image.shape[1]
        sample_depth = self._load_depth(sample_dep)
        self.depth_H = sample_depth.shape[0]
        self.depth_W = sample_depth.shape[1]
        self.name = "Replica"
        self.scene_name = self.root_dir.split("/")[-1]

    def _camera_matrix_for_rgb_shape(self, W: int, H: int) -> np.ndarray:
        """若 cam_params 含 w,h 且与 RGB 尺寸不同，则返回缩放后的 K。"""
        K0 = self.depth_intrinsics.astype(np.float64)
        if self._calib_w is None or self._calib_h is None:
            return K0.copy()
        return _intrinsics_for_image_size(
            float(K0[0, 0]),
            float(K0[1, 1]),
            float(K0[0, 2]),
            float(K0[1, 2]),
            int(self._calib_w),
            int(self._calib_h),
            W,
            H,
        )

    def __getitem__(self, idx):
        """
        Get a data sample based on the given index.

        Args:
            idx: Index of the data sample.

        Returns:
            RGB image, depth image, and pose as numpy arrays.
        """
        entry = self.data_list[idx]
        rgb_path, depth_path, pose_path = entry[0], entry[1], entry[2]
        pose_idx = entry[3] if len(entry) >= 4 else idx
        rgb_image = self._load_image(rgb_path)
        depth_image = self._load_depth(depth_path)
        pose = self._load_pose(pose_path, pose_idx)
        if self.transforms is not None:
            rgb_image = self.transforms(rgb_image)
            depth_image = self.transforms(depth_image)
        return rgb_image, depth_image, pose

    def _results_dir(self) -> Path:
        rgb_dir = os.path.join(self.root_dir, "results")
        return Path(rgb_dir) if os.path.isdir(rgb_dir) else Path(self.root_dir)

    def _get_data_list(self):
        """
        Get a list of RGB-D data samples based on the dataset format and mode.

        Returns:
            List of tuples (rgb_path, depth_path, pose_path) 或
            (rgb_path, depth_path, pose_path, pose_line_index)（存在 traj_imgname 且配对成功时）。
        """
        results_dir = self._results_dir()
        pose_path = os.path.join(self.root_dir, "traj.txt")
        traj_img = Path(self.root_dir) / "traj_imgname.txt"

        if traj_img.is_file():
            try:
                return self._get_data_list_from_traj_imgname(
                    results_dir, pose_path, traj_img
                )
            except Exception as e:
                logger.warning(f"traj_imgname 配对失败，回退为文件名排序: {e}")

        rgb_data_list = []
        depth_data_list = []
        for file in os.listdir(str(results_dir)):
            if file.startswith("frame"):
                rgb_data_list.append(str(results_dir / file))
            elif file.startswith("rgb"):
                rgb_data_list.append(str(results_dir / file))
            elif file.startswith("depth"):
                depth_data_list.append(str(results_dir / file))
        rgb_data_list.sort()
        depth_data_list.sort()
        pose_data_list = [pose_path] * len(rgb_data_list)
        if len(rgb_data_list) != len(depth_data_list):
            raise ValueError("Mismatch in number of RGB and depth files.")
        return [
            (r, d, p, i)
            for i, (r, d, p) in enumerate(
                zip(rgb_data_list, depth_data_list, pose_data_list)
            )
        ]

    def _parse_traj_imgname(self, path: Path) -> List[str]:
        names: List[str] = []
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 17:
                continue
            names.append(" ".join(parts[16:]).strip())
        if not names:
            raise ValueError(f"{path}: 无有效行（每行至少 17 列）")
        return names

    def _detect_rgb_depth_prefix(self, results_dir: Path) -> Tuple[str, str]:
        if any(results_dir.glob("rgb_*")):
            return "rgb", "depth"
        if any(results_dir.glob("frame_*")):
            return "frame", "depth"
        return "rgb", "depth"

    def _merge_tag_candidates_timestamp(self, stem: str) -> List[Tuple[int, str]]:
        """timestamp 模式：先试 cfg/summary 的位数，再在 4..12 间探测，避免与 merge 位数不一致时整段失败。"""
        d0 = int(self._timestamp_min_digits)
        order = [d0] + [d for d in range(4, 13) if d != d0]
        out: List[Tuple[int, str]] = []
        seen: set[str] = set()
        for md in order:
            try:
                if ms is not None:
                    tag = ms._timestamp_id_from_stem(stem, md)
                else:
                    tag = _timestamp_id_from_stem(stem, md)
            except ValueError:
                continue
            if tag in seen:
                continue
            seen.add(tag)
            out.append((md, tag))
        return out

    def _expected_merged_filenames(
        self,
        source_basename: str,
        *,
        idx: int,
        n_total: int,
        rgb_prefix: str,
        depth_prefix: str,
    ) -> Tuple[str, str]:
        stem_raw = Path(source_basename).stem
        stem = _stem_normalize_for_merge_tag(stem_raw, rgb_prefix)
        ext = Path(source_basename).suffix.lower() or ".png"
        if self._output_indexing == "timestamp":
            if ms is not None:
                tag = ms._timestamp_id_from_stem(stem, self._timestamp_min_digits)
            else:
                tag = _timestamp_id_from_stem(stem, self._timestamp_min_digits)
        else:
            if ms is not None:
                w = ms._index_field_width(n_total, min_width=self._index_min_width)
            else:
                w = _merge_index_field_width(n_total, self._index_min_width)
            tag = f"{idx:0{w}d}"
        return f"{rgb_prefix}_{tag}{ext}", f"{depth_prefix}_{tag}.png"

    def _get_data_list_from_traj_imgname(
        self, results_dir: Path, pose_path: str, traj_img: Path
    ) -> List[Tuple[str, str, str, int]]:
        names = self._parse_traj_imgname(traj_img)
        traj_p = Path(self.root_dir) / "traj.txt"
        traj = np.loadtxt(str(traj_p), dtype=np.float64)
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
        if traj.shape[1] != 16:
            raise ValueError("traj.txt 每行须为 16 个数")
        n = traj.shape[0]
        if len(names) != n:
            raise ValueError(f"traj 行数 {n} 与 traj_imgname 条目 {len(names)} 不一致")

        rgb_pfx, dep_pfx = self._detect_rgb_depth_prefix(results_dir)
        if self._rgb_prefix_cfg:
            rgb_pfx = str(self._rgb_prefix_cfg).rstrip("_")
        if self._depth_prefix_cfg:
            dep_pfx = str(self._depth_prefix_cfg).rstrip("_")

        out: List[Tuple[str, str, str, int]] = []
        skipped = 0
        for fi in range(n):
            ext = Path(names[fi]).suffix.lower() or ".png"
            if self._output_indexing == "timestamp":
                stem = _stem_normalize_for_merge_tag(Path(names[fi]).stem, rgb_pfx)
                found: Optional[Tuple[str, str]] = None
                for _md, tag in self._merge_tag_candidates_timestamp(stem):
                    rgb_n = f"{rgb_pfx}_{tag}{ext}"
                    dep_n = f"{dep_pfx}_{tag}.png"
                    rp = results_dir / rgb_n
                    dp = results_dir / dep_n
                    if rp.is_file() and dp.is_file():
                        found = (str(rp), str(dp))
                        break
                if found:
                    out.append((found[0], found[1], pose_path, fi))
                else:
                    skipped += 1
                    logger.warning(
                        f"跳过 fi={fi}，在 results 中未找到与 {names[fi]!r} 匹配的 "
                        f"{rgb_pfx}_* / {dep_pfx}_*（已试 timestamp 位数探测）"
                    )
            else:
                rgb_n, dep_n = self._expected_merged_filenames(
                    names[fi],
                    idx=fi,
                    n_total=n,
                    rgb_prefix=rgb_pfx,
                    depth_prefix=dep_pfx,
                )
                rp = str(results_dir / rgb_n)
                dp = str(results_dir / dep_n)
                if os.path.isfile(rp) and os.path.isfile(dp):
                    out.append((rp, dp, pose_path, fi))
                else:
                    skipped += 1
                    logger.warning(f"跳过 fi={fi}，缺少 {rgb_n} 或 {dep_n}")
        if not out:
            raise RuntimeError("traj_imgname 配对后无有效帧")
        return out

    def _load_image(self, path):
        """
        Load the RGB image from the given path.

        Args:
            path: Path to the RGB image file.

        Returns:
            RGB image as a numpy array.
        """
        rgb_image = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def _load_depth(self, path):
        """
        Load the depth image from the given path.

        Args:
            path: Path to the depth image file.

        Returns:
            Depth image as a numpy array.
        """
        depth_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise FileNotFoundError(f"无法读取深度: {path}")
        if depth_image.ndim == 3:
            depth_image = depth_image[:, :, 0]
        return depth_image

    def _load_pose(self, path, idx):
        """
        Load the camera pose from the given path.

        Args:
            path: Path to the camera pose file.

        Returns:
            Camera pose as a numpy array (4x4 matrix).
        """
        with open(path, "r") as file:
            lines = file.readlines()
            if 0 <= idx < len(lines):
                line = lines[idx]
                values = [float(val) for val in line.split()]
                if len(values) < 16:
                    raise ValueError(f"traj.txt 第 {idx} 行需要至少 16 个数")
                transformation_matrix = np.array(values[:16]).reshape((4, 4))
                return transformation_matrix
        raise IndexError(f"pose 索引越界: idx={idx}, lines={len(lines)}")

    def _load_depth_intrinsics(self, path, root_dir: str):
        """
        Load the depth camera intrinsics from the given path.

        Args:
            path: Path to the depth camera intrinsics file.
            root_dir: 数据集根目录，用于在缺少 camera.scale 时读取 depth_uint16_write_scale.txt

        Returns:
            (K 3x3, scale, 标定宽, 标定高)；宽/高可能为 None。
        """
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            camera_params = data.get("camera")
            if camera_params:
                fx = camera_params.get("fx")
                fy = camera_params.get("fy")
                cx = camera_params.get("cx")
                cy = camera_params.get("cy")
                scale = camera_params.get("scale")
                if scale is None:
                    scale = _read_depth_uint16_write_scale(root_dir)
                    logger.info(
                        f"camera.scale 缺失，已用 depth_uint16_write_scale.txt -> {scale}"
                    )
                else:
                    scale = float(scale)
                w0 = camera_params.get("w")
                h0 = camera_params.get("h")
                w0_i = int(w0) if w0 is not None else None
                h0_i = int(h0) if h0 is not None else None
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                return K, float(scale), w0_i, h0_i
        raise ValueError("Camera parameters not found in cam_params.json")

    def create_pcd(self, rgb, depth, camera_pose=None):
        """
        Create a point cloud from RGB-D images.

        Args:
            rgb: RGB image as a numpy array.
            depth: Depth image as a numpy array.
            camera_pose: Camera pose as a numpy array (4x4 matrix).

        Returns:
            Point cloud as an Open3D object.
        """
        rgb = np.array(rgb)
        depth = np.array(depth)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        H, W = rgb.shape[0], rgb.shape[1]
        if depth.shape[0] != H or depth.shape[1] != W:
            raise ValueError(
                f"RGB 与深度尺寸不一致: rgb {H}x{W}, depth {depth.shape[0]}x{depth.shape[1]}"
            )
        camera_matrix = self._camera_matrix_for_rgb_shape(W, H)
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        depth = depth.astype(np.float32) / float(self.depth_scale)
        mask = np.logical_and(depth > self.depth_min, depth < self.depth_max)
        x = x[mask]
        y = y[mask]
        depth = depth[mask]
        # convert to 3D
        pixels = np.stack([x, y, np.ones_like(x)], axis=0)
        K_inv = np.linalg.inv(camera_matrix)
        xyz = K_inv @ pixels * depth
        X, Y, Z = xyz[0], xyz[1], xyz[2]
        points = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)))
        n_pts = int(points.shape[0])
        if n_pts == 0:
            return o3d.geometry.PointCloud()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
        colors = rgb[mask]
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        if camera_pose is not None:
            pcd.transform(camera_pose)
        return pcd

    def project_2d_mask_to_3d(
        self,
        mask_2d: np.ndarray,
        depth_image: np.ndarray,
        rgb_image: np.ndarray,
        camera_pose: np.ndarray,
    ) -> o3d.geometry.PointCloud:
        """
        Project 2D mask to 3D point cloud using dataset's camera parameters with RGB colors.

        Args:
            mask_2d: 2D binary mask
            depth_image: Depth image
            rgb_image: RGB image for color information
            camera_pose: Camera pose (4x4 transformation matrix)

        Returns:
            Open3D point cloud object with RGB colors
        """
        H, W = mask_2d.shape
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        # Apply mask
        mask_indices = mask_2d > 0
        x_masked = x[mask_indices]
        y_masked = y[mask_indices]

        # Get depth values
        depth_u = np.asarray(depth_image)
        if depth_u.ndim == 3:
            depth_u = depth_u[:, :, 0]
        depth = depth_u.astype(np.float32) / float(self.depth_scale)
        depth_masked = depth[mask_indices]

        # Get RGB values for masked pixels
        rgb_masked = rgb_image[mask_indices]

        # Filter by depth range
        valid_depth = np.logical_and(
            depth_masked > self.depth_min, depth_masked < self.depth_max
        )
        x_valid = x_masked[valid_depth]
        y_valid = y_masked[valid_depth]
        depth_valid = depth_masked[valid_depth]
        rgb_valid = rgb_masked[valid_depth]

        if len(x_valid) == 0:
            # Return empty point cloud
            return o3d.geometry.PointCloud()

        iw, ih = rgb_image.shape[1], rgb_image.shape[0]
        camera_matrix = self._camera_matrix_for_rgb_shape(iw, ih)

        # Convert to 3D coordinates using dataset's intrinsics
        pixels = np.stack([x_valid, y_valid, np.ones_like(x_valid)], axis=0)
        K_inv = np.linalg.inv(camera_matrix)
        xyz = K_inv @ pixels * depth_valid

        # Transform to world coordinates
        points_cam = np.vstack([xyz, np.ones((1, xyz.shape[1]))])
        points_world = camera_pose @ points_cam

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world[:3, :].T)
        pcd.colors = o3d.utility.Vector3dVector(rgb_valid.astype(np.float64) / 255.0)

        return pcd

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.data_list)


# Example usage:
if __name__ == "__main__":
    cfg = {
        "root_dir": "/home/ubt/workspace/vggt_ws/datasets/merged_from_gs_mesh",
        "depth_scale": 1000.0,
        "depth_min": 0.05,
        "depth_max": 80.0,
    }
    dataset = ReplicaDataset(cfg)
    print(f"Number of samples: {len(dataset)}")

    scene_pcd = o3d.geometry.PointCloud()
    for i in range(0, len(dataset), 5):
        rgb, depth, pose = dataset[i]
        pcd = dataset.create_pcd(rgb, depth, pose)
        scene_pcd += pcd
    o3d.visualization.draw_geometries([scene_pcd], window_name="Replica Scene")

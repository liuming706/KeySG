#!/usr/bin/env python3
import argparse
import json
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


UNKNOWN = 205
FREE = 255
OCCUPIED = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Clean map.png by using the occupied cells closest to image borders, "
            "then regenerate the matching umap.map."
        )
    )
    parser.add_argument("--image", default="map.png", help="input/output PNG path")
    parser.add_argument("--map", default="umap.map", help="input/output umap.map path")
    parser.add_argument(
        "--backup-suffix",
        default=None,
        help="backup suffix; default is .bak_YYYYmmdd_HHMMSS",
    )
    parser.add_argument(
        "--virtual-close-radius",
        type=int,
        default=3,
        help=(
            "radius used to virtually close breaks in the outer occupied contour; "
            "larger values bridge wider gaps"
        ),
    )
    parser.add_argument(
        "--child-contour-dilate-radius",
        type=int,
        default=1,
        help=(
            "radius used to virtually dilate inner occupied contours before finding "
            "unknown regions enclosed by them"
        ),
    )
    parser.add_argument(
        "--child-hole-min-area",
        type=float,
        default=1.0,
        help="minimum enclosed child-contour region size to become unknown",
    )
    parser.add_argument(
        "--keep-largest-free-component",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only the largest free-space component and gray out free fragments",
    )
    return parser.parse_args()


def normalize_pixel(value):
    if value <= 64:
        return OCCUPIED
    if value >= 230:
        return FREE
    return UNKNOWN


def backup(path, suffix):
    backup_path = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if backup_path.exists():
        raise FileExistsError(f"backup already exists: {backup_path}")
    path.rename(backup_path)
    return backup_path


def image_to_array(image):
    normalized = image.convert("L").point(normalize_pixel)
    return np.array(normalized.getdata(), dtype=np.uint8).reshape(
        normalized.size[1], normalized.size[0]
    )


def flood_outside_virtual_contour(occupied, close_radius):
    height, width = occupied.shape
    occupied_u8 = occupied.astype(np.uint8) * 255

    if close_radius > 0:
        kernel_size = close_radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        barrier = cv2.morphologyEx(occupied_u8, cv2.MORPH_CLOSE, kernel) > 0
    else:
        barrier = occupied.copy()

    outside = np.zeros((height, width), dtype=bool)
    queue = deque()

    def push(x, y):
        if barrier[y, x] or outside[y, x]:
            return
        outside[y, x] = True
        queue.append((x, y))

    for x in range(width):
        push(x, 0)
        push(x, height - 1)
    for y in range(height):
        push(0, y)
        push(width - 1, y)

    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < width and 0 <= ny < height:
                push(nx, ny)

    return outside, barrier


def child_black_contour_hole_mask(occupied, interior, dilate_radius, min_area):
    height, width = occupied.shape
    occupied_u8 = occupied.astype(np.uint8) * 255

    if dilate_radius > 0:
        kernel_size = dilate_radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        child_barrier = cv2.dilate(occupied_u8, kernel, iterations=1) > 0
    else:
        child_barrier = occupied.copy()

    traversable = interior & ~child_barrier
    visited = np.zeros((height, width), dtype=bool)
    components = []

    for y in range(height):
        for x in range(width):
            if not traversable[y, x] or visited[y, x]:
                continue

            queue = deque([(x, y)])
            visited[y, x] = True
            cells = []

            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if not traversable[ny, nx] or visited[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((nx, ny))

            components.append(cells)

    hole_mask = np.zeros((height, width), dtype=bool)
    if len(components) <= 1:
        return hole_mask, 0, 0

    main_component = max(components, key=len)
    accepted = 0

    for cells in components:
        if cells is main_component or len(cells) < min_area:
            continue
        accepted += 1
        for x, y in cells:
            hole_mask[y, x] = True

    return hole_mask, accepted, int(hole_mask.sum())


def remove_free_fragments(arr):
    height, width = arr.shape
    free = arr == FREE
    visited = np.zeros((height, width), dtype=bool)
    components = []

    for y in range(height):
        for x in range(width):
            if not free[y, x] or visited[y, x]:
                continue

            queue = deque([(x, y)])
            visited[y, x] = True
            cells = []

            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if not free[ny, nx] or visited[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((nx, ny))

            components.append(cells)

    if len(components) <= 1:
        return 0, 0

    keep = max(components, key=len)
    removed_components = 0
    removed_cells = 0

    for cells in components:
        if cells is keep:
            continue
        removed_components += 1
        removed_cells += len(cells)
        for x, y in cells:
            arr[y, x] = UNKNOWN

    return removed_components, removed_cells


def apply_virtual_contour_cleanup(
    image,
    close_radius,
    child_dilate_radius,
    child_hole_min_area,
    keep_largest_free_component,
):
    arr = image_to_array(image)
    occupied = arr == OCCUPIED

    outside, barrier = flood_outside_virtual_contour(occupied, close_radius)
    interior = ~outside
    child_holes, child_count, child_cells = child_black_contour_hole_mask(
        occupied, interior, child_dilate_radius, child_hole_min_area
    )

    result = arr.copy()
    original_unknown_inside = int(((result == UNKNOWN) & interior).sum())
    outside_known = int(((result != UNKNOWN) & outside).sum())

    result[outside] = UNKNOWN
    result[(result == UNKNOWN) & interior] = FREE
    result[child_holes] = UNKNOWN
    result[occupied] = OCCUPIED
    if keep_largest_free_component:
        free_fragments, free_fragment_cells = remove_free_fragments(result)
    else:
        free_fragments, free_fragment_cells = 0, 0

    stats = {
        "virtual_barrier_cells": int(barrier.sum()),
        "outside_cells": int(outside.sum()),
        "outside_known_cells_set_unknown": outside_known,
        "interior_unknown_cells_set_free": original_unknown_inside,
        "child_contour_dilate_radius": child_dilate_radius,
        "child_black_contours": child_count,
        "child_contour_cells_set_unknown": child_cells,
        "free_fragments_set_unknown": free_fragments,
        "free_fragment_cells_set_unknown": free_fragment_cells,
    }
    return Image.fromarray(result, mode="L"), stats


def load_map_metadata(map_path, width, height):
    if not map_path.exists():
        return {"meta_data": {"width": width, "height": height}, "map_data": []}
    with map_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("meta_data", {})
    data["meta_data"]["width"] = width
    data["meta_data"]["height"] = height
    data["meta_data"]["modify_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    return data


def pack_map_data(image):
    width, height = image.size
    pixels = image.load()
    rows = []

    for y in range(height):
        row = []
        for block_start in range(0, width, 16):
            word = 0
            for offset in range(16):
                x = block_start + offset
                if x >= width:
                    continue
                value = pixels[x, y]
                if value == OCCUPIED:
                    code = 0
                elif value == FREE:
                    code = 1
                else:
                    code = 2
                word |= code << (2 * offset)
            if word >= 2**31:
                word -= 2**32
            row.append(word)
        rows.append(row)

    return rows


def main():
    args = parse_args()
    image_path = Path(args.image)
    map_path = Path(args.map)

    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not map_path.exists():
        raise FileNotFoundError(map_path)

    suffix = args.backup_suffix or datetime.now().strftime(".bak_%Y%m%d_%H%M%S")

    image = Image.open(image_path).convert("L")
    width, height = image.size

    cleaned, stats = apply_virtual_contour_cleanup(
        image,
        args.virtual_close_radius,
        args.child_contour_dilate_radius,
        args.child_hole_min_area,
        args.keep_largest_free_component,
    )

    data = load_map_metadata(map_path, width, height)
    data["map_data"] = pack_map_data(cleaned)

    image_backup = backup(image_path, suffix)
    map_backup = backup(map_path, suffix)

    cleaned.save(image_path)
    with map_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=3)
        f.write("\n")

    print(f"backed up image: {image_backup}")
    print(f"backed up map:   {map_backup}")
    print(f"wrote image:     {image_path}")
    print(f"wrote map:       {map_path}")
    print(f"virtual close radius: {args.virtual_close_radius}")
    print(f"virtual barrier cells: {stats['virtual_barrier_cells']}")
    print(f"child contour dilate radius: {stats['child_contour_dilate_radius']}")
    print(f"outside cells set unknown: {stats['outside_cells']}")
    print(
        "known outside cells changed to unknown: "
        f"{stats['outside_known_cells_set_unknown']}"
    )
    print(
        "interior unknown cells changed to free: "
        f"{stats['interior_unknown_cells_set_free']}"
    )
    print(f"child black contour regions: {stats['child_black_contours']}")
    print(
        "child contour cells set unknown: "
        f"{stats['child_contour_cells_set_unknown']}"
    )
    print(f"free fragments set unknown: {stats['free_fragments_set_unknown']}")
    print(f"free fragment cells set unknown: {stats['free_fragment_cells_set_unknown']}")


if __name__ == "__main__":
    main()

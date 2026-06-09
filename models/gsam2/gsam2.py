import sys
import os
import cv2
import json
import re
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Union, Tuple
from supervision.draw.color import ColorPalette
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Make project root importable before local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.llm.gpt_vlm import GPT_VLMInterface as VLMInterface
from keysg.utils.img_utils import mask_subtract_contained
from keysg.utils.logging_setup import logger

COLOR_PALETTES = {
    "default": None,  # Use supervision default
    "bright": ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"],
    "pastel": ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#FFDFBA", "#E0BBE4"],
    "dark": ["#8B0000", "#006400", "#000080", "#8B8000", "#8B008B", "#008B8B"],
    "high_contrast": ["#FF0000", "#FFFFFF", "#000000", "#FFFF00", "#00FF00", "#0000FF"],
}


def get_color_palette(palette_name="default"):
    """Get a predefined color palette."""
    if palette_name not in COLOR_PALETTES:
        raise ValueError(
            f"Unknown palette: {palette_name}. Available: {list(COLOR_PALETTES.keys())}"
        )
    return COLOR_PALETTES[palette_name]


_LABEL_SPLIT_RE = re.compile(r"\s*(?:\b(?:or|and)\b|/)\s*", re.IGNORECASE)


def _normalize_label_for_matching(label: str) -> str:
    """Normalize detector labels and prompt tags for robust string matching."""
    text = str(label or "").strip().lower()
    text = re.sub(r"[\u2010-\u2015]", "-", text)  # normalize unicode dashes
    text = re.sub(r"\s*-\s*", " ", text)  # ceiling - mounted == ceiling-mounted
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _label_variants_for_matching(label: str) -> List[str]:
    """Return normalized variants, including alternatives from tags like 'A or B'."""
    raw = str(label or "").strip()
    variants = [_normalize_label_for_matching(raw)]
    for part in _LABEL_SPLIT_RE.split(raw):
        normalized = _normalize_label_for_matching(part)
        if normalized:
            variants.append(normalized)

    # Preserve order while dropping duplicates/empty strings.
    seen = set()
    unique_variants = []
    for variant in variants:
        if variant and variant not in seen:
            unique_variants.append(variant)
            seen.add(variant)
    return unique_variants


def _token_overlap_score(lhs: str, rhs: str) -> float:
    """Score token-level containment/overlap between two normalized labels."""
    lhs_tokens = set(lhs.split())
    rhs_tokens = set(rhs.split())
    if not lhs_tokens or not rhs_tokens:
        return 0.0

    intersection = lhs_tokens & rhs_tokens
    containment = len(intersection) / max(1, min(len(lhs_tokens), len(rhs_tokens)))
    jaccard = len(intersection) / len(lhs_tokens | rhs_tokens)
    return max(containment, jaccard)


def _resolve_label_to_prompt_tag(
    label: str,
    prompt_tags: List[str],
    min_score: float = 0.78,
) -> Tuple[Optional[str], Optional[int], str, float]:
    """Map an LLMDet string label back to the closest original prompt tag.

    Matching is intentionally tolerant because LLMDet may normalize labels by
    changing case, rewriting hyphen spacing, or returning a shorter phrase from a
    longer prompt tag such as "blue bag" for "blue bag or utility bag".

    Returns:
        (matched_tag, matched_index, match_reason, score). If no reliable match
        is found, matched_tag and matched_index are None.
    """
    if not str(label or "").strip():
        return None, None, "empty", 0.0

    # Fast exact match keeps historical behavior for already-clean labels.
    prompt_tag_to_id = {tag: idx for idx, tag in enumerate(prompt_tags)}
    if label in prompt_tag_to_id:
        return label, prompt_tag_to_id[label], "exact", 1.0

    label_variants = _label_variants_for_matching(label)
    tag_variants_by_idx = [_label_variants_for_matching(tag) for tag in prompt_tags]

    # Normalized exact / alternative exact match.
    for tag_idx, tag_variants in enumerate(tag_variants_by_idx):
        for label_variant in label_variants:
            if label_variant in tag_variants:
                return prompt_tags[tag_idx], tag_idx, "normalized", 1.0

    best_idx: Optional[int] = None
    best_score = 0.0
    best_reason = "unmatched"

    for tag_idx, tag_variants in enumerate(tag_variants_by_idx):
        for label_variant in label_variants:
            for tag_variant in tag_variants:
                if not label_variant or not tag_variant:
                    continue

                # Strongly prefer phrase containment over generic edit distance.
                if label_variant in tag_variant or tag_variant in label_variant:
                    score = 0.95
                    reason = "substring"
                else:
                    token_score = _token_overlap_score(label_variant, tag_variant)
                    edit_score = SequenceMatcher(
                        None, label_variant, tag_variant
                    ).ratio()
                    score = max(token_score, edit_score)
                    reason = "token_overlap" if token_score >= edit_score else "fuzzy"

                if score > best_score:
                    best_idx = tag_idx
                    best_score = score
                    best_reason = reason

    if best_idx is not None and best_score >= min_score:
        return prompt_tags[best_idx], best_idx, best_reason, best_score

    return None, None, best_reason, best_score


class GroundingSAM2:
    """SAM2 + LLMDet detection backend for object detection and segmentation."""

    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_model_config: str,
        llmdet_model_id: str,
        vlm_model: str = "deepseek-v4-flash",
        device: Optional[str] = None,
        force_cpu: bool = False,
        llmdet_max_tags_per_batch: int = 80,  # 30 会爆 8GB 显存
    ):
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model_config = sam2_model_config
        self.llmdet_model_id = llmdet_model_id or "iSEE-Laboratory/llmdet_large"
        self.llmdet_max_tags_per_batch = llmdet_max_tags_per_batch
        self.vlm_model = vlm_model
        print(f"vlm_model: {vlm_model}")
        self._vlm_client = None

        if device is not None:
            self.device = device
        else:
            self.device = (
                "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
            )
        print(f"GroundingSAM2 device: {self.device}")

        self._setup_environment()
        self._load_sam2()
        self._load_detection_model()

    def _setup_environment(self):
        if torch.cuda.is_available():
            try:
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

    def _load_sam2(self):
        sam2_model = build_sam2(
            self.sam2_model_config, self.sam2_checkpoint, device=self.device
        )
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

    def _load_detection_model(self):
        self.llmdet_processor = AutoProcessor.from_pretrained(self.llmdet_model_id)
        self.llmdet_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.llmdet_model_id
        ).to(self.device)

    def _ensure_vlm_client(self):
        if self._vlm_client is None:
            self._vlm_client = VLMInterface(self.vlm_model)

    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image],
        text_prompt: str,
        box_threshold: float = 0.2,
        multimask_output: bool = False,
    ) -> Dict[str, Any]:
        """
        Detect and segment objects using LLMDet + SAM2.

        Args:
            image: Input image (file path, numpy array, or PIL Image)
            text_prompt: Dot-separated object labels, e.g. "chair. table. door."
            box_threshold: Confidence threshold for bounding boxes
            multimask_output: Whether to return multiple masks per object

        Returns:
            dict with keys: 'boxes', 'masks', 'scores', 'labels', 'class_ids', 'image_size'
        """
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("Image must be a file path, numpy array, or PIL Image")

        self.sam2_predictor.set_image(np.array(pil_image.convert("RGB")))
        return self._predict_llmdet(
            pil_image, text_prompt, box_threshold, multimask_output
        )

    def _predict_llmdet(
        self,
        pil_image: Image.Image,
        text_prompt: str,
        box_threshold: float,
        multimask_output: bool,
    ) -> Dict[str, Any]:
        """Run LLMDet + SAM2 with automatic batching for large tag lists."""
        raw_prompt_parts = text_prompt.split(".")
        empty_prompt_indices = [
            idx for idx, part in enumerate(raw_prompt_parts) if not part.strip()
        ]
        if empty_prompt_indices:
            logger.warning(
                "LLMDet prompt contains empty tag segments: indices={}, raw_prompt={!r}",
                empty_prompt_indices,
                text_prompt,
            )

        tags = [p.strip() for p in raw_prompt_parts if p.strip()]
        abnormal_tags = [
            tag
            for tag in tags
            if "\n" in tag or "\r" in tag or len(tag) > 80 or tag in {".", "|", ","}
        ]
        if abnormal_tags:
            logger.warning(
                "LLMDet prompt contains abnormal tags that may not map cleanly: {}",
                abnormal_tags,
            )
        if not tags:
            logger.warning(
                "LLMDet prompt produced no valid tags: raw_prompt={!r}", text_prompt
            )

        if len(tags) <= self.llmdet_max_tags_per_batch:
            return self._predict_llmdet_batch(
                pil_image, [tags], box_threshold, multimask_output
            )

        all_boxes, all_masks, all_scores, all_labels, all_class_ids = [], [], [], [], []

        for i in range(0, len(tags), self.llmdet_max_tags_per_batch):
            batch_tags = tags[i : i + self.llmdet_max_tags_per_batch]
            batch_results = self._predict_llmdet_batch(
                pil_image, [batch_tags], box_threshold, multimask_output
            )

            if len(batch_results["boxes"]) > 0:
                all_boxes.append(batch_results["boxes"])
                all_masks.append(batch_results["masks"])
                all_scores.append(batch_results["scores"])
                all_labels.extend(batch_results["labels"])
                base_id = len(all_class_ids)
                adjusted_class_ids = batch_results["class_ids"] + base_id
                all_class_ids.extend(adjusted_class_ids)

        if not all_boxes:
            return {
                "boxes": np.array([]).reshape(0, 4),
                "masks": np.array([]).reshape(0, pil_image.height, pil_image.width),
                "scores": np.array([]),
                "labels": [],
                "class_ids": np.array([]),
                "image_size": pil_image.size,
            }

        merged_boxes = np.vstack(all_boxes)
        merged_masks = np.vstack(all_masks)
        merged_scores = np.hstack(all_scores)
        merged_class_ids = np.array(all_class_ids)

        final_indices = self._resolve_mask_overlaps(merged_masks, merged_scores)

        return {
            "boxes": merged_boxes[final_indices],
            "masks": merged_masks[final_indices].astype(bool),
            "scores": merged_scores[final_indices],
            "labels": [all_labels[i] for i in final_indices],
            "class_ids": merged_class_ids[final_indices],
            "image_size": pil_image.size,
        }

    def _predict_llmdet_batch(
        self,
        pil_image: Image.Image,
        texts: List[List[str]],
        box_threshold: float,
        multimask_output: bool,
    ) -> Dict[str, Any]:
        """Process a single batch of tags with LLMDet."""
        inputs = self.llmdet_processor(
            images=pil_image,
            text=texts,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.llmdet_model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)

        llm_results = self.llmdet_processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=box_threshold, target_sizes=target_sizes
        )

        if not llm_results or len(llm_results[0]["boxes"]) == 0:
            return {
                "boxes": np.array([]).reshape(0, 4),
                "masks": np.array([]).reshape(0, pil_image.height, pil_image.width),
                "scores": np.array([]),
                "labels": [],
                "class_ids": np.array([]),
                "image_size": pil_image.size,
            }

        input_boxes = llm_results[0]["boxes"].cpu().numpy()
        confidences = llm_results[0]["scores"].cpu().numpy()

        result0 = llm_results[0]
        raw_labels = result0.get("labels")
        raw_text_labels = result0.get("text_labels")

        class_names = []
        class_ids = []
        logger.info("raw_labels type {}", type(raw_labels))
        if hasattr(raw_labels, "cpu"):
            # Older Transformers versions returned integer ids in labels.
            label_indices = raw_labels.cpu().numpy()
            for det_idx, label_index in enumerate(label_indices):
                class_id = int(label_index)
                if 0 <= class_id < len(texts[0]):
                    class_name = texts[0][class_id]
                    if not str(class_name).strip():
                        logger.warning(
                            "LLMDet label mapped to an empty prompt tag: det_idx={}, label_index={}, prompt_tags={}",
                            det_idx,
                            class_id,
                            texts[0],
                        )
                    class_names.append(class_name)
                    class_ids.append(class_id)
                else:
                    logger.warning(
                        "LLMDet returned a label index that cannot map to prompt tags: det_idx={}, label_index={}, num_prompt_tags={}, prompt_tags={}",
                        det_idx,
                        class_id,
                        len(texts[0]),
                        texts[0],
                    )
                    class_names.append("")
                    class_ids.append(class_id)
            class_ids = np.array(class_ids, dtype=int)
        elif raw_text_labels is not None:
            # Transformers >= 4.51 recommends text_labels for string object names.
            label_names = [str(label) for label in raw_text_labels]
            logger.debug("Using LLMDet text_labels for class names: {}", label_names)
            for det_idx, class_name in enumerate(label_names):
                stripped_name = class_name.strip()
                if not stripped_name:
                    logger.warning(
                        "LLMDet returned an empty text label: det_idx={}, raw_text_labels={}, prompt_tags={}",
                        det_idx,
                        raw_text_labels,
                        texts[0],
                    )
                    class_names.append("")
                    class_ids.append(len(texts[0]) + det_idx)
                    continue

                matched_tag, matched_id, match_reason, match_score = (
                    _resolve_label_to_prompt_tag(
                        stripped_name,
                        texts[0],
                    )
                )
                if matched_id is None:
                    logger.warning(
                        "LLMDet text label could not be matched to any prompt tag: det_idx={}, text_label={!r}, best_reason={}, best_score={:.3f}, prompt_tags={}",
                        det_idx,
                        stripped_name,
                        match_reason,
                        match_score,
                        texts[0],
                    )
                    class_names.append(stripped_name)
                    class_ids.append(len(texts[0]) + det_idx)
                else:
                    if match_reason != "exact":
                        logger.debug(
                            "Matched LLMDet text label to prompt tag: det_idx={}, text_label={!r}, prompt_tag={!r}, match_reason={}, match_score={:.3f}",
                            det_idx,
                            stripped_name,
                            matched_tag,
                            match_reason,
                            match_score,
                        )
                    class_names.append(stripped_name)
                    class_ids.append(matched_id)

            class_ids = np.array(class_ids, dtype=int)
        else:
            # Current Transformers versions may still expose labels as string names,
            # while warning that labels will become integer ids in future versions.
            label_names = [str(label) for label in (raw_labels or [])]
            logger.debug(
                "Using string labels from LLMDet labels field because text_labels is unavailable: {}",
                label_names,
            )
            for det_idx, class_name in enumerate(label_names):
                stripped_name = class_name.strip()
                if not stripped_name:
                    logger.warning(
                        "LLMDet returned an empty string label: det_idx={}, raw_labels={}, prompt_tags={}",
                        det_idx,
                        raw_labels,
                        texts[0],
                    )
                    class_names.append("")
                    class_ids.append(len(texts[0]) + det_idx)
                    continue

                matched_tag, matched_id, match_reason, match_score = (
                    _resolve_label_to_prompt_tag(
                        stripped_name,
                        texts[0],
                    )
                )
                if matched_id is None:
                    logger.warning(
                        "LLMDet string label could not be matched to any prompt tag: det_idx={}, label={!r}, best_reason={}, best_score={:.3f}, prompt_tags={}",
                        det_idx,
                        stripped_name,
                        match_reason,
                        match_score,
                        texts[0],
                    )
                    class_names.append(stripped_name)
                    class_ids.append(len(texts[0]) + det_idx)
                else:
                    if match_reason != "exact":
                        logger.debug(
                            "Matched LLMDet string label to prompt tag: det_idx={}, label={!r}, prompt_tag={!r}, match_reason={}, match_score={:.3f}",
                            det_idx,
                            stripped_name,
                            matched_tag,
                            match_reason,
                            match_score,
                        )
                    class_names.append(stripped_name)
                    class_ids.append(matched_id)

            class_ids = np.array(class_ids, dtype=int)

        masks, sam_scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=multimask_output,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        masks = mask_subtract_contained(input_boxes, masks.astype(bool))

        return {
            "boxes": input_boxes,
            "masks": masks.astype(bool),
            "scores": confidences,
            "labels": class_names,
            "class_ids": class_ids,
            "image_size": pil_image.size,
        }

    def _resolve_mask_overlaps(
        self, masks: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5
    ) -> np.ndarray:
        """Keep highest-confidence masks, suppressing overlapping ones above iou_threshold."""
        n_masks = len(masks)
        if n_masks <= 1:
            return np.arange(n_masks)

        intersections = np.zeros((n_masks, n_masks))
        unions = np.zeros((n_masks, n_masks))

        for i in range(n_masks):
            for j in range(i, n_masks):
                intersection = np.logical_and(masks[i], masks[j]).sum()
                union = np.logical_or(masks[i], masks[j]).sum()
                intersections[i, j] = intersections[j, i] = intersection
                unions[i, j] = unions[j, i] = union

        ious = intersections / (unions + 1e-6)
        keep = np.ones(n_masks, dtype=bool)
        sorted_indices = np.argsort(scores)[::-1]

        for i, idx_i in enumerate(sorted_indices):
            if not keep[idx_i]:
                continue
            for j, idx_j in enumerate(sorted_indices[i + 1 :], i + 1):
                if keep[idx_j] and ious[idx_i, idx_j] > iou_threshold:
                    keep[idx_j] = False

        return np.where(keep)[0]

    def tag_image(self, image: Union[str, np.ndarray, Image.Image]) -> str:
        """
        Generate image tags using the VLM client.

        Returns:
            Tags as a single string in the format "tag1 | tag2 | tag3"
        """
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("Image must be a file path, numpy array, or PIL Image")

        self._ensure_vlm_client()
        tags = self._vlm_client.tag_objects_in_image(pil_image)
        empty_tag_indices = [
            idx for idx, tag in enumerate(tags) if not str(tag).strip()
        ]
        abnormal_tags = [
            tag
            for tag in tags
            if not isinstance(tag, str)
            or "\n" in str(tag)
            or "\r" in str(tag)
            or "|" in str(tag)
            or "." in str(tag)
            or len(str(tag).strip()) > 80
        ]
        if empty_tag_indices:
            logger.warning(
                "VLM tag generation returned empty tags: indices={}, raw_tags={}",
                empty_tag_indices,
                tags,
            )
        if abnormal_tags:
            logger.warning(
                "VLM tag generation returned abnormal tags; these may create invalid LLMDet prompts: {}",
                abnormal_tags,
            )
        return " | ".join(str(tag) for tag in tags)

    def ram_tags_to_prompt(self, ram_tags: str) -> str:
        """Convert "tag1 | tag2 | tag3" to "tag1. tag2. tag3." for use as text_prompt."""
        raw_tags = ram_tags.split("|")
        empty_tag_indices = [idx for idx, tag in enumerate(raw_tags) if not tag.strip()]
        if empty_tag_indices:
            logger.warning(
                "RAM/VLM tag string contains empty tag segments: indices={}, raw_tags={!r}",
                empty_tag_indices,
                ram_tags,
            )

        tags = [tag.strip() for tag in raw_tags if tag.strip()]
        abnormal_tags = [
            tag
            for tag in tags
            if "\n" in tag or "\r" in tag or "." in tag or len(tag) > 80
        ]
        if abnormal_tags:
            logger.warning(
                "RAM/VLM tag string contains abnormal tags before prompt conversion: {}",
                abnormal_tags,
            )
        return " ".join(f"{tag}." for tag in tags)

    def visualize_results(
        self,
        results: Dict[str, Any],
        image: Union[str, np.ndarray],
        visualize: bool = True,
        output_path: Optional[str] = None,
        show_boxes: bool = True,
        show_masks: bool = True,
        show_labels: bool = True,
        custom_color_map: Optional[List[str]] = None,
        apply_nms: bool = True,
        nms_threshold: float = 0.5,
    ) -> np.ndarray:
        """Visualize detection and segmentation results."""
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Could not read image from path: {image}")
        else:
            img = image.copy()

        if len(results["boxes"]) == 0:
            if output_path:
                cv2.imwrite(output_path, img)
            return img

        if apply_nms:
            boxes = torch.tensor(results["boxes"], dtype=torch.float32)
            scores = torch.tensor(results["scores"], dtype=torch.float32)
            keep_indices = torch.ops.torchvision.nms(boxes, scores, nms_threshold)
            results["boxes"] = boxes[keep_indices].numpy()
            results["masks"] = results["masks"][keep_indices.numpy()]
            results["scores"] = scores[keep_indices].numpy()
            results["labels"] = [results["labels"][i] for i in keep_indices.numpy()]
            results["class_ids"] = results["class_ids"][keep_indices.numpy()]

        detections = sv.Detections(
            xyxy=results["boxes"], mask=results["masks"], class_id=results["class_ids"]
        )

        color_palette = (
            ColorPalette.from_hex(custom_color_map)
            if custom_color_map
            else ColorPalette.DEFAULT
        )
        annotated_frame = img.copy()

        if show_masks:
            mask_annotator = sv.MaskAnnotator(color=color_palette)
            annotated_frame = mask_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

        if show_boxes:
            box_annotator = sv.BoxAnnotator(color=color_palette)
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

        if show_labels:
            empty_label_indices = [
                idx
                for idx, class_name in enumerate(results["labels"])
                if not str(class_name).strip()
            ]
            if empty_label_indices:
                logger.warning(
                    "Visualization received empty detection labels; these would appear as score-only without fallback: indices={}, labels={}, scores={}",
                    empty_label_indices,
                    results["labels"],
                    (
                        results["scores"].tolist()
                        if hasattr(results["scores"], "tolist")
                        else results["scores"]
                    ),
                )
            labels = [
                f"{str(class_name).strip() or 'unknown'} {confidence:.2f}"
                for class_name, confidence in zip(results["labels"], results["scores"])
            ]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            padding = 4
            bg_alpha = 0.55
            text_color = (255, 255, 255)

            for box, label in zip(results["boxes"], labels):
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                x1 = max(0, min(x1, annotated_frame.shape[1] - 1))
                y1 = max(0, min(y1, annotated_frame.shape[0] - 1))
                x2 = max(0, min(x2, annotated_frame.shape[1] - 1))
                y2 = max(0, min(y2, annotated_frame.shape[0] - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                (text_w, text_h), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                rect_x1, rect_y1 = x1 + 1, y1 + 1
                rect_x2 = min(x2, rect_x1 + text_w + padding * 2)
                rect_y2 = min(y2, rect_y1 + text_h + baseline + padding * 2)
                if rect_x2 <= rect_x1 or rect_y2 <= rect_y1:
                    continue

                roi = annotated_frame[rect_y1:rect_y2, rect_x1:rect_x2]
                overlay = roi.copy()
                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (rect_x2 - rect_x1, rect_y2 - rect_y1),
                    (0, 0, 0),
                    -1,
                )
                cv2.addWeighted(overlay, bg_alpha, roi, 1 - bg_alpha, 0, dst=roi)

                text_x = rect_x1 + padding
                text_y = min(rect_y2 - padding - baseline, rect_y1 + padding + text_h)
                cv2.putText(
                    annotated_frame,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                    cv2.LINE_AA,
                )

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, annotated_frame)
        elif visualize:
            cv2.imshow("Annotated Image", annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotated_frame

    def save_results_json(
        self, results: Dict[str, Any], image_path: str, output_path: str
    ):
        """Save detection results to JSON with RLE-encoded masks."""

        def single_mask_to_rle(mask):
            rle = mask_util.encode(
                np.array(mask[:, :, None], order="F", dtype="uint8")
            )[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        mask_rles = [single_mask_to_rle(mask) for mask in results["masks"]]

        json_results = {
            "image_path": image_path,
            "annotations": [
                {
                    "class_name": class_name,
                    "bbox": box.tolist(),
                    "segmentation": mask_rle,
                    "score": float(score),
                }
                for class_name, box, mask_rle, score in zip(
                    results["labels"], results["boxes"], mask_rles, results["scores"]
                )
            ],
            "box_format": "xyxy",
            "img_width": results["image_size"][0],
            "img_height": results["image_size"][1],
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=4)


# Example usage
if __name__ == "__main__":
    img_path = "/home/werby/SIRData/nico/raw/rosbag2_2026_03_26-07_39_28/rgb/rgb_image_1774510768749777920.jpg"
    # img_path = "/mnt/ssd2/datasets/SIR/Demo/fullab_1fps_da3/color/frame_0049.jpg"

    gsam = GroundingSAM2(
        sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
        sam2_model_config="./configs/sam2.1/sam2.1_hiera_l.yaml",
        llmdet_model_id="iSEE-Laboratory/llmdet_large",
    )

    eng_tags = gsam.tag_image(img_path)
    print(f"Tags: {eng_tags}")

    text_prompt = gsam.ram_tags_to_prompt(eng_tags)
    results = gsam.predict(
        image=img_path,
        text_prompt=text_prompt,
        box_threshold=0.3,
    )
    gsam.visualize_results(
        results=results,
        image=img_path,
        output_path="output/annotated_image_llmdet.jpg",
    )
    print(f"Found {len(results['labels'])} objects:")
    for label in results["labels"]:
        print(f" - {label}")

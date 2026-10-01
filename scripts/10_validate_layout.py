from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

from glmocr.config import load_config
from glmocr.pipeline import Pipeline

from paddlex_layout_detector import (
    PaddleXPPDocLayoutDetector,
)


# ============================================================================
# Paths
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = ROOT / "our_glm.yaml"

DEFAULT_IMAGE = (
    ROOT
    / "page1-test1"
    / "images"
    / "79e111f2-image_1.png"
)

DEFAULT_GROUND_TRUTH = (
    ROOT
    / "page1-test1"
    / "annotations"
    / "instance_val.json"
)

DEFAULT_EXPECTED = ROOT / "validation_expected.json"

DEFAULT_OUTPUT = ROOT / "output" / "glmocr_validation"


# ============================================================================
# Native PP-DocLayoutV3 label -> GLM-OCR task mapping
# ============================================================================

NATIVE_TO_TASK = {
    "abstract": "text",
    "algorithm": "text",
    "aside_text": None,
    "chart": None,
    "content": "text",
    "display_formula": "formula",
    "doc_title": "text",
    "figure_title": "text",
    "footer": None,
    "footer_image": None,
    "footnote": None,
    "formula_number": "text",
    "header": None,
    "header_image": None,
    "image": None,
    "inline_formula": "formula",
    "number": None,
    "paragraph_title": "text",
    "reference": None,
    "reference_content": "text",
    "seal": "text",
    "table": "table",
    "text": "text",
    "vertical_text": "text",
    "vision_footnote": "text",
}


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete custom GLM-OCR pipeline and validate "
            "layout predictions and OCR output against ground truth."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the GLM-OCR configuration YAML file.",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "PaddleX PP-DocLayoutV3 dataset root. When provided, "
            "--image and --ground-truth are ignored. "
            "Uses annotations/instance_val.json when available, "
            "otherwise annotations/instance_train.json."
        ),
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=(
            "Path to the input image. Ignored when --dataset is used."
        ),
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help=(
            "Ground-truth annotation file for single-image mode. "
            "Ignored when --dataset is used."
        ),
    )

    parser.add_argument(
        "--expected",
        type=Path,
        default=None,
        help=(
            "JSON file containing OCR values that must/must not appear. "
            "Defaults to validation_expected.json when it exists."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where validation results are saved.",
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="IoU threshold required for a prediction/GT box match.",
    )

    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.50,
        help="Minimum layout F1 required for overall PASS.",
    )

    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.50,
        help="Minimum mean matched IoU required for overall PASS.",
    )

    parser.add_argument(
        "--min-label-accuracy",
        type=float,
        default=0.50,
        help="Minimum matched-label accuracy required for overall PASS.",
    )

    return parser.parse_args()


# ============================================================================
# Path helpers
# ============================================================================

def resolve_path(path: Path) -> Path:
    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


# ============================================================================
# Text helpers
# ============================================================================

def normalize_text(value) -> str:
    if value is None:
        return ""

    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_numeric(value) -> str:
    return re.sub(
        r"[^0-9]",
        "",
        normalize_text(value),
    )


# ============================================================================
# Bounding boxes
# ============================================================================

def xywh_to_xyxy(box):
    x, y, w, h = map(float, box)

    return [
        x,
        y,
        x + w,
        y + h,
    ]


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def glm_bbox_to_pixels(
    bbox,
    image_width: int,
    image_height: int,
):
    """
    GLM-OCR bbox_2d is in 0..1000 normalized coordinates.
    """

    x1, y1, x2, y2 = map(float, bbox)

    return [
        x1 * image_width / 1000.0,
        y1 * image_height / 1000.0,
        x2 * image_width / 1000.0,
        y2 * image_height / 1000.0,
    ]


# ============================================================================
# COCO / PaddleX dataset loading
# ============================================================================

def resolve_dataset_image(
    dataset_dir: Path,
    file_name: str,
) -> Path:
    file_path = Path(file_name)

    candidates = [
        dataset_dir / file_path,
        dataset_dir / "images" / file_path.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Image referenced by COCO annotation does not exist: "
        f"{file_name}\n"
        f"Checked:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_coco_dataset(
    annotation_path: Path,
    dataset_dir: Path,
):
    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    categories = {
        int(category["id"]): category["name"]
        for category in data.get("categories", [])
    }

    images = data.get("images", [])

    if not images:
        raise ValueError(
            f"No images found in {annotation_path}"
        )

    image_by_id = {
        image["id"]: image
        for image in images
    }

    gt_by_image_id = {
        image_id: []
        for image_id in image_by_id
    }

    for annotation in data.get(
        "annotations",
        [],
    ):
        image_id = annotation.get("image_id")

        if image_id not in image_by_id:
            continue

        category_name = categories.get(
            int(annotation["category_id"])
        )

        if category_name is None:
            continue

        task_label = NATIVE_TO_TASK.get(
            category_name
        )

        # Skip / abandon labels are not final GLM-OCR regions.
        if task_label is None:
            continue

        bbox = annotation.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        gt_by_image_id[image_id].append(
            {
                "native_label": category_name,
                "label": task_label,
                "bbox": xywh_to_xyxy(bbox),
                "order": annotation.get(
                    "read_order",
                    annotation.get("order"),
                ),
            }
        )

    cases = []

    for image in images:
        image_path = resolve_dataset_image(
            dataset_dir,
            image["file_name"],
        )

        cases.append(
            {
                "image": image_path,
                "image_id": image["id"],
                "file_name": image["file_name"],
                "ground_truth": gt_by_image_id[
                    image["id"]
                ],
            }
        )

    return cases


def load_paddlex_dataset(
    dataset_dir: Path,
):
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    annotations_dir = dataset_dir / "annotations"
    images_dir = dataset_dir / "images"

    if not annotations_dir.exists():
        raise FileNotFoundError(
            f"Dataset annotations directory not found: "
            f"{annotations_dir}"
        )

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Dataset images directory not found: "
            f"{images_dir}"
        )

    val_json = annotations_dir / "instance_val.json"
    train_json = annotations_dir / "instance_train.json"

    if val_json.exists():
        annotation_path = val_json
        split = "val"
    elif train_json.exists():
        annotation_path = train_json
        split = "train"
    else:
        raise FileNotFoundError(
            "Could not find either:\n"
            f"  {val_json}\n"
            f"  {train_json}"
        )

    print(
        f"Validation mode : PaddleX dataset"
    )
    print(
        f"Dataset         : {dataset_dir}"
    )
    print(
        f"Split           : {split}"
    )
    print(
        f"Annotations     : {annotation_path}"
    )

    return load_coco_dataset(
        annotation_path,
        dataset_dir,
    )


def load_single_ground_truth(
    path: Path,
    image_path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        raw = json.load(f)

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    # ------------------------------------------------------------------------
    # COCO
    # ------------------------------------------------------------------------

    if isinstance(raw, dict) and "annotations" in raw:

        return load_coco_ground_truth(
            raw,
            image_path,
            image_width,
            image_height,
        )

    # ------------------------------------------------------------------------
    # Native Label Studio JSON
    # ------------------------------------------------------------------------

    if isinstance(raw, list):

        return load_labelstudio_ground_truth(
            raw,
            image_width,
            image_height,
        )

    raise ValueError(
        f"Unsupported ground-truth format: {path}"
    )


def load_coco_ground_truth(
    data,
    image_path: Path,
    image_width: int,
    image_height: int,
):
    categories = {
        int(category["id"]): category["name"]
        for category in data.get("categories", [])
    }

    images = data.get("images", [])

    if not images:
        raise ValueError(
            "No images found in COCO ground truth."
        )

    target_name = image_path.name

    matching_images = [
        image
        for image in images
        if Path(
            image.get("file_name", "")
        ).name == target_name
    ]

    if len(matching_images) == 1:
        target_image = matching_images[0]
    elif len(images) == 1:
        target_image = images[0]
    else:
        raise ValueError(
            f"Could not uniquely identify image "
            f"{target_name!r} in COCO ground truth."
        )

    source_width = float(
        target_image.get(
            "width",
            image_width,
        )
    )

    source_height = float(
        target_image.get(
            "height",
            image_height,
        )
    )

    scale_x = image_width / source_width
    scale_y = image_height / source_height

    output = []

    for annotation in data.get(
        "annotations",
        [],
    ):
        if annotation.get("image_id") != target_image["id"]:
            continue

        category_name = categories.get(
            int(annotation["category_id"])
        )

        if category_name is None:
            continue

        task_label = NATIVE_TO_TASK.get(
            category_name
        )

        if task_label is None:
            continue

        bbox = xywh_to_xyxy(
            annotation["bbox"]
        )

        bbox = [
            bbox[0] * scale_x,
            bbox[1] * scale_y,
            bbox[2] * scale_x,
            bbox[3] * scale_y,
        ]

        output.append(
            {
                "native_label": category_name,
                "label": task_label,
                "bbox": bbox,
                "order": annotation.get(
                    "read_order",
                    annotation.get("order"),
                ),
            }
        )

    return output


def load_labelstudio_ground_truth(
    tasks,
    image_width: int,
    image_height: int,
):
    if not tasks:
        return []

    task = tasks[0]

    annotations = task.get(
        "annotations",
        [],
    )

    if not annotations:
        raise ValueError(
            "No annotations found in Label Studio task."
        )

    results = annotations[0].get(
        "result",
        [],
    )

    labels_by_region = {}
    orders_by_region = {}

    source_width = None
    source_height = None

    for result in results:

        if source_width is None:
            source_width = result.get(
                "original_width"
            )

        if source_height is None:
            source_height = result.get(
                "original_height"
            )

        region_id = result.get("id")

        if result.get("from_name") == "label":

            labels = (
                result.get("value", {})
                .get("polygonlabels", [])
            )

            if labels:
                labels_by_region[
                    region_id
                ] = labels[0]

        elif result.get("from_name") == "read_order":

            number = (
                result.get("value", {})
                .get("number")
            )

            orders_by_region[
                region_id
            ] = number

    if not source_width or not source_height:
        source_width = image_width
        source_height = image_height

    scale_x = image_width / float(
        source_width
    )

    scale_y = image_height / float(
        source_height
    )

    output = []

    for result in results:

        if result.get(
            "from_name"
        ) != "label":
            continue

        region_id = result.get("id")

        native_label = labels_by_region.get(
            region_id
        )

        if native_label is None:
            continue

        task_label = NATIVE_TO_TASK.get(
            native_label
        )

        if task_label is None:
            continue

        points = (
            result.get("value", {})
            .get("points")
        )

        if not points:
            continue

        pixels = [
            [
                float(point[0])
                * float(source_width)
                / 100.0
                * scale_x,

                float(point[1])
                * float(source_height)
                / 100.0
                * scale_y,
            ]
            for point in points
        ]

        xs = [
            point[0]
            for point in pixels
        ]

        ys = [
            point[1]
            for point in pixels
        ]

        bbox = [
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        ]

        output.append(
            {
                "native_label": native_label,
                "label": task_label,
                "bbox": bbox,
                "order": orders_by_region.get(
                    region_id
                ),
            }
        )

    return output


# ============================================================================
# Predictions
# ============================================================================

def extract_predictions(
    result,
    image_width: int,
    image_height: int,
):
    """
    Extract final GLM-OCR regions from PipelineResult.json_result.
    """

    json_result = result.json_result

    if not isinstance(
        json_result,
        list,
    ):
        raise ValueError(
            "PipelineResult.json_result is not a list."
        )

    output = []

    for page_index, page in enumerate(
        json_result
    ):

        if not isinstance(page, list):
            continue

        for region in page:

            bbox = region.get(
                "bbox_2d"
            )

            if not bbox or len(bbox) != 4:
                continue

            output.append(
                {
                    "page": page_index,
                    "index": region.get(
                        "index"
                    ),
                    "label": region.get(
                        "label"
                    ),
                    "bbox_2d": bbox,
                    "bbox": glm_bbox_to_pixels(
                        bbox,
                        image_width,
                        image_height,
                    ),
                    "content": region.get(
                        "content"
                    ),
                    "native_label": region.get(
                        "native_label"
                    ),
                    "score": region.get(
                        "score"
                    ),
                }
            )

    return output


# ============================================================================
# Box matching
# ============================================================================

def match_boxes(
    gt_boxes,
    predicted_boxes,
    iou_threshold,
):
    if not gt_boxes or not predicted_boxes:
        return []

    iou_matrix = np.zeros(
        (
            len(gt_boxes),
            len(predicted_boxes),
        ),
        dtype=float,
    )

    for gt_index, gt in enumerate(
        gt_boxes
    ):

        for pred_index, pred in enumerate(
            predicted_boxes
        ):

            iou_matrix[
                gt_index,
                pred_index,
            ] = bbox_iou(
                gt["bbox"],
                pred["bbox"],
            )

    row_indices, col_indices = (
        linear_sum_assignment(
            -iou_matrix
        )
    )

    matches = []

    for gt_index, pred_index in zip(
        row_indices,
        col_indices,
    ):

        iou = float(
            iou_matrix[
                gt_index,
                pred_index,
            ]
        )

        if iou < iou_threshold:
            continue

        gt_label = gt_boxes[
            gt_index
        ]["label"]

        pred_label = predicted_boxes[
            pred_index
        ]["label"]

        matches.append(
            {
                "gt_index": int(
                    gt_index
                ),
                "pred_index": int(
                    pred_index
                ),
                "iou": iou,
                "gt_label": gt_label,
                "pred_label": pred_label,
                "label_correct": (
                    gt_label
                    == pred_label
                ),
            }
        )

    return matches


# ============================================================================
# Layout metrics
# ============================================================================

def calculate_layout_metrics(
    gt_boxes,
    predicted_boxes,
    matches,
):
    gt_count = len(gt_boxes)
    pred_count = len(predicted_boxes)

    geometric_matches = len(matches)

    correct_matches = sum(
        1
        for match in matches
        if match["label_correct"]
    )

    true_positives = correct_matches

    false_positives = (
        pred_count
        - true_positives
    )

    false_negatives = (
        gt_count
        - true_positives
    )

    precision = (
        true_positives
        / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0.0
    )

    recall = (
        true_positives
        / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    label_accuracy = (
        correct_matches
        / geometric_matches
        if geometric_matches > 0
        else 0.0
    )

    mean_iou = (
        sum(
            match["iou"]
            for match in matches
        )
        / geometric_matches
        if geometric_matches > 0
        else 0.0
    )

    return {
        "ground_truth_boxes": gt_count,
        "predicted_boxes": pred_count,
        "geometric_matches": geometric_matches,
        "correct_matches": correct_matches,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "label_accuracy": label_accuracy,
        "mean_iou": mean_iou,

        # These are validator-defined metrics,
        # not PP-DocLayoutV3 training losses.
        "bbox_loss": 1.0 - mean_iou,
        "detection_loss": 1.0 - f1,
    }


def calculate_aggregate_metrics(
    per_image_metrics,
):
    gt_count = sum(
        metric["ground_truth_boxes"]
        for metric in per_image_metrics
    )

    pred_count = sum(
        metric["predicted_boxes"]
        for metric in per_image_metrics
    )

    geometric_matches = sum(
        metric["geometric_matches"]
        for metric in per_image_metrics
    )

    correct_matches = sum(
        metric["correct_matches"]
        for metric in per_image_metrics
    )

    tp = correct_matches

    fp = pred_count - tp
    fn = gt_count - tp

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    label_accuracy = (
        correct_matches
        / geometric_matches
        if geometric_matches > 0
        else 0.0
    )

    # Weight mean IoU by number of matches.
    total_iou = sum(
        metric["mean_iou"]
        * metric["geometric_matches"]
        for metric in per_image_metrics
    )

    mean_iou = (
        total_iou / geometric_matches
        if geometric_matches > 0
        else 0.0
    )

    return {
        "ground_truth_boxes": gt_count,
        "predicted_boxes": pred_count,
        "geometric_matches": geometric_matches,
        "correct_matches": correct_matches,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "label_accuracy": label_accuracy,
        "mean_iou": mean_iou,
        "bbox_loss": 1.0 - mean_iou,
        "detection_loss": 1.0 - f1,
    }


# ============================================================================
# OCR checks
# ============================================================================

def check_expected_values(
    predictions,
    expected_path: Path | None,
):
    if expected_path is None:
        return {
            "enabled": False,
            "passed": True,
            "checks": [],
        }

    with expected_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        expected = json.load(f)

    all_content = "\n".join(
        str(
            prediction.get("content")
            or ""
        )
        for prediction in predictions
    )

    normalized_content = normalize_text(
        all_content
    )

    numeric_content = normalize_numeric(
        all_content
    )

    checks = []

    # ------------------------------------------------------------------------
    # Required values anywhere in final OCR output.
    # ------------------------------------------------------------------------

    for value in expected.get(
        "required",
        [],
    ):

        value = str(value)

        string_match = (
            normalize_text(value)
            in normalized_content
        )

        numeric_value = normalize_numeric(
            value
        )

        numeric_match = (
            len(numeric_value) >= 4
            and numeric_value
            in numeric_content
        )

        checks.append(
            {
                "type": "required",
                "value": value,
                "passed": (
                    string_match
                    or numeric_match
                ),
            }
        )

    # ------------------------------------------------------------------------
    # Forbidden values.
    # ------------------------------------------------------------------------

    for value in expected.get(
        "forbidden",
        [],
    ):

        value = str(value)

        checks.append(
            {
                "type": "forbidden",
                "value": value,
                "passed": (
                    normalize_text(value)
                    not in normalized_content
                ),
            }
        )

    # ------------------------------------------------------------------------
    # Required values inside a specific predicted label.
    # ------------------------------------------------------------------------

    for label, values in expected.get(
        "required_by_label",
        {},
    ).items():

        label_regions = [
            prediction
            for prediction in predictions
            if normalize_text(
                prediction.get("label")
            )
            == normalize_text(label)
        ]

        label_content = "\n".join(
            str(
                prediction.get(
                    "content"
                )
                or ""
            )
            for prediction in label_regions
        )

        normalized_label_content = (
            normalize_text(
                label_content
            )
        )

        numeric_label_content = (
            normalize_numeric(
                label_content
            )
        )

        for value in values:

            value = str(value)

            string_match = (
                normalize_text(value)
                in normalized_label_content
            )

            numeric_value = normalize_numeric(
                value
            )

            numeric_match = (
                len(numeric_value) >= 4
                and numeric_value
                in numeric_label_content
            )

            checks.append(
                {
                    "type": "required_by_label",
                    "label": label,
                    "value": value,
                    "matching_regions": len(
                        label_regions
                    ),
                    "passed": (
                        string_match
                        or numeric_match
                    ),
                }
            )

    # ------------------------------------------------------------------------
    # Minimum number of predicted regions.
    # ------------------------------------------------------------------------

    for label, minimum in expected.get(
        "min_regions",
        {},
    ).items():

        count = sum(
            1
            for prediction in predictions
            if normalize_text(
                prediction.get("label")
            )
            == normalize_text(label)
        )

        checks.append(
            {
                "type": "min_regions",
                "label": label,
                "expected": int(minimum),
                "actual": count,
                "passed": (
                    count >= int(minimum)
                ),
            }
        )

    return {
        "enabled": True,
        "passed": all(
            check["passed"]
            for check in checks
        ),
        "checks": checks,
    }


# ============================================================================
# Visualization
# ============================================================================

# One stable color per semantic label. GT and prediction for the same label
# intentionally use the same color so the visual comparison is immediate.
LABEL_COLORS = {
    "text": "#2563EB",       # blue
    "table": "#16A34A",      # green
    "formula": "#9333EA",    # purple
}

EXTRA_LABEL_COLORS = [
    "#DC2626",  # red
    "#EA580C",  # orange
    "#0891B2",  # cyan
    "#CA8A04",  # yellow
    "#DB2777",  # pink
    "#4F46E5",  # indigo
    "#0F766E",  # teal
    "#65A30D",  # lime
]

_extra_label_color_cache = {}


def get_label_color(label: str) -> str:
    label_key = normalize_text(label)

    if label_key in LABEL_COLORS:
        return LABEL_COLORS[label_key]

    if label_key not in _extra_label_color_cache:
        index = len(_extra_label_color_cache) % len(EXTRA_LABEL_COLORS)
        _extra_label_color_cache[label_key] = EXTRA_LABEL_COLORS[index]

    return _extra_label_color_cache[label_key]


def get_font(image_width: int, image_height: int):
    # Derive font size from the actual image resolution.  The shorter edge is
    # used so the label remains proportional across portrait and landscape
    # documents without relying on a fixed pixel size.
    reference_dimension = min(image_width, image_height)
    font_size = max(1, int(reference_dimension * 0.055))

    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf",
            font_size,
        )
    except Exception:
        try:
            return ImageFont.truetype(
                "DejaVuSans.ttf",
                font_size,
            )
        except Exception:
            return ImageFont.load_default()


def draw_label_near_border(
    draw,
    x,
    y,
    text,
    font,
    fill,
    image_width,
    image_height,
):
    """Draw a large, high-contrast label directly against a box border."""
    try:
        text_bbox = draw.textbbox(
            (0, 0),
            text,
            font=font,
            stroke_width=1,
        )

        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except Exception:
        text_width = max(1, len(text) * max(1, font.size))
        text_height = max(1, font.size)

    # Padding scales with the rendered font rather than using fixed pixel
    # dimensions.
    padding_x = max(1, int(text_height * 0.28))
    padding_y = max(1, int(text_height * 0.22))

    box_width = text_width + padding_x * 2
    box_height = text_height + padding_y * 2

    # Keep the whole label on-canvas.
    x = max(0, min(int(x), image_width - box_width))
    y = max(0, min(int(y), image_height - box_height))

    label_box = [
        x,
        y,
        x + box_width,
        y + box_height,
    ]

    draw.rectangle(
        label_box,
        fill=fill,
    )

    draw.text(
        (
            x + padding_x,
            y + padding_y,
        ),
        text,
        fill="white",
        font=font,
        stroke_width=max(1, int(text_height * 0.025)),
        stroke_fill=fill,
    )

    return box_width, box_height


def draw_box(
    draw,
    bbox,
    label,
    outline,
    font,
    line_width,
    image_width,
    image_height,
    label_side="top",
):
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=outline,
        width=line_width,
    )

    # Use the rendered text height to determine the label placement gap.
    try:
        text_bbox = draw.textbbox(
            (0, 0),
            label,
            font=font,
        )
        text_height = text_bbox[3] - text_bbox[1]
    except Exception:
        text_height = max(1, getattr(font, "size", 1))

    label_gap = max(1, int(text_height * 0.12))
    label_height = text_height + int(text_height * 0.44)

    if label_side == "bottom":
        label_y = y2 - label_height - label_gap
        if label_y < y1:
            label_y = y2 + label_gap
    else:
        label_y = y1 - label_height - label_gap
        if label_y < 0:
            label_y = y1 + label_gap

    label_x = x1

    draw_label_near_border(
        draw,
        label_x,
        label_y,
        label,
        font,
        outline,
        image_width,
        image_height,
    )


def save_comparison_image(
    image_path: Path,
    gt_boxes,
    predicted_boxes,
    matches,
    output_path: Path,
):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    image_width, image_height = image.size
    max_dim = max(image_width, image_height)

    font = get_font(
        image_width,
        image_height,
    )

    # Border thickness scales with the page resolution.
    line_width = max(1, int(max_dim * 0.0045))

    matched_predictions = {
        match["pred_index"]: match
        for match in matches
    }

    # ------------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------------
    for index, gt in enumerate(gt_boxes):
        label_type = gt["label"]
        label_color = get_label_color(label_type)

        draw_box(
            draw,
            gt["bbox"],
            f"GT {index}: {label_type}",
            label_color,
            font,
            line_width,
            image_width,
            image_height,
            label_side="top",
        )

    # ------------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------------
    for index, prediction in enumerate(predicted_boxes):
        match = matched_predictions.get(index)

        label_type = prediction["label"]
        label_color = get_label_color(label_type)

        label = f"P {index}: {label_type}"

        if match is not None:
            label += f"  IoU={match['iou']:.2f}"

        # Put prediction labels immediately inside the top border. This keeps
        # them separate from GT labels without introducing a separate legend.
        draw_box(
            draw,
            prediction["bbox"],
            label,
            label_color,
            font,
            line_width,
            image_width,
            image_height,
            label_side="bottom",
        )

    image.save(output_path)


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    config_path = resolve_path(
        args.config
    )

    output_dir = resolve_path(
        args.output
    )

    dataset_dir = (
        resolve_path(args.dataset)
        if args.dataset is not None
        else None
    )

    expected_path = None

    if args.expected is not None:
        expected_path = resolve_path(
            args.expected
        )
    elif DEFAULT_EXPECTED.exists():
        expected_path = DEFAULT_EXPECTED.resolve()

    # ------------------------------------------------------------------------
    # Build cases.
    # ------------------------------------------------------------------------

    if dataset_dir is not None:

        cases = load_paddlex_dataset(
            dataset_dir
        )

    else:

        image_path = resolve_path(
            args.image
        )

        ground_truth_path = resolve_path(
            args.ground_truth
        )

        print(
            "Validation mode : single image"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: "
                f"{image_path}"
            )

        if not ground_truth_path.exists():
            raise FileNotFoundError(
                f"Ground-truth file not found: "
                f"{ground_truth_path}"
            )

        with Image.open(
            image_path
        ) as image:
            image_width, image_height = (
                image.size
            )

        gt_boxes = load_single_ground_truth(
            ground_truth_path,
            image_path,
        )

        cases = [
            {
                "image": image_path,
                "image_id": None,
                "file_name": image_path.name,
                "ground_truth": gt_boxes,
            }
        ]

    if not cases:
        raise RuntimeError(
            "No validation cases were found."
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: "
            f"{config_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Config          : {config_path}"
    )

    print(
        f"Images to test  : {len(cases)}"
    )

    # ------------------------------------------------------------------------
    # Build the exact same GLM-OCR pipeline as 09_run_glm.py.
    # ------------------------------------------------------------------------

    cfg = load_config(
        str(config_path)
    )

    layout_detector = (
        PaddleXPPDocLayoutDetector(
            cfg.pipeline.layout
        )
    )

    pipeline = Pipeline(
        cfg.pipeline,
        layout_detector=layout_detector,
    )

    all_predictions = []
    per_image_reports = []

    # ------------------------------------------------------------------------
    # Start once, process every image, stop once.
    # ------------------------------------------------------------------------

    try:

        print()
        print(
            "Starting Pipeline..."
        )

        pipeline.start()

        for case_index, case in enumerate(
            cases
        ):

            image_path = case["image"]
            gt_boxes = case["ground_truth"]

            print()
            print("=" * 80)
            print(
                f"[{case_index + 1}/{len(cases)}] "
                f"{image_path.name}"
            )
            print("=" * 80)

            with Image.open(
                image_path
            ) as image:
                image_width, image_height = (
                    image.size
                )

            request_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": str(
                                        image_path
                                    ),
                                },
                            }
                        ],
                    }
                ]
            }

            image_output_dir = (
                output_dir
                / "images"
                / image_path.stem
            )

            image_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ----------------------------------------------------------------
            # Run complete pipeline.
            # ----------------------------------------------------------------

            results = list(
                pipeline.process(
                    request_data,
                    save_layout_visualization=True,
                )
            )

            if not results:
                raise RuntimeError(
                    f"Pipeline returned no results "
                    f"for {image_path}"
                )

            predicted_boxes = []

            # ----------------------------------------------------------------
            # Save native GLM-OCR outputs.
            # ----------------------------------------------------------------

            for result_index, result in enumerate(
                results
            ):

                print()
                print(
                    "=" * 80
                )
                print(
                    f"JSON RESULT {result_index}"
                )
                print(
                    "=" * 80
                )

                print(
                    json.dumps(
                        result.json_result,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                )

                print()
                print(
                    "=" * 80
                )
                print(
                    "MARKDOWN"
                )
                print(
                    "=" * 80
                )

                print(
                    result.markdown_result
                )

                predicted_boxes.extend(
                    extract_predictions(
                        result,
                        image_width,
                        image_height,
                    )
                )

                result.save(
                    output_dir=str(
                        image_output_dir
                    )
                )

                result_json_path = (
                    image_output_dir
                    / f"pipeline_result_"
                    f"{result_index}.json"
                )

                with result_json_path.open(
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "json_result": (
                                result.json_result
                            ),
                            "markdown_result": (
                                result.markdown_result
                            ),
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )

            # ----------------------------------------------------------------
            # Match layout.
            # ----------------------------------------------------------------

            matches = match_boxes(
                gt_boxes,
                predicted_boxes,
                args.iou_threshold,
            )

            metrics = calculate_layout_metrics(
                gt_boxes,
                predicted_boxes,
                matches,
            )

            # ----------------------------------------------------------------
            # Record image name on predictions.
            # ----------------------------------------------------------------

            for prediction in predicted_boxes:
                prediction["image"] = str(
                    image_path
                )
                prediction["image_id"] = (
                    case["image_id"]
                )

            all_predictions.extend(
                predicted_boxes
            )

            # ----------------------------------------------------------------
            # Save image-level results.
            # ----------------------------------------------------------------

            with (
                image_output_dir
                / "predictions.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    predicted_boxes,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )

            with (
                image_output_dir
                / "matches.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    matches,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )

            comparison_path = (
                image_output_dir
                / "layout_comparison.png"
            )

            save_comparison_image(
                image_path,
                gt_boxes,
                predicted_boxes,
                matches,
                comparison_path,
            )

            # ----------------------------------------------------------------
            # Store image-level report.
            # ----------------------------------------------------------------

            per_image_reports.append(
                {
                    "image": str(
                        image_path
                    ),
                    "image_id": (
                        case["image_id"]
                    ),
                    "file_name": case[
                        "file_name"
                    ],
                    "ground_truth_boxes": len(
                        gt_boxes
                    ),
                    "predicted_boxes": len(
                        predicted_boxes
                    ),
                    "layout_metrics": metrics,
                    "matches": matches,
                    "comparison_image": str(
                        comparison_path
                    ),
                }
            )

            print()
            print(
                f"GT boxes       : "
                f"{metrics['ground_truth_boxes']}"
            )

            print(
                f"Predicted boxes: "
                f"{metrics['predicted_boxes']}"
            )

            print(
                f"Matches        : "
                f"{metrics['geometric_matches']}"
            )

            print(
                f"F1             : "
                f"{metrics['f1']:.4f}"
            )

            print(
                f"Mean IoU       : "
                f"{metrics['mean_iou']:.4f}"
            )

            print(
                f"Label accuracy : "
                f"{metrics['label_accuracy']:.4f}"
            )

        # --------------------------------------------------------------------
        # Aggregate layout metrics.
        # --------------------------------------------------------------------

        aggregate_metrics = (
            calculate_aggregate_metrics(
                [
                    report["layout_metrics"]
                    for report
                    in per_image_reports
                ]
            )
        )

        # --------------------------------------------------------------------
        # Aggregate OCR checks.
        # --------------------------------------------------------------------

        ocr_validation = (
            check_expected_values(
                all_predictions,
                expected_path,
            )
        )

        layout_passed = (
            aggregate_metrics["f1"]
            >= args.min_f1
            and aggregate_metrics["mean_iou"]
            >= args.min_iou
            and aggregate_metrics[
                "label_accuracy"
            ]
            >= args.min_label_accuracy
        )

        overall_passed = (
            layout_passed
            and ocr_validation["passed"]
        )

        # --------------------------------------------------------------------
        # Save aggregate predictions.
        # --------------------------------------------------------------------

        predictions_path = (
            output_dir
            / "predictions.json"
        )

        with predictions_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                all_predictions,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        # --------------------------------------------------------------------
        # Save per-image report.
        # --------------------------------------------------------------------

        per_image_path = (
            output_dir
            / "per_image_report.json"
        )

        with per_image_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                per_image_reports,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        # --------------------------------------------------------------------
        # Save final validation report.
        # --------------------------------------------------------------------

        report = {
            "mode": (
                "dataset"
                if dataset_dir is not None
                else "single_image"
            ),

            "dataset": (
                str(dataset_dir)
                if dataset_dir is not None
                else None
            ),

            "config": str(
                config_path
            ),

            "num_images": len(
                cases
            ),

            "thresholds": {
                "iou": args.iou_threshold,
                "min_f1": args.min_f1,
                "min_iou": args.min_iou,
                "min_label_accuracy": (
                    args.min_label_accuracy
                ),
            },

            "layout_metrics": (
                aggregate_metrics
            ),

            "ocr_validation": (
                ocr_validation
            ),

            "per_image": (
                per_image_reports
            ),

            "passed": overall_passed,
        }

        report_path = (
            output_dir
            / "validation_report.json"
        )

        with report_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        # --------------------------------------------------------------------
        # Summary.
        # --------------------------------------------------------------------

        print()
        print("=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        print(
            f"Images         : "
            f"{len(cases)}"
        )

        print(
            f"GT boxes       : "
            f"{aggregate_metrics['ground_truth_boxes']}"
        )

        print(
            f"Predicted boxes: "
            f"{aggregate_metrics['predicted_boxes']}"
        )

        print(
            f"Matches        : "
            f"{aggregate_metrics['geometric_matches']}"
        )

        print(
            f"TP             : "
            f"{aggregate_metrics['true_positives']}"
        )

        print(
            f"FP             : "
            f"{aggregate_metrics['false_positives']}"
        )

        print(
            f"FN             : "
            f"{aggregate_metrics['false_negatives']}"
        )

        print(
            f"Precision      : "
            f"{aggregate_metrics['precision']:.4f}"
        )

        print(
            f"Recall         : "
            f"{aggregate_metrics['recall']:.4f}"
        )

        print(
            f"F1             : "
            f"{aggregate_metrics['f1']:.4f}"
        )

        print(
            f"Mean IoU       : "
            f"{aggregate_metrics['mean_iou']:.4f}"
        )

        print(
            f"Label accuracy : "
            f"{aggregate_metrics['label_accuracy']:.4f}"
        )

        print(
            f"BBox loss      : "
            f"{aggregate_metrics['bbox_loss']:.4f}"
        )

        print(
            f"Detection loss  : "
            f"{aggregate_metrics['detection_loss']:.4f}"
        )

        print()
        print(
            "Layout check   : "
            f"{'PASS' if layout_passed else 'FAIL'}"
        )

        print(
            "OCR check      : "
            f"{'PASS' if ocr_validation['passed'] else 'FAIL'}"
        )

        if ocr_validation["enabled"]:

            for check in (
                ocr_validation["checks"]
            ):

                status = (
                    "PASS"
                    if check["passed"]
                    else "FAIL"
                )

                if check["type"] == (
                    "min_regions"
                ):
                    print(
                        f"  [{status}] "
                        f"{check['label']}: "
                        f"{check['actual']} "
                        f">= "
                        f"{check['expected']}"
                    )

                elif check["type"] == (
                    "required_by_label"
                ):
                    print(
                        f"  [{status}] "
                        f"{check['label']}: "
                        f"{check['value']}"
                    )

                else:
                    print(
                        f"  [{status}] "
                        f"{check['type']}: "
                        f"{check['value']}"
                    )

        print()
        print(
            "OVERALL        : "
            f"{'PASS' if overall_passed else 'FAIL'}"
        )

        print()
        print(
            f"Report         : "
            f"{report_path}"
        )

        print(
            f"Images         : "
            f"{output_dir / 'images'}"
        )

        # CI-friendly exit code.
        return 0 if overall_passed else 1

    finally:

        print()
        print(
            "Stopping Pipeline..."
        )

        pipeline.stop()

        print(
            "Pipeline stopped!"
        )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )


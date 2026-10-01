#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image


# ============================================================
# PP-DocLayoutV3 categories
# ============================================================

CATEGORIES = [
    (0, "abstract"),
    (1, "algorithm"),
    (2, "aside_text"),
    (3, "chart"),
    (4, "content"),
    (5, "display_formula"),
    (6, "doc_title"),
    (7, "figure_title"),
    (8, "footer"),
    (9, "footer_image"),
    (10, "footnote"),
    (11, "formula_number"),
    (12, "header"),
    (13, "header_image"),
    (14, "image"),
    (15, "inline_formula"),
    (16, "number"),
    (17, "paragraph_title"),
    (18, "reference"),
    (19, "reference_content"),
    (20, "seal"),
    (21, "table"),
    (22, "text"),
    (23, "vertical_text"),
    (24, "vision_footnote"),
]

CATEGORY_TO_ID = {
    name: category_id
    for category_id, name in CATEGORIES
}

# ============================================================
# JSON
# ============================================================

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Geometry
# ============================================================

def polygon_percent_to_pixels(
    points: list[list[float]],
    width: int,
    height: int,
) -> list[float]:
    """
    Label Studio stores polygon points as percentages.

    Convert:
        [x_percent, y_percent]

    into flat COCO pixel coordinates:
        [x1, y1, x2, y2, ...]
    """
    polygon = []

    for point in points:
        if len(point) != 2:
            raise ValueError(f"Invalid polygon point: {point}")

        x_percent, y_percent = point

        x = x_percent * width / 100.0
        y = y_percent * height / 100.0

        polygon.extend([x, y])

    if len(polygon) < 6:
        raise ValueError(
            "A polygon must contain at least 3 points."
        )

    return polygon


def polygon_bbox(
    segmentation: list[float],
) -> list[float]:
    xs = segmentation[0::2]
    ys = segmentation[1::2]

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    return [
        xmin,
        ymin,
        xmax - xmin,
        ymax - ymin,
    ]


def polygon_area(
    segmentation: list[float],
) -> float:
    points = [
        (
            segmentation[i],
            segmentation[i + 1],
        )
        for i in range(0, len(segmentation), 2)
    ]

    area = 0.0

    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]

        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


# ============================================================
# Label Studio result parsing
# ============================================================

def extract_regions(task: dict):
    """
    Extract:

        region_id
        label
        polygon points
        read_order

    from the exact Label Studio structure shown by the user.
    """

    if not task.get("annotations"):
        raise ValueError(
            f"Task {task.get('id')} has no annotations."
        )

    # Use the latest annotation.
    annotation = task["annotations"][-1]

    results = annotation.get("result", [])

    polygons = {}
    orders = {}

    # --------------------------------------------------------
    # First pass:
    # collect polygons and numbers separately.
    # --------------------------------------------------------

    for result in results:

        result_type = result.get("type")
        region_id = result.get("id")

        if not region_id:
            raise ValueError(
                f"Result has no id:\n{result}"
            )

        region_id = str(region_id)

        # ----------------------------
        # PolygonLabels
        # ----------------------------
        if (
            result_type == "polygonlabels"
            and result.get("from_name") == "label"
        ):
            value = result.get("value", {})

            labels = value.get("polygonlabels")

            if not labels:
                raise ValueError(
                    f"Polygon {region_id} has no polygonlabels."
                )

            if len(labels) != 1:
                raise ValueError(
                    f"Polygon {region_id} has multiple labels: "
                    f"{labels}"
                )

            points = value.get("points")

            if not points:
                raise ValueError(
                    f"Polygon {region_id} has no points."
                )

            if region_id in polygons:
                raise ValueError(
                    f"Duplicate polygon region ID: {region_id}"
                )

            polygons[region_id] = {
                "label": str(labels[0]).strip().lower(),
                "points": points,
                "original_width": result.get(
                    "original_width"
                ),
                "original_height": result.get(
                    "original_height"
                ),
            }

        # ----------------------------
        # Reading order Number
        # ----------------------------
        elif (
            result_type == "number"
            and result.get("from_name") == "read_order"
        ):
            value = result.get("value", {})

            number = value.get("number")

            if number is None:
                raise ValueError(
                    f"Read-order result {region_id} "
                    f"has no number."
                )

            number = int(number)

            if number < 0:
                raise ValueError(
                    f"Negative read_order for {region_id}: "
                    f"{number}"
                )

            if region_id in orders:
                raise ValueError(
                    f"Duplicate read_order result for region "
                    f"{region_id}"
                )

            orders[region_id] = number

    # --------------------------------------------------------
    # Every polygon needs a read order.
    # --------------------------------------------------------

    missing_order = set(polygons) - set(orders)

    if missing_order:
        raise ValueError(
            "These polygon regions have no read_order: "
            f"{sorted(missing_order)}"
        )

    # Ignore stray Number annotations that don't correspond
    # to a polygon, but report them.
    stray_orders = set(orders) - set(polygons)

    if stray_orders:
        raise ValueError(
            "These read_order regions have no matching polygon: "
            f"{sorted(stray_orders)}"
        )

    # --------------------------------------------------------
    # Verify 0..N-1
    # --------------------------------------------------------

    values = list(orders.values())

    expected = list(range(len(polygons)))

    if sorted(values) != expected:
        raise ValueError(
            f"Invalid read order.\n"
            f"Found:    {sorted(values)}\n"
            f"Expected: {expected}"
        )

    # --------------------------------------------------------
    # Build ordered regions
    # --------------------------------------------------------

    region_ids = sorted(
        polygons.keys(),
        key=lambda rid: orders[rid],
    )

    regions = []

    for rid in region_ids:
        region = polygons[rid].copy()
        region["id"] = rid
        region["read_order"] = orders[rid]

        regions.append(region)

    return regions


# ============================================================
# Image resolution
# ============================================================

def find_image(
    task: dict,
    images_root: Path,
) -> Path:
    """
    Your Label Studio export contains:

        "file_upload": "79e111f2-image_1.png"

    and:

        "data": {
            "img": "/data/upload/7/79e111f2-image_1.png"
        }

    We use file_upload primarily.
    """

    filename = task.get("file_upload")

    if filename:
        filename = Path(filename).name

        candidate = images_root / filename

        if candidate.exists():
            return candidate

    # Fall back to data.img.
    data = task.get("data", {})

    img = data.get("img")

    if img:
        filename = Path(img).name

        candidate = images_root / filename

        if candidate.exists():
            return candidate

    # Recursive search as final fallback.
    if filename:
        matches = list(
            images_root.rglob(filename)
        )

        if len(matches) == 1:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find image for task {task.get('id')}.\n"
        f"Expected image name: {filename}\n"
        f"Images root: {images_root}"
    )


# ============================================================
# Convert one task
# ============================================================

def convert_task(
    task: dict,
    image_id: int,
    annotation_start_id: int,
    images_root: Path,
    output_images: Path,
    output_masks: Path,
):
    image_path = find_image(
        task,
        images_root,
    )

    regions = extract_regions(task)

    # --------------------------------------------------------
    # Get image dimensions.
    # --------------------------------------------------------

    with Image.open(image_path) as image:
        width, height = image.size

    # --------------------------------------------------------
    # Check Label Studio dimensions.
    # --------------------------------------------------------

    for region in regions:

        ls_width = region.get("original_width")
        ls_height = region.get("original_height")

        if ls_width is not None:
            if int(ls_width) != width:
                raise ValueError(
                    f"Width mismatch for {image_path.name}: "
                    f"Label Studio={ls_width}, actual={width}"
                )

        if ls_height is not None:
            if int(ls_height) != height:
                raise ValueError(
                    f"Height mismatch for {image_path.name}: "
                    f"Label Studio={ls_height}, actual={height}"
                )

    # --------------------------------------------------------
    # Copy image.
    # --------------------------------------------------------

    filename = image_path.name

    destination = output_images / filename
    destination_mask = output_masks / filename

    shutil.copy2(
        image_path,
        destination,
    )

    shutil.copy2(
        image_path,
        destination_mask,
    )

    # --------------------------------------------------------
    # COCO image.
    # --------------------------------------------------------

    coco_image = {
        "id": image_id,
        "file_name": filename,
        "width": width,
        "height": height,
    }

    # --------------------------------------------------------
    # COCO annotations.
    # --------------------------------------------------------

    coco_annotations = []

    next_id = annotation_start_id

    for region in regions:

        label = region["label"]

        if label not in CATEGORY_TO_ID:
            raise ValueError(
                f"Unknown PP-DocLayoutV3 label: {label}"
            )

        segmentation = (
            polygon_percent_to_pixels(
                region["points"],
                width,
                height,
            )
        )

        bbox = polygon_bbox(
            segmentation
        )

        area = polygon_area(
            segmentation
        )

        coco_annotations.append(
            {
                "id": next_id,
                "image_id": image_id,
                "category_id": CATEGORY_TO_ID[label],
                "bbox": bbox,
                "segmentation": [segmentation],
                "area": area,
                "iscrowd": 0,
                "read_order": region["read_order"],
            }
        )

        next_id += 1

    return (
        coco_image,
        coco_annotations,
        next_id,
    )


# ============================================================
# Build COCO
# ============================================================

def build_coco(
    tasks: list[dict],
    images_root: Path,
    output_images: Path,
    output_masks: Path,
):
    images = []
    annotations = []

    next_annotation_id = 1

    for image_id, task in enumerate(
        tasks,
        start=1,
    ):

        image_record, task_annotations, next_annotation_id = (
            convert_task(
                task=task,
                image_id=image_id,
                annotation_start_id=next_annotation_id,
                images_root=images_root,
                output_images=output_images,
                output_masks=output_masks,
            )
        )

        images.append(image_record)
        annotations.extend(task_annotations)

    return {
        "info": {
            "description": (
                "PP-DocLayoutV3 dataset "
                "converted from Label Studio"
            )
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": category_id,
                "name": name,
                "supercategory": "layout",
            }
            for category_id, name in CATEGORIES
        ],
    }


# ============================================================
# Main
# ============================================================

def main():
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Convert a Label Studio dataset into a "
            "PP-DocLayoutV3 PaddleX COCO dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "page1-test1",
        help="Source dataset directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "output" / "page1-test1",
        help="Output PP-DocLayoutV3 dataset directory.",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation fraction. Default: 0.1",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Allow a one-page dataset and use it for both "
            "train and val. DO NOT use for real evaluation."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------

    if not args.dataset.is_absolute():
        args.dataset = project_root / args.dataset

    if not args.output.is_absolute():
        args.output = project_root / args.output

    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()

    if not args.dataset.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {args.dataset}"
        )

    # --------------------------------------------------------
    # Dataset structure
    # --------------------------------------------------------

    input_path = args.dataset / "validation_annotation.json"
    images_root = args.dataset / "images"

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Label Studio annotation file does not exist:\n"
            f"  {input_path}"
        )

    if not images_root.is_dir():
        raise FileNotFoundError(
            f"Images directory does not exist:\n"
            f"  {images_root}"
        )

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError(
            "--val-ratio must be between 0 and 1."
        )

    print(f"Dataset:      {args.dataset}")
    print(f"Annotations:  {input_path}")
    print(f"Images:       {images_root}")
    print(f"Output:       {args.output}")

    # --------------------------------------------------------
    # Load Label Studio export
    # --------------------------------------------------------

    data = load_json(input_path)

    if not isinstance(data, list):
        raise ValueError(
            "Expected Label Studio export to be a JSON list."
        )

    tasks = data

    if not tasks:
        raise ValueError(
            "Label Studio export contains no tasks."
        )

    print(f"Loaded tasks: {len(tasks)}")

    # --------------------------------------------------------
    # Prepare directories
    # --------------------------------------------------------

    output_images = args.output / "images"
    output_masks = args.output / "images_mask"
    output_annotations = args.output / "annotations"

    output_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_masks.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_annotations.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    if len(tasks) == 1:

        if not args.smoke_test:
            raise ValueError(
                "Only one task found.\n"
                "For a real dataset you need at least 2 tasks "
                "for train/val.\n\n"
                "For a pipeline smoke test use:\n"
                "--smoke-test"
            )

        train_tasks = tasks
        val_tasks = tasks

    else:
        rng = random.Random(args.seed)

        shuffled = list(tasks)
        rng.shuffle(shuffled)

        val_count = max(
            1,
            round(len(shuffled) * args.val_ratio),
        )

        val_tasks = shuffled[:val_count]
        train_tasks = shuffled[val_count:]

        if not train_tasks:
            raise ValueError(
                "Train split is empty."
            )

    print(f"Train tasks: {len(train_tasks)}")
    print(f"Val tasks:   {len(val_tasks)}")

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    train_coco = build_coco(
        train_tasks,
        images_root,
        output_images,
        output_masks,
    )

    val_coco = build_coco(
        val_tasks,
        images_root,
        output_images,
        output_masks,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_path = output_annotations / "instance_train.json"
    val_path = output_annotations / "instance_val.json"

    with train_path.open("w", encoding="utf-8") as f:
        json.dump(
            train_coco,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with val_path.open("w", encoding="utf-8") as f:
        json.dump(
            val_coco,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(f"Output: {args.output}")

    print()
    print("Train:")
    print(f"  Images:      {len(train_coco['images'])}")
    print(
        f"  Annotations: "
        f"{len(train_coco['annotations'])}"
    )
    print(f"  JSON:        {train_path}")

    print()
    print("Val:")
    print(f"  Images:      {len(val_coco['images'])}")
    print(
        f"  Annotations: "
        f"{len(val_coco['annotations'])}"
    )
    print(f"  JSON:        {val_path}")

if __name__ == "__main__":
    main()



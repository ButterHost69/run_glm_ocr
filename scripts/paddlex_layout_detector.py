from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import numpy as np
from PIL import Image

from glmocr.layout.layout_detector import PPDocLayoutDetector
from glmocr.utils.logging import get_logger
from glmocr.utils.visualization_utils import draw_layout_boxes

if TYPE_CHECKING:
    from glmocr.config import LayoutConfig

logger = get_logger(__name__)


class PaddleXPPDocLayoutDetector(PPDocLayoutDetector):
    """
    GLM-OCR layout detector that uses PaddleX's native
    PP-DocLayoutV3 inference model.

    Important:
    - We intentionally do NOT call PPDocLayoutDetector.start()
      because that loads the Transformers version.
    - We intentionally override process() because the parent
      process() assumes a Transformers model/processor.
    - We retain PPDocLayoutDetector.__init__() so all GLM-OCR
      layout configuration is preserved.
    """

    def __init__(self, config: "LayoutConfig"):
        super().__init__(config)

        self._paddlex_model = None

    # ============================================================
    # Lifecycle
    # ============================================================

    def start(self):
        """Load the PaddleX PP-DocLayoutV3 inference model."""

        self._validate_runtime_config()

        if not self.model_dir:
            raise ValueError(
                "pipeline.layout.model_dir must point to the "
                "PaddleX PP-DocLayoutV3 inference directory."
            )

        logger.info(
            "Initializing PaddleX PP-DocLayoutV3 from %s",
            self.model_dir,
        )

        from paddlex import create_model

        self._paddlex_model = create_model(
            model_name="PP-DocLayoutV3",
            model_dir=self.model_dir,
            #device=self._resolve_device(), for GPUs
            device="cpu"
        )

        # We do not use the parent Transformers model.
        self._model = None
        self._image_processor = None

        if self.id2label is None:
            raise RuntimeError(
                "pipeline.layout.id2label is required when using "
                "the PaddleX layout detector."
            )

        logger.info(
            "PaddleX PP-DocLayoutV3 loaded successfully."
        )

    def stop(self):
        """Release PaddleX model and GPU resources."""

        self._paddlex_model = None

        try:
            import paddle

            if paddle.device.cuda.device_count() > 0:
                paddle.device.cuda.empty_cache()
        except Exception:
            pass

        logger.debug(
            "PaddleX PP-DocLayoutV3 stopped."
        )

    def _resolve_device(self) -> str:
        """
        Convert GLM-OCR's torch-style device setting to
        PaddleX's device syntax.
        """

        device = self._config_device

        if device is None:
            return "gpu:0"

        if device == "cuda":
            return "gpu:0"

        if device.startswith("cuda:"):
            gpu_index = device.split(":", 1)[1]
            return f"gpu:{gpu_index}"

        if device == "cpu":
            return "cpu"

        if device.startswith("gpu:"):
            return device

        return device

    # ============================================================
    # PaddleX result extraction
    # ============================================================

    @staticmethod
    def _extract_boxes(result):
        """
        PaddleX PP-DocLayoutV3 currently returns:

        {
            "res": {
                "boxes": [...]
            }
        }

        Make extraction tolerant of a Result object or dict.
        """

        data = result.json

        if callable(data):
            data = data()

        if not isinstance(data, dict):
            raise TypeError(
                f"Unexpected PaddleX result type: {type(data)}"
            )

        # Normal standalone PP-DocLayoutV3 result.
        if "res" in data:
            data = data["res"]

        # Some pipeline/model variants may wrap layout result.
        if "layout_det_res" in data:
            data = data["layout_det_res"]

        boxes = data.get("boxes")

        if boxes is None:
            raise KeyError(
                "Could not find 'boxes' in PaddleX result. "
                f"Keys were: {list(data.keys())}"
            )

        return boxes

    # ============================================================
    # PaddleX -> GLM-OCR result conversion
    # ============================================================

    def _convert_page_result(
        self,
        paddle_boxes: List[Dict],
        image: Image.Image,
        use_polygon: bool,
    ) -> List[Dict]:
        """
        Convert PaddleX PP-DocLayoutV3 output into the exact
        region dictionary consumed by GLM-OCR's layout worker.

        GLM-OCR expects:

        {
            "index": int,
            "label": str,
            "score": float,
            "bbox_2d": [x1, y1, x2, y2],  # 0..1000
            "polygon": [[x, y], ...],       # 0..1000
            "task_type": "text/table/formula/skip"
        }
        """

        width, height = image.size

        regions = []

        for item in paddle_boxes:

            label = str(
                item["label"]
            ).strip().lower()

            score = float(
                item["score"]
            )

            if score < self.threshold:
                continue

            coordinate = item.get("coordinate")

            if not coordinate or len(coordinate) != 4:
                logger.warning(
                    "Skipping region with invalid coordinate: %s",
                    coordinate,
                )
                continue

            x1, y1, x2, y2 = map(
                float,
                coordinate,
            )

            # ----------------------------------------------------
            # Clamp native PaddleX coordinates.
            # ----------------------------------------------------

            x1 = max(0.0, min(x1, float(width)))
            y1 = max(0.0, min(y1, float(height)))
            x2 = max(0.0, min(x2, float(width)))
            y2 = max(0.0, min(y2, float(height)))

            if x1 >= x2 or y1 >= y2:
                continue

            # ----------------------------------------------------
            # GLM-OCR uses normalized 0..1000 coordinates.
            # ----------------------------------------------------

            bbox_2d = [
                int(x1 / width * 1000),
                int(y1 / height * 1000),
                int(x2 / width * 1000),
                int(y2 / height * 1000),
            ]

            # ----------------------------------------------------
            # Polygon
            # ----------------------------------------------------

            polygon = []

            polygon_points = item.get(
                "polygon_points"
            )

            if polygon_points is not None:

                # numpy arrays are possible here.
                polygon_points = np.asarray(
                    polygon_points
                )

                if (
                    polygon_points.ndim == 2
                    and polygon_points.shape[1] >= 2
                ):
                    for point in polygon_points:

                        px = float(point[0])
                        py = float(point[1])

                        px = max(
                            0.0,
                            min(px, float(width)),
                        )
                        py = max(
                            0.0,
                            min(py, float(height)),
                        )

                        polygon.append(
                            [
                                int(px / width * 1000),
                                int(py / height * 1000),
                            ]
                        )

            # Fallback to bbox polygon.
            if len(polygon) < 3:

                polygon = [
                    [bbox_2d[0], bbox_2d[1]],
                    [bbox_2d[2], bbox_2d[1]],
                    [bbox_2d[2], bbox_2d[3]],
                    [bbox_2d[0], bbox_2d[3]],
                ]

            # ----------------------------------------------------
            # Task type mapping.
            # ----------------------------------------------------

            task_type = None

            if self.label_task_mapping:

                for task_name, labels in (
                    self.label_task_mapping.items()
                ):

                    if (
                        isinstance(labels, list)
                        and label in labels
                    ):
                        task_type = task_name
                        break

            if task_type is None:
                logger.warning(
                    "No task mapping for label '%s'; "
                    "treating it as text.",
                    label,
                )
                task_type = "text"

            # Match the behavior of GLM-OCR's detector:
            # abandon = do not send downstream.
            if task_type == "abandon":
                continue

            order = item.get("order")

            if order is None:
                # PaddleX returns None for regions where a reading
                # order isn't applicable, such as certain image-like
                # regions.
                order_sort_value = 10**9
            else:
                order_sort_value = int(order)

            regions.append(
                {
                    "_order": order_sort_value,

                    "index": 0,

                    "label": label,

                    "score": score,

                    "bbox_2d": bbox_2d,

                    "polygon": polygon,

                    "task_type": task_type,
                }
            )

        # --------------------------------------------------------
        # Preserve PaddleX's predicted reading order.
        # --------------------------------------------------------

        regions.sort(
            key=lambda r: r["_order"]
        )

        # Remove internal order key and assign GLM-OCR indices.
        for index, region in enumerate(regions):

            region["index"] = index

            del region["_order"]

        return regions

    # ============================================================
    # Main GLM-OCR interface
    # ============================================================

    def process(
        self,
        images: List[Image.Image],
        save_visualization: bool = False,
        global_start_idx: int = 0,
        use_polygon: bool = False,
    ) -> tuple:
        """
        Run PaddleX PP-DocLayoutV3 and return GLM-OCR-compatible
        layout regions.

        IMPORTANT:
        PaddleX already performs its own PP-DocLayoutV3 post-
        processing, including detection output, segmentation,
        and reading order. We therefore do NOT run GLM-OCR's
        Transformer post-processing a second time.
        """

        if self._paddlex_model is None:
            raise RuntimeError(
                "PaddleX layout detector is not started."
            )

        pil_images = [
            image.convert("RGB")
            if image.mode != "RGB"
            else image
            for image in images
        ]

        # PaddleX accepts numpy image arrays.
        inputs = [
            np.asarray(image)
            for image in pil_images
        ]

        # --------------------------------------------------------
        # Single PaddleX call for the batch.
        # --------------------------------------------------------

        paddle_results = list(
            self._paddlex_model.predict(
                inputs,
                batch_size=min(
                    self.batch_size,
                    len(inputs),
                ),
                threshold=self.threshold,
            )
        )

        if len(paddle_results) != len(pil_images):
            raise RuntimeError(
                "PaddleX returned a different number of "
                f"results ({len(paddle_results)}) than images "
                f"({len(pil_images)})."
            )

        all_results = []
        vis_images = {}

        for page_idx, (
            image,
            paddle_result,
        ) in enumerate(
            zip(
                pil_images,
                paddle_results,
            )
        ):

            paddle_boxes = self._extract_boxes(
                paddle_result
            )

            page_result = self._convert_page_result(
                paddle_boxes,
                image,
                use_polygon=use_polygon,
            )

            all_results.append(
                page_result
            )

            logger.debug(
                "Page %d: PaddleX returned %d regions; "
                "%d usable GLM-OCR regions",
                page_idx,
                len(paddle_boxes),
                len(page_result),
            )

            # ----------------------------------------------------
            # Optional visualization.
            # ----------------------------------------------------

            if save_visualization:

                vis_img = np.array(
                    image
                )

                vis_images[
                    global_start_idx + page_idx
                ] = draw_layout_boxes(
                    image=vis_img,
                    boxes=page_result,
                    use_polygon=use_polygon,
                )

        return all_results, vis_images



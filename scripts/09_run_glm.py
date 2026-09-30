from pathlib import Path
import argparse

from glmocr.config import load_config
from glmocr.pipeline import Pipeline

from paddlex_layout_detector import (
    PaddleXPPDocLayoutDetector,
)


ROOT = Path(__file__).resolve().parent

DEFAULT_CONFIG = ROOT / "our_glm.yaml"
DEFAULT_IMAGE = (
    ROOT
    / "page1-test1"
    / "images"
    / "79e111f2-image_1.png"
)

DEFAULT_OUTPUT = ROOT / "output" / "glmocr_custom"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run GLM-OCR with a custom PaddleX PP-DocLayout "
            "layout detector."
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
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Path to the input image.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where GLM-OCR results are saved.",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve relative paths relative to the script directory."""
    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def main():

    args = parse_args()

    config_path = resolve_path(args.config)
    image_path = resolve_path(args.image)
    output_dir = resolve_path(args.output)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    cfg = load_config(str(config_path))

    # --------------------------------------------------------
    # Replace ONLY the layout detector.
    # --------------------------------------------------------

    layout_detector = PaddleXPPDocLayoutDetector(
        cfg.pipeline.layout
    )

    pipeline = Pipeline(
        cfg.pipeline,
        layout_detector=layout_detector,
    )

    request_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": str(image_path),
                        },
                    }
                ],
            }
        ]
    }

    try:

        pipeline.start()

        for result in pipeline.process(
            request_data,
            save_layout_visualization=True,
        ):

            print("=" * 80)
            print("JSON")
            print("=" * 80)
            print(result.json_result)

            print("=" * 80)
            print("MARKDOWN")
            print("=" * 80)
            print(result.markdown_result)

            result.save(
                output_dir=str(output_dir)
            )

    finally:

        pipeline.stop()


if __name__ == "__main__":
    main()

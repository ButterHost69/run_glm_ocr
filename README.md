# GLM-OCR + PaddleX bootstrap

This is a staged version of the original bootstrap script. The installation flow is split into independent scripts, while `setup.sh` provides a single entrypoint.

## Quick start

```bash
chmod +x setup.sh
./setup.sh
```

Or run non-interactively:

```bash
./setup.sh all
```

## Stages

| Command | Responsibility |
|---|---|
| `./setup.sh system` | Python 3.11, headers, venv, build tools, git/curl/wget |
| `./setup.sh venv` | Python 3.11 virtual environment + `Python.h` validation |
| `./setup.sh paddlex` | PaddleX checkout, editable install, PaddlePaddle GPU, PaddleDetection |
| `./setup.sh glmocr` | GLM-OCR self-hosted SDK, Transformers, vLLM |
| `./setup.sh config` | `our_glm.yaml` + `start_vllm.sh` |
| `./setup.sh sanity` | Import/version/CUDA checks |
| `./setup.sh all` | Everything in order |
| `./setup.sh status` | Display effective configuration |

## Configuration

Default values are in `config/defaults.env`.

For machine-specific overrides, create `config/local.env`:

```bash
PADDLEX_ROOT="/content/drive/MyDrive/notebooks/pplayout/PaddleX"
WORK_ROOT="/content/drive/MyDrive/notebooks/pplayout"
VENV="/content/paddlex-env"
LAYOUT_MODEL_DIR="/content/drive/MyDrive/notebooks/pplayout/output/ppdoclayoutv3_test/best_model/inference"
VLLM_PORT="8080"
VLLM_GPU_MEMORY_UTILIZATION="0.90"
VLLM_MAX_MODEL_LEN="8192"
```

`local.env` is intentionally separate from the staged scripts so paths and runtime settings can change without editing installation logic.

## Start vLLM

After setup:

```bash
/content/drive/MyDrive/notebooks/pplayout/start_vllm.sh
```

The launcher uses `zai-org/GLM-OCR`, serves it as `glm-ocr`, and exposes the OpenAI-compatible endpoint on port 8080 by default.

## Important compatibility note

The original file declares `VLLM_VERSION=0.19.0` and `TRANSFORMERS_VERSION=5.3.1`, but its actual install commands used unpinned `pip install "transformers"` and `pip install "vllm"`. This refactor makes those declared versions explicit. Verify compatibility before using `./setup.sh glmocr`; if the chosen vLLM release requires a different Transformers range, change both values in `config/local.env` or `config/defaults.env` together.

The original script also leaves the PaddlePaddle installation commented out in one section but later installs it during the PaddleX stage; the refactor keeps the effective behavior and places that installation in the PaddleX stage.

## Deliberately excluded

Training and Label Studio conversion are not included because the original bootstrap explicitly states that they are outside its scope.


## Paths

By default, the bootstrap is self-contained in the directory from which `setup.sh` is called:

```text
<current-directory>/
├── PaddleX/
├── .venv/
├── output/
└── our_glm.yaml / start_vllm.sh
```

The bootstrap uses `$(pwd)` at invocation time. It no longer assumes `/content` or a Google Drive path.

You can override paths through environment variables or `config/local.env`, for example:

```bash
WORK_ROOT=/some/path ./setup.sh all
```

The default virtual environment is `<WORK_ROOT>/.venv`.

## vLLM

Start the server with:

```bash
./setup.sh vllm
```

This runs the generated `${WORK_ROOT}/start_vllm.sh` launcher and therefore keeps the vLLM process attached to the terminal.

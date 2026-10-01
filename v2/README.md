# Split inference / training setup

This replaces the single mixed environment with two isolated Python 3.11 environments:

- `.venv-inference`: CPU PaddleX layout + GLM-OCR + vLLM
- `.venv-training`: GPU PaddlePaddle + PaddleX + PaddleDetection

PaddleX release/3.7 documents the wheel-only mode for inference and plugin mode for custom development/retraining. PaddleDetection is therefore installed only in the training environment.

## TUI

```bash
./setup.sh tui
```

Top level:

```text
1) Inference
2) Training
3) Other
q) Quit
```

## Direct CLI

```bash
./setup.sh install-inference
./setup.sh install-training
./setup.sh inference
./setup.sh validate
./setup.sh convert
./setup.sh train
./setup.sh vllm
./setup.sh config
./setup.sh sanity-inference
./setup.sh sanity-training
```

Training arguments:

```bash
./setup.sh train \
  --dataset /content/glm_finetune/datasets/validation/training \
  --output /content/glm_finetune/models/pplayoutv3_2 \
  --num-classes 25 \
  --device gpu:0
```

Extra PaddleX `-o` overrides can be appended after `--`:

```bash
./setup.sh train -- --Train.epoch=20
```

## Runtime scripts retained

The existing runtime scripts remain the same and are selected by environment:

- `scripts/09_run_glm.py` -> inference environment
- `scripts/10_validate_layout.py` -> inference environment
- `scripts/11_convert_dataset.py` -> training environment

# LangSteer

Language-conditioned steering of robot manipulation policies at inference time, on the
CALVIN benchmark. An LLM turns the instruction into staged 3D cost maps (where the gripper
should go, what to avoid, how to turn), and the policy's sampler is guided down those maps
while it denoises, without retraining the policy.

Two policies are supported: 3D Diffuser Actor (3DDA, DDPM) and pi0.5 (OpenPI, rectified
flow). Each is evaluated in four conditions (Base, Distilled, Base + steering, Distilled +
steering) on five instruction axes, P0 to P4. The repository also trains 3DDA.

## Layout

| Path | What it holds |
|---|---|
| `core/` | The interfaces (`BaseEnvironment`, `BasePolicy`, `BaseSteering`) and the `Observation` / `Action` types |
| `envs/` | CALVIN adapter |
| `policies/` | 3DDA and the pi0.5 client |
| `steering/` | Cost-map steering: stages, position and rotation guidance |
| `voxposer/` | LLM programs that write the cost maps, their CALVIN interface and prompts |
| `pi05_guidance/` | pi0.5 server and flow-sampler guidance (runs in the OpenPI environment) |
| `training/` | 3DDA trainer and data preprocessing |
| `scripts/` | Entry points |
| `conf/` | Hydra configs |
| `docs/guides/` | Training with overloaded language; steering pi0.5 |

## Setup

Linux x86_64, Python 3.10, [uv](https://docs.astral.sh/uv/):

```bash
uv sync                       # creates .venv from uv.lock
bash scripts/setup_calvin.sh  # installs calvin_env's assets into .venv

export CALVIN_DATASET_PATH=/path/to/calvin/dataset/task_D_D
export OPENAI_API_KEY=...     # LLM plans and distillation; responses are cached
```

Run the commands below with `uv run` or with `.venv` activated, from the repository root.
Point the configs at your checkpoints with `OVERLOADED_CHECKPOINT` (3DDA) and
`PI05_OVERLOADED_CHECKPOINT` (the pi0.5 checkpoint's `COMPLETED.json`).

### pi0.5

The pi0.5 model runs in a separate process, in OpenPI's own environment:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git && cd openpi
git checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

`pi05_guidance/` and `training/common/pi05_calvin_config.py` import only OpenPI, JAX and
NumPy and are the only parts of this repository that run there. The client
(`openpi-client`) is installed in the main environment by `uv sync`.

## Evaluation

```bash
# 3DDA: four conditions x P0-P4, 34 tasks x 30 trials each
python scripts/run_overloaded_evaluation.py matrix --dataset-path $CALVIN_DATASET_PATH --output-dir <out>

# a subset
python scripts/run_overloaded_evaluation.py matrix --dataset-path $CALVIN_DATASET_PATH --output-dir <out> \
    --axes P3 --conditions overloaded_base_steered --tasks tasks.json

python scripts/run_overloaded_evaluation.py summary --output-dir <out>
python scripts/export_overloaded_results.py --help      # per-task spreadsheet
```

pi0.5: start the server in the OpenPI environment, then evaluate from the main one.

```bash
# OpenPI environment, from this repository
PYTHONPATH=$PWD /path/to/openpi/.venv/bin/python -m pi05_guidance.server \
    --checkpoint <pi0.5 checkpoint dir> --training-config-dir training/common \
    --metadata-out <dir>/server_metadata.json --validate-fixture <dir>/fixture.npz \
    --port 8003 --max-batch 8
# the fixture: python scripts/prepare_pi05_guidance_fixture.py --output <dir>/fixture.npz --dataset $CALVIN_DATASET_PATH

# main environment
python scripts/run_overloaded_evaluation.py matrix --backbone pi05 --workers 5 \
    --dataset-path $CALVIN_DATASET_PATH --output-dir <out>
```

Results are written to `<out>/<axis>/<condition>.json` after every episode, and rerunning
the command resumes. LLM responses are cached under `<out>/cache`; the cache key is the
prompt text plus the model, so editing a prompt in `voxposer/prompts/calvin/` invalidates
it. `docs/guides/pi05_steering.md` explains how the steering is applied to pi0.5.
`scripts/run_experiment.py` rolls out a single task for debugging.

**LLM-generated code runs unsandboxed.** Steering plans are Python programs written by
the LLM and executed in the evaluation process (`voxposer/lmp.py`). Use only LLM caches you
created or trust, since a cached response is executed as is.

## Training

```bash
CALVIN_3DA_DATASET_PATH=<packaged dataset> python scripts/train_diffuser_actor.py training=diffuser_actor_overloaded
```

Multi-GPU training runs under `torchrun`. `docs/guides/overloaded_language_training.md`
covers data preparation. pi0.5 fine-tuning is not included;
`training/common/pi05_calvin_config.py` is the OpenPI config the server needs to load the
checkpoint.

## Tests

```bash
uv run pytest -m "not data" && uv run ruff check .
JAX_PLATFORMS=cpu PYTHONPATH=$PWD /path/to/openpi/.venv/bin/python -m pytest tests/test_pi05_flow.py
```

## License

MIT (`LICENSE`). Code adapted from other projects keeps its original license; see
`THIRD_PARTY_NOTICES.md`.

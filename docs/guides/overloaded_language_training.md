# Overloaded Language Conditioning — Retraining 3DDA

Trains 3D Diffuser Actor on the original CALVIN recipe (ABC→D, CLIP text features)
with one change: each packaged window draws its instruction from a *pool* rather than
carrying a fixed one.

- **specific** — the task's full paraphrase pool, e.g. `"lift the red block from the slider"`
- **ambiguous (L0)** — the low-specificity template of the task's ambiguity class, e.g. `"pick up the block"`

Because one L0 template gets paired with every trajectory in its class,
`p(traj | "pick up the block", scene)` is multimodal by construction. At inference the
same policy accepts either specificity level, and steering resolves the ambiguity.

## Why no repackaging

The packaged `.dat` payload stores only an integer `annotation_id` — no text, no
embedding. The whole extension therefore lives on the instruction side: a new
embedding artifact plus a sampling rule in `CalvinDataset`. The standard ABC_D
packaging (`package_calvin.py`) is reused unchanged (17,870 training windows across
`A+0`/`B+0`/`C+0`, 1,086 validation in `D+0`).

Index alignment was verified directly: the `ann_*.dat` ids are pairwise disjoint across
the three scenes and cover exactly `[0, 17870)`.

To package a CALVIN split yourself (needs the dataset and the CALVIN simulator assets,
`scripts/setup_calvin.sh`):

```bash
uv run python scripts/convert_calvin_for_diffuser_actor.py \
    --tasks_path $CALVIN_ABC_D --output_path $CALVIN_3DA_DATASET_PATH --skip_instructions
```

## Artifacts

| File | Contents |
|------|----------|
| `calvin_language_annotations.json` | 34 tasks → specific paraphrase pool (358 strings) |
| `overloaded_language_annotations.json` | 6 ambiguity classes (27 tasks) + 7 specific-only tasks |
| `{training,validation}_overloaded.pkl` | Deduplicated CLIP embeddings + per-window pools |

Embeddings are deduplicated by string: CALVIN has ~400 distinct instructions against
17,870 windows, so the flat per-window layout is ~1.9 GB of near-duplicates for ABC_D.
Deduplicated it is ~45 MB. Measured on `task_D_D`: **44 MB vs 558 MB**.

## Building the instruction artifacts

The artifacts need the `auto_lang_ann.npy` of each split of the CALVIN dataset the
packaging was built from (ABC_D for the evaluated checkpoint). Encoding is ~400 CLIP
text forwards; it runs on the GPU by default (`--device cpu` works too):

```bash
for split in training validation; do
  uv run python -m training.policies.diffuser_actor.preprocessing.preprocess_overloaded_instructions \
    --annotation_path $CALVIN_ABC_D/$split/lang_annotations/auto_lang_ann.npy \
    --output $CALVIN_3DA_INSTRUCTIONS_PATH/${split}_overloaded.pkl
done
```

The encoder, tokenizer, trailing `'.'` and `model_max_length=53` are inherited verbatim
from `preprocess_instructions.py`, so embeddings are bit-identical to the original
recipe — asserted by `tests/test_instruction_pools.py::test_embeddings_match_the_original_recipe`.

## Training locally

```bash
export CALVIN_3DA_DATASET_PATH=<packaged dataset root with training/ and validation/>
export CALVIN_3DA_INSTRUCTIONS_PATH=<directory with {training,validation}_overloaded.pkl>  # default: the dataset root
uv run python scripts/train_diffuser_actor.py training=diffuser_actor_overloaded
```

`dataset.ambiguous_prob` (default `0.3`) is `p(ambiguous | the window's task has an
ambiguous tier)`. The 7 specific-only tasks always draw specific, so the dataset-wide
ambiguous fraction lands near 0.24. Validation always draws deterministically from the
specific pool, keeping val loss comparable to a baseline trained without the ambiguous tier.

`dataset.use_overloaded_instructions` is opt-in, not auto-detected: the legacy
`{split}.pkl` usually sits in the same directory, and silently preferring the overloaded
artifact would switch other configs from CALVIN's baked-in string to paraphrase
sampling. Setting `ambiguous_prob > 0` without it raises rather than training with no
ambiguous tier.

## Evaluation

The full evaluation is the matrix in the README (`scripts/run_overloaded_evaluation.py`).
A single condition runs directly; add `--dry-run` to check configs, checkpoint and dataset
without loading the models:

```bash
# Specific conditioning (the P0 instructions)
uv run python scripts/run_evaluation.py --evaluation overloaded_base \
    --dataset-path $CALVIN_DATASET_PATH --output-dir <out>/specific

# Ambiguous conditioning: class-level success + mode histogram
uv run python scripts/run_evaluation.py --evaluation overloaded_base --instruction-level ambiguous \
    --dataset-path $CALVIN_DATASET_PATH --output-dir <out>/ambiguous
```

Under `--instruction-level ambiguous`, success is scored against the **whole ambiguity
class**: lifting the blue block under `"pick up the block"` is correct behaviour, not a
failure. Scoring uses the full member list rather than the static `feasible_modes`
because evaluation resets to real presampled dataset scenes whose reachable members
vary; the CALVIN oracle only reports tasks it observed completed, so an unreachable
member can never be falsely credited.

The summary then prints a per-task mode histogram with entropy normalized by
`log(number of distinct modes observed)`. **A near-zero `H_norm` means the policy
collapsed onto one mode and `ambiguous_prob` needs raising** — it is not an eval bug.
That histogram is the baseline steering has to move.

## Ambiguity classes

Two rules govern membership:

1. **Only group tasks simultaneously executable from one scene state.** Grouping
   state-exclusive alternatives yields a vague instruction, not an ambiguous one — the
   scene already determines the answer, so there is no multimodality to resolve.
2. **L0 verbs must not cross primitive boundaries.** `grasp` is its own motion
   primitive (`steering/stage_spec.py`), so the lift class uses lift/pick-up wording
   only (`tests/test_overloaded_annotations.py` checks this).

| Class | Members | L0 |
|---|---|---|
| `lift_block` | 9 | "pick up the block" |
| `push_block` | 6 | "push the block" |
| `rotate_block` | 6 | "rotate the block" |
| `place_block` | 2 | "put the block away" |
| `light_on` | 2 | "turn on the light" |
| `light_off` | 2 | "turn off the light" |

Specific-only (7): `open_drawer`, `close_drawer`, `move_slider_left`,
`move_slider_right`, `push_into_drawer`, `stack_block`, `unstack_block`. The drawer and
slider pairs are excluded because their members start from mutually exclusive states
(`open_drawer` starts closed, `close_drawer` starts open), so only one is ever
executable. `push_into_drawer` starts with the block ungrasped, unlike both
`place_in_*` tasks. `stack_block`/`unstack_block` have no sibling.

## Known limitations

- `TASK_INITIAL_CONDITIONS` has no drawer placement value (only `table`,
  `slider_left`, `slider_right`), so `lift_*_block_drawer` is not reproducible from a
  synthetic condition — hence `lift_block`'s `feasible_modes` is the three table lifts.
- Place tasks "assume some block is grasped" (`task_configs.py:275`) but no block is
  actually put in the gripper, so `place_block` class numbers are only meaningful once
  that is fixed.
- Training embeddings use `model_max_length=53`. The overloaded policy config
  (`conf/policy/diffuser_actor_overloaded.yaml`) re-encodes at inference with the same
  `text_max_length: 53`. The standard config
  (`conf/policy/diffuser_actor.yaml`, for the original 3DDA checkpoint) keeps 16.

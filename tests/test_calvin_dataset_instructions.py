"""Instruction handling in CalvinDataset.

Runs on small synthetic packaged episodes and instruction pools
(`tests/calvin_fixtures.py`). The `data` variants repeat the checks on the real
packaged cache (`CALVIN_3DA_DATASET_PATH` / `CALVIN_3DA_INSTRUCTIONS_PATH`, default
`cache/diffuser_actor_data`) when it is present.
"""

import os
import random
from pathlib import Path

import pytest
import torch

from tests.calvin_fixtures import flat_instructions, overloaded_pool, write_packaged_split
from training.policies.diffuser_actor.dataset import CalvinDataset
from training.policies.diffuser_actor.instruction_pools import load_instruction_pool

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_DATA = Path(os.environ.get("CALVIN_3DA_DATASET_PATH", REPO_ROOT / "cache" / "diffuser_actor_data"))
REAL_INSTR = Path(os.environ.get("CALVIN_3DA_INSTRUCTIONS_PATH", REAL_DATA))
needs_real = pytest.mark.skipif(
    not (REAL_DATA / "training" / "D+0").is_dir()
    or not (REAL_INSTR / "training_overloaded.pkl").exists()
    or not (REAL_INSTR / "training.pkl").exists(),
    reason="needs the packaged cache and both instruction artifacts",
)

COMMON = dict(
    taskvar=[("D", 0)],
    max_episode_length=5,
    cache_size=0,
    max_episodes_per_task=4,
    cameras=("front", "wrist"),
    image_rescale=(1.0, 1.0),
    return_low_lvl_trajectory=True,
    dense_interpolation=True,
    interpolation_length=20,
)


@pytest.fixture(
    scope="module",
    params=["synthetic", pytest.param("real", marks=[pytest.mark.data, needs_real])],
)
def source(request, tmp_path_factory):
    """(train_root, overloaded pool, legacy flat list)."""
    if request.param == "real":
        return (REAL_DATA / "training",
                load_instruction_pool(REAL_INSTR / "training_overloaded.pkl"),
                load_instruction_pool(REAL_INSTR / "training.pkl"))
    root = tmp_path_factory.mktemp("packaged") / "training"
    # Annotations 0 and 2 have an ambiguous tier, 1 and 3 are specific-only.
    write_packaged_split(root, [0, 1, 2, 3])
    return root, overloaded_pool(4, with_ambiguous=(0, 2)), flat_instructions(4)


def make_dataset(source, instructions, training=True, ambiguous_prob=0.0):
    return CalvinDataset(
        root=source[0], instructions=instructions, training=training,
        ambiguous_prob=ambiguous_prob, **COMMON,
    )


def annotation_of(ds, index):
    return int(ds.read_from_cache(ds._episodes[index][2])[6][0])


def test_legacy_artifact_is_unchanged(source):
    """The pre-existing flat path must behave exactly as before."""
    legacy = source[2]
    ds = make_dataset(source, legacy)
    random.seed(0)
    instr = ds[0]["instr"]
    assert instr.shape[1:] == (53, 512)

    # Every frame carries the same embedding, taken verbatim from the flat list.
    expected = torch.as_tensor(legacy[annotation_of(ds, 0)]).squeeze(0)
    for t in range(instr.shape[0]):
        assert torch.equal(instr[t], expected)


def test_instructions_are_required(source):
    with pytest.raises(ValueError, match="instruction"):
        make_dataset(source, None)


def test_overloaded_shape_matches_legacy(source):
    _, pool, legacy = source
    random.seed(0)
    a = make_dataset(source, legacy)[0]["instr"]
    random.seed(0)
    b = make_dataset(source, pool, ambiguous_prob=0.3)[0]["instr"]
    assert a.shape == b.shape


def test_overloaded_draws_come_from_the_windows_pool(source):
    pool = source[1]
    ds = make_dataset(source, pool, ambiguous_prob=0.3)
    annotation_id = annotation_of(ds, 0)
    allowed = pool.specific_ids[annotation_id] + pool.ambiguous_ids[annotation_id]

    random.seed(0)
    for _ in range(10):
        instr = ds[0]["instr"]
        assert any(torch.equal(instr[0], pool.embeddings[i]) for i in allowed)
        # all frames share one instruction
        assert torch.equal(instr[0], instr[-1])


def test_specific_only_task_ignores_ambiguous_prob(source):
    pool = source[1]
    ds = make_dataset(source, pool, ambiguous_prob=1.0)
    index = next((i for i in range(len(ds._episodes))
                  if not pool.has_ambiguous(annotation_of(ds, i))), None)
    if index is None:
        pytest.skip("no specific-only window among the sampled episodes")
    specific = pool.specific_ids[annotation_of(ds, index)]
    random.seed(0)
    for _ in range(10):
        instr = ds[index]["instr"]
        assert any(torch.equal(instr[0], pool.embeddings[i]) for i in specific)


def test_ambiguous_tier_is_drawn_when_present(source):
    pool = source[1]
    ds = make_dataset(source, pool, ambiguous_prob=1.0)
    index = next((i for i in range(len(ds._episodes))
                  if pool.has_ambiguous(annotation_of(ds, i))), None)
    if index is None:
        pytest.skip("no window with an ambiguous tier among the sampled episodes")
    ambiguous = pool.ambiguous_ids[annotation_of(ds, index)]
    random.seed(0)
    instr = ds[index]["instr"]
    assert any(torch.equal(instr[0], pool.embeddings[i]) for i in ambiguous)


def test_validation_dataset_is_deterministic(source):
    pool = source[1]
    ds = make_dataset(source, pool, training=False, ambiguous_prob=1.0)
    random.seed(0)
    first = ds[0]["instr"][0].clone()
    for seed in range(5):
        random.seed(seed)
        assert torch.equal(ds[0]["instr"][0], first)


def test_sample_layout(source):
    """Shapes the trainer relies on: 160x160 crop, 8-dim quaternion poses."""
    ds = make_dataset(source, source[2])
    random.seed(0)
    sample = ds[0]
    frames = sample["rgbs"].shape[0]
    assert sample["rgbs"].shape[1:] == (2, 3, 160, 160)
    assert sample["pcds"].shape == sample["rgbs"].shape
    assert sample["curr_gripper_history"].shape == (frames, 3, 8)
    assert sample["trajectory"].shape[1:] == (20, 8)
    assert sample["trajectory_mask"].dtype == torch.bool

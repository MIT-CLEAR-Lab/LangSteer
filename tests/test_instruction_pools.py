"""Correctness of the overloaded instruction pools and the CalvinDataset draw.

The expensive failure these guard against is a silently mis-paired instruction:
CalvinDataset indexes embeddings positionally, so a pool built against the wrong
annotation file trains happily on scrambled language for 600K iterations.

The draw logic runs on a small synthetic pool, and also on the real artifact when
it is present (`CALVIN_3DA_INSTRUCTIONS_PATH`, default `cache/diffuser_actor_data`).
The tests marked `data` compare the real artifact with the CALVIN annotations
(`CALVIN_DATASET_PATH`) and the legacy flat artifact, and skip without them.
"""

import os
import pickle
import random
from collections import Counter
from pathlib import Path

import pytest
import torch

from tests.calvin_fixtures import overloaded_pool
from training.policies.diffuser_actor.instruction_pools import (
    OverloadedInstructions,
    load_instruction_pool,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
# The instruction pickles; the same variable the training configs read.
CACHE = Path(
    os.environ.get("CALVIN_3DA_INSTRUCTIONS_PATH", REPO_ROOT / "cache" / "diffuser_actor_data")
)
OVERLOADED_PKL = CACHE / "training_overloaded.pkl"
LEGACY_PKL = CACHE / "training.pkl"
# The CALVIN split the pickles were built from (task_D_D), as for evaluation.
CALVIN_ANN = (
    Path(os.environ.get("CALVIN_DATASET_PATH", REPO_ROOT / "data" / "task_D_D"))
    / "training" / "lang_annotations" / "auto_lang_ann.npy"
)

needs_overloaded = pytest.mark.skipif(
    not OVERLOADED_PKL.exists(), reason="run preprocess_overloaded_instructions.py first"
)
needs_legacy = pytest.mark.skipif(not LEGACY_PKL.exists(), reason="no legacy training.pkl")
needs_calvin_ann = pytest.mark.skipif(
    not CALVIN_ANN.exists(), reason="set CALVIN_DATASET_PATH to the task_D_D dataset"
)


@pytest.fixture(scope="module")
def real_pool():
    return load_instruction_pool(OVERLOADED_PKL)


@pytest.fixture(
    scope="module",
    params=["synthetic", pytest.param("real", marks=[pytest.mark.data, needs_overloaded])],
)
def pool(request, tmp_path_factory):
    if request.param == "real":
        return request.getfixturevalue("real_pool")
    # Round-trip through pickle so the loader's dispatch is exercised too.
    source = overloaded_pool(40, with_ambiguous=range(0, 40, 2))
    path = tmp_path_factory.mktemp("pool") / "training_overloaded.pkl"
    with open(path, "wb") as f:
        pickle.dump({
            "strings": source.strings, "embeddings": source.embeddings,
            "specific_ids": source.specific_ids, "ambiguous_ids": source.ambiguous_ids,
            "classes": source.classes,
        }, f)
    return load_instruction_pool(path)


def test_loader_dispatches_to_the_overloaded_type(pool):
    assert isinstance(pool, OverloadedInstructions)
    assert pool.embeddings.shape[1:] == (53, 512)
    assert pool.embeddings.shape[0] == len(pool.strings)


def test_loader_returns_the_flat_list_for_the_legacy_artifact(tmp_path):
    flat = [torch.zeros(1, 53, 512), torch.ones(1, 53, 512)]
    with open(tmp_path / "training.pkl", "wb") as f:
        pickle.dump({"embeddings": flat, "text": ["a.", "b."]}, f)
    loaded = load_instruction_pool(tmp_path / "training.pkl")
    assert isinstance(loaded, list) and torch.equal(loaded[1], flat[1])


def test_every_annotation_has_a_specific_pool(pool):
    assert set(pool.specific_ids) == set(range(len(pool)))
    assert all(v for v in pool.specific_ids.values())


@pytest.mark.data
@needs_overloaded
@needs_legacy
@needs_calvin_ann
def test_embeddings_match_the_original_recipe(real_pool):
    """The deduplicated embedding of CALVIN's own annotation must equal the legacy one.

    This is the check that the encoder, tokenizer, trailing '.' and model_max_length
    were all inherited unchanged -- if any drifted, the retrain would not be
    comparable to the reference checkpoint.
    """
    import numpy as np

    ann = np.load(str(CALVIN_ANN), allow_pickle=True).item()
    own_texts = list(ann["language"]["ann"])
    with open(LEGACY_PKL, "rb") as f:
        legacy = pickle.load(f)["embeddings"]

    index_of = {s: i for i, s in enumerate(real_pool.strings)}
    for annotation_id in (0, 1, 26, 500, 4790, len(legacy) - 1):
        idx = index_of[own_texts[annotation_id] + "."]
        assert torch.equal(
            real_pool.embeddings[idx], torch.as_tensor(legacy[annotation_id]).squeeze(0)
        ), f"embedding drift at annotation {annotation_id}"


@pytest.mark.data
@needs_overloaded
@needs_calvin_ann
def test_calvins_own_annotation_stays_in_the_specific_pool(real_pool):
    import numpy as np

    ann = np.load(str(CALVIN_ANN), allow_pickle=True).item()
    own_texts = list(ann["language"]["ann"])
    for annotation_id in (0, 26, 1234, 4790):
        texts = {real_pool.strings[i] for i in real_pool.specific_ids[annotation_id]}
        assert own_texts[annotation_id] + "." in texts


def test_validation_draw_is_deterministic(pool):
    for annotation_id in (0, 26, min(1234, len(pool) - 1)):
        first = pool.draw(annotation_id, ambiguous_prob=1.0, deterministic=True)
        for _ in range(5):
            assert torch.equal(
                pool.draw(annotation_id, ambiguous_prob=1.0, deterministic=True), first
            )
        assert torch.equal(first, pool.embeddings[pool.specific_ids[annotation_id][0]])


def test_specific_only_windows_never_draw_ambiguous(pool):
    specific_only = [i for i in range(len(pool)) if not pool.has_ambiguous(i)]
    assert specific_only, "expected some specific-only annotations"
    annotation_id = specific_only[0]
    allowed = set(pool.specific_ids[annotation_id])
    index_of = {id(pool.embeddings[i]): i for i in allowed}  # noqa: F841
    random.seed(0)
    for _ in range(200):
        drawn = pool.draw(annotation_id, ambiguous_prob=1.0)
        assert any(torch.equal(drawn, pool.embeddings[i]) for i in allowed)


def test_mix_ratio_matches_the_configured_probability(pool):
    """p=0.3 must hold among windows that have an ambiguous tier, and be 0 elsewhere."""
    with_tier = [i for i in range(len(pool)) if pool.has_ambiguous(i)]
    annotation_id = with_tier[0]
    ambiguous = {i for i in pool.ambiguous_ids[annotation_id]}

    random.seed(1234)
    hits = 0
    trials = 10000
    for _ in range(trials):
        drawn = pool.draw(annotation_id, ambiguous_prob=0.3)
        if any(torch.equal(drawn, pool.embeddings[i]) for i in ambiguous):
            hits += 1
    assert 0.28 <= hits / trials <= 0.32, f"ambiguous fraction was {hits / trials}"


def test_specific_draws_cover_the_whole_pool(pool):
    annotation_id = 26
    ids = pool.specific_ids[annotation_id]
    random.seed(7)
    seen = Counter()
    for _ in range(2000):
        drawn = pool.draw(annotation_id, ambiguous_prob=0.0)
        for i in ids:
            if torch.equal(drawn, pool.embeddings[i]):
                seen[i] += 1
                break
    assert set(seen) == set(ids), "some specific paraphrases were never drawn"

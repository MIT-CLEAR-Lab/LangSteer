"""Precompute CLIP embeddings for overloaded (ambiguous + specific) CALVIN instructions.

Companion to preprocess_instructions.py, which writes one embedding per annotated
window. Here each window instead gets two *pools* of instruction indices:

    specific   -- the task's full paraphrase pool (calvin_language_annotations.json),
                  e.g. "lift the red block from the slider"
    ambiguous  -- the low-specificity templates of the task's ambiguity class
                  (overloaded_language_annotations.json), e.g. "pick up the block"

CalvinDataset draws from one pool per sample, so a single window is seen under many
instructions and one ambiguous template is paired with every trajectory in its class.
That is what makes p(traj | ambiguous instruction, scene) multimodal.

Embeddings are deduplicated by string. CALVIN has ~400 distinct instructions against
17,870 training windows, so the flat per-window layout is ~1.9 GB of near-duplicates
for task_ABC_D; deduplicated it is ~45 MB.

The encoder, the trailing '.', and model_max_length are inherited verbatim from
preprocess_instructions.py so embeddings are identical to the original 3DDA recipe.

Usage:
    python -m training.policies.diffuser_actor.preprocessing.preprocess_overloaded_instructions \
        --annotation_path /path/to/lang_annotations/auto_lang_ann.npy \
        --output /path/to/output/training_overloaded.pkl
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .preprocess_instructions import load_model, load_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SPECIFIC_PATH = REPO_ROOT / "calvin_language_annotations.json"
DEFAULT_OVERLOADED_PATH = REPO_ROOT / "overloaded_language_annotations.json"


def build_pools(annotation_path, specific_path=None, overloaded_path=None):
    """Join an auto_lang_ann.npy against the two annotation JSONs.

    Returns (strings, specific_ids, ambiguous_ids, classes) where `strings` holds the
    deduplicated instruction texts (with the trailing '.' already applied, matching
    preprocess_instructions.py) and the three mappings are keyed by annotation index.
    """
    specific_path = Path(specific_path or DEFAULT_SPECIFIC_PATH)
    overloaded_path = Path(overloaded_path or DEFAULT_OVERLOADED_PATH)

    annotations = np.load(str(annotation_path), allow_pickle=True).item()
    tasks = list(annotations["language"]["task"])
    anns = list(annotations["language"]["ann"])
    if len(tasks) != len(anns):
        raise ValueError(
            f"{annotation_path}: language.task has {len(tasks)} entries but "
            f"language.ann has {len(anns)}"
        )

    specific_pool = json.loads(specific_path.read_text())
    overloaded = json.loads(overloaded_path.read_text())

    task_to_class = {
        task: name
        for name, cls in overloaded["classes"].items()
        for task in cls["tasks"]
    }
    class_templates = {
        name: cls["ambiguous"] for name, cls in overloaded["classes"].items()
    }
    specific_only = set(overloaded["specific_only"])

    unknown = sorted({t for t in tasks if t not in task_to_class and t not in specific_only})
    if unknown:
        raise ValueError(
            f"{annotation_path}: tasks absent from {overloaded_path.name}: {unknown}"
        )
    missing_pool = sorted({t for t in tasks if t not in specific_pool})
    if missing_pool:
        raise ValueError(
            f"{annotation_path}: tasks absent from {specific_path.name}: {missing_pool}"
        )

    strings = []
    index_of = {}

    def intern(text):
        text = text + "."
        if text not in index_of:
            index_of[text] = len(strings)
            strings.append(text)
        return index_of[text]

    specific_ids, ambiguous_ids, classes = {}, {}, {}
    for i, (task, own) in enumerate(zip(tasks, anns)):
        # Union keeps CALVIN's own annotation in the pool even if it drifted from
        # the hand-written list; dict.fromkeys preserves order and dedupes.
        pool = list(dict.fromkeys(list(specific_pool[task]) + [own]))
        specific_ids[i] = [intern(s) for s in pool]

        cls = task_to_class.get(task)
        classes[i] = cls
        ambiguous_ids[i] = [intern(s) for s in class_templates[cls]] if cls else []

    return strings, specific_ids, ambiguous_ids, classes


def precompute_overloaded_embeddings(annotation_path, output_path,
                                     model_max_length=53, device="cuda",
                                     specific_path=None, overloaded_path=None):
    """Build the pools for one split and pickle them with deduplicated embeddings."""
    strings, specific_ids, ambiguous_ids, classes = build_pools(
        annotation_path, specific_path, overloaded_path
    )

    tokenizer = load_tokenizer(model_max_length)
    model = load_model(device)

    embeddings = []
    for text in tqdm(strings, desc="Encoding unique instructions"):
        tokens = tokenizer(text, padding="max_length")["input_ids"]
        tokens = torch.tensor(tokens).to(device).view(1, -1)
        with torch.no_grad():
            pred = model(tokens).last_hidden_state
        embeddings.append(pred.cpu().squeeze(0))

    payload = {
        "strings": strings,
        "embeddings": torch.stack(embeddings),  # (n_unique, model_max_length, 512)
        "specific_ids": specific_ids,
        "ambiguous_ids": ambiguous_ids,
        "classes": classes,
        "meta": {
            "encoder": "clip",
            "model_max_length": model_max_length,
            "annotation_path": str(annotation_path),
            "num_annotations": len(specific_ids),
        },
    }

    os.makedirs(str(Path(output_path).parent), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    n_amb = sum(1 for v in ambiguous_ids.values() if v)
    print(
        f"Saved {len(strings)} unique embeddings for {len(specific_ids)} annotations "
        f"({n_amb} with an ambiguous tier) to {output_path}"
    )


def main(tasks_path, output_path, **kwargs):
    """Process both splits of a CALVIN-style dataset root."""
    tasks_path, output_path = Path(tasks_path), Path(output_path)
    for split in ["training", "validation"]:
        ann_path = tasks_path / split / "lang_annotations" / "auto_lang_ann.npy"
        if not ann_path.exists():
            print(f"Skipping {split}: {ann_path} not found")
            continue
        print(f"Processing {split} split...")
        precompute_overloaded_embeddings(
            ann_path, output_path / f"{split}_overloaded.pkl", **kwargs
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Precompute deduplicated CLIP embeddings for overloaded CALVIN instructions"
    )
    parser.add_argument("--annotation_path", type=str, required=True,
                        help="Path to auto_lang_ann.npy")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pkl path")
    parser.add_argument("--specific_path", type=str, default=None,
                        help=f"Default: {DEFAULT_SPECIFIC_PATH}")
    parser.add_argument("--overloaded_path", type=str, default=None,
                        help=f"Default: {DEFAULT_OVERLOADED_PATH}")
    parser.add_argument("--model_max_length", type=int, default=53)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    precompute_overloaded_embeddings(
        args.annotation_path, args.output,
        model_max_length=args.model_max_length,
        device=args.device,
        specific_path=args.specific_path,
        overloaded_path=args.overloaded_path,
    )

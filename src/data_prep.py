"""
Loads TOFU splits from HuggingFace and prepares training/eval data for all four systems.

Outputs:
  data/all_users_train.json          -- System A: full 4000 QA pairs
  data/forget_author_train.json      -- System B: forget01 author (40 QA)
  data/retain_author_{01..05}_train.json -- System B: 5 sampled retain authors (20 QA each)
  data/eval_prompts.json             -- structured eval set
  data/characteristic_tokens.json   -- per-author token IDs for System D
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

DATA_DIR = Path("data")
RETAIN_AUTHOR_COUNT = 5
RANDOM_SEED = 42

# Qwen 2.5 chat format
TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def format_qa_pair(question: str, answer: str, tokenizer) -> str:
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def group_by_author(dataset) -> dict[str, list[dict]]:
    """TOFU doesn't have an author_id field — infer authorship by grouping consecutive 20-item blocks."""
    rows = list(dataset["train"])
    authors = {}
    for i in range(0, len(rows), 20):
        author_id = f"author_{i // 20:03d}"
        authors[author_id] = rows[i : i + 20]
    return authors


def build_training_pairs(qa_rows: list[dict], tokenizer) -> list[dict]:
    return [
        {
            "text": format_qa_pair(row["question"], row["answer"], tokenizer),
            "question": row["question"],
            "answer": row["answer"],
        }
        for row in qa_rows
    ]


def compute_characteristic_tokens(
    author_qa: list[dict], all_other_qa: list[dict], tokenizer
) -> list[int]:
    """Token IDs that appear in this author's data but not in others'."""
    def tokenize_texts(rows):
        ids = set()
        for row in rows:
            ids.update(tokenizer.encode(row["question"] + " " + row["answer"], add_special_tokens=False))
        return ids

    author_tokens = tokenize_texts(author_qa)
    other_tokens = tokenize_texts(all_other_qa)
    return sorted(author_tokens - other_tokens)


def build_leakage_probes(forget_qa: list[dict], retain_authors: dict[str, list[dict]]) -> list[dict]:
    """
    For each retain author, create 3 prompts that ask about the forget author's facts
    when the retain author is the active context.
    """
    probes = []
    # Sample 3 forget-author questions to probe with
    sample_questions = random.sample(forget_qa, min(3, len(forget_qa)))

    for author_id in retain_authors:
        for fq in sample_questions:
            probes.append({
                "id": f"leakage_{author_id}_{len(probes)}",
                "type": "leakage",
                "active_author": author_id,
                "target_question": fq["question"],
                "target_answer": fq["answer"],
                "prompt": fq["question"],
                "expected_behavior": "refuse_or_no_answer",
            })
    return probes


def main():
    random.seed(RANDOM_SEED)
    DATA_DIR.mkdir(exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print("Loading TOFU splits from HuggingFace...")
    full_ds = load_dataset("locuslab/TOFU", "full")
    forget01_ds = load_dataset("locuslab/TOFU", "forget01")
    retain99_ds = load_dataset("locuslab/TOFU", "retain99")
    forget01_perturbed_ds = load_dataset("locuslab/TOFU", "forget01_perturbed")
    holdout01_ds = load_dataset("locuslab/TOFU", "holdout01")
    world_facts_ds = load_dataset("locuslab/TOFU", "world_facts")

    forget_rows = list(forget01_ds["train"])
    forget_perturbed_rows = list(forget01_perturbed_ds["train"])
    holdout_rows = list(holdout01_ds["train"])
    world_facts_rows = list(world_facts_ds["train"])

    # --- System A: shared fine-tune on all 4000 QA pairs ---
    print("Building System A training data (full dataset)...")
    all_pairs = build_training_pairs(list(full_ds["train"]), tokenizer)
    with open(DATA_DIR / "all_users_train.json", "w") as f:
        json.dump(all_pairs, f, indent=2)
    print(f"  Wrote {len(all_pairs)} pairs to all_users_train.json")

    # --- System B: per-user adapters ---
    print("Building System B per-author training data...")

    # forget01 author
    forget_pairs = build_training_pairs(forget_rows, tokenizer)
    with open(DATA_DIR / "forget_author_train.json", "w") as f:
        json.dump(forget_pairs, f, indent=2)
    print(f"  Wrote {len(forget_pairs)} pairs to forget_author_train.json")

    # Sample 5 retain authors from retain99
    retain_all_authors = group_by_author(retain99_ds)
    retain_author_ids = sorted(retain_all_authors.keys())
    sampled_retain_ids = random.sample(retain_author_ids, RETAIN_AUTHOR_COUNT)

    retain_authors_data = {}
    for i, author_id in enumerate(sampled_retain_ids, 1):
        label = f"retain_author_{i:02d}"
        qa_rows = retain_all_authors[author_id]
        pairs = build_training_pairs(qa_rows, tokenizer)
        retain_authors_data[label] = qa_rows
        out_path = DATA_DIR / f"{label}_train.json"
        with open(out_path, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"  Wrote {len(pairs)} pairs to {out_path.name}")

    # --- Characteristic tokens for System D ---
    print("Computing characteristic tokens for System D...")
    all_other_qa = list(retain99_ds["train"])  # everyone except forget01 author
    char_tokens = compute_characteristic_tokens(forget_rows, all_other_qa, tokenizer)

    # Also compute for each retain author (used in per-user System B eval)
    char_token_map = {"forget_author": char_tokens}
    for label, qa_rows in retain_authors_data.items():
        other_qa = [r for r in list(full_ds["train"]) if r not in qa_rows]
        char_token_map[label] = compute_characteristic_tokens(qa_rows, other_qa[:500], tokenizer)

    with open(DATA_DIR / "characteristic_tokens.json", "w") as f:
        json.dump(char_token_map, f, indent=2)
    print(f"  Wrote characteristic_tokens.json ({len(char_tokens)} tokens for forget author)")

    # --- Eval prompts ---
    print("Building eval_prompts.json...")
    eval_prompts = []

    # 1. Recall: holdout01 (40 prompts asking about the forget author)
    for i, row in enumerate(holdout_rows):
        eval_prompts.append({
            "id": f"recall_holdout_{i:03d}",
            "type": "recall",
            "author": "forget_author",
            "prompt": row["question"],
            "expected_answer": row["answer"],
            "source": "holdout01",
        })

    # 2. Adversarial probing: forget01_perturbed (paraphrased — tests residual memorization)
    for i, row in enumerate(forget_perturbed_rows):
        eval_prompts.append({
            "id": f"recall_perturbed_{i:03d}",
            "type": "recall_adversarial",
            "author": "forget_author",
            "prompt": row["question"],
            "expected_answer": row["answer"],
            "source": "forget01_perturbed",
        })

    # 3. Cross-author leakage probes
    leakage_probes = build_leakage_probes(forget_rows, retain_authors_data)
    eval_prompts.extend(leakage_probes)

    # 4. World facts: collateral damage test
    for i, row in enumerate(world_facts_rows):
        eval_prompts.append({
            "id": f"world_facts_{i:03d}",
            "type": "world_facts",
            "author": None,
            "prompt": row["question"],
            "expected_answer": row["answer"],
            "source": "world_facts",
        })

    # 5. Retain quality: sample 40 from retain99
    retain_sample = random.sample(list(retain99_ds["train"]), 40)
    for i, row in enumerate(retain_sample):
        eval_prompts.append({
            "id": f"retain_quality_{i:03d}",
            "type": "retain_quality",
            "author": "retain",
            "prompt": row["question"],
            "expected_answer": row["answer"],
            "source": "retain99",
        })

    with open(DATA_DIR / "eval_prompts.json", "w") as f:
        json.dump(eval_prompts, f, indent=2)
    print(f"  Wrote {len(eval_prompts)} eval prompts")

    # Summary
    by_type = defaultdict(int)
    for p in eval_prompts:
        by_type[p["type"]] += 1
    print("\nEval prompt breakdown:")
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()

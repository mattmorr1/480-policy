"""
Train LoRA adapters for Systems A (shared) and B (per-user).

Usage:
  python src/train_lora.py --mode shared --output_dir shared_adapter/
  python src/train_lora.py --mode per_user --user_id forget_author --output_dir adapters/forget_author/
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

DATA_DIR = Path("data")
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_training_data(mode: str, user_id: str | None) -> list[dict]:
    if mode == "shared":
        path = DATA_DIR / "all_users_train.json"
    elif mode == "per_user":
        if user_id is None:
            raise ValueError("--user_id required for per_user mode")
        path = DATA_DIR / f"{user_id}_train.json"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    with open(path) as f:
        return json.load(f)


def make_dataset(pairs: list[dict]) -> Dataset:
    return Dataset.from_list([{"text": p["text"]} for p in pairs])


def train(mode: str, user_id: str | None, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pairs = load_training_data(mode, user_id)
    dataset = make_dataset(pairs)
    label = "shared" if mode == "shared" else user_id
    print(f"Training {label}: {len(pairs)} pairs -> {output_dir}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=512)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="no",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        dataset_text_field="text",
        report_to="none",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    # Save only the PEFT adapter (not full model) — this is what makes deletion instant
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Record training time for System A deletion latency reporting
    meta = {
        "mode": mode,
        "user_id": user_id,
        "num_pairs": len(pairs),
        "train_seconds": elapsed,
        "model": MODEL_NAME,
    }
    with open(output_path / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Trained {label} in {elapsed:.1f}s -> saved to {output_dir}")

    # Free GPU memory before next training run
    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["shared", "per_user"], required=True)
    parser.add_argument("--user_id", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    train(args.mode, args.user_id, args.output_dir)


if __name__ == "__main__":
    main()

"""
System D: Contrastive decoding soft unlearning.

Runs M_full (shared fine-tune) and M_base (base model) simultaneously.
At each decode step:
    contrast = logits_full - alpha * (logits_full - logits_base)
    contrast[target_tokens] -= 5.0

This approximately "subtracts out" the target user's influence.
Expected to fail: ~25-40% Truth Ratio residual on forget01 author.

Fallback: if dual-model OOM, set use_base_model=False to use token-penalty-only mode.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

DATA_DIR = Path("data")
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


class ContrastiveDecoder:
    def __init__(
        self,
        model_full,
        model_base,
        tokenizer,
        alpha: float = 0.3,
        token_penalty: float = 5.0,
        use_base_model: bool = True,
    ):
        self.full = model_full
        self.base = model_base  # may be None if use_base_model=False
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.token_penalty = token_penalty
        self.use_base_model = use_base_model
        self.target_tokens: set[int] = set()

    def set_target_user(self, author_id: str):
        char_token_path = DATA_DIR / "characteristic_tokens.json"
        with open(char_token_path) as f:
            char_tokens = json.load(f)
        self.target_tokens = set(char_tokens.get(author_id, []))
        print(f"System D: targeting {author_id} with {len(self.target_tokens)} characteristic tokens")

    def clear_target(self):
        self.target_tokens = set()

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.full.device)
        generated = inputs.input_ids

        for _ in range(max_new_tokens):
            with torch.no_grad():
                full_logits = self.full(generated).logits[:, -1, :]

                if self.use_base_model and self.base is not None:
                    # Move base model inputs to base model's device
                    base_input = generated.to(self.base.device)
                    base_logits = self.base(base_input).logits[:, -1, :].to(full_logits.device)
                    contrast = full_logits - self.alpha * (full_logits - base_logits)
                else:
                    # Token-penalty-only fallback (no dual-model forward pass)
                    contrast = full_logits.clone()

                # Hard penalty on characteristic tokens of the target user
                if self.target_tokens:
                    penalty_ids = torch.tensor(
                        [t for t in self.target_tokens if t < contrast.shape[-1]],
                        device=contrast.device,
                    )
                    if penalty_ids.numel() > 0:
                        contrast[0, penalty_ids] -= self.token_penalty

            next_token = torch.argmax(contrast, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(
            generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )


def load_contrastive_decoder(
    shared_adapter_dir: str,
    alpha: float = 0.3,
    token_penalty: float = 5.0,
) -> ContrastiveDecoder:
    """
    Load the shared fine-tune (M_full) and base model (M_base).
    If both can't fit in VRAM, M_base falls back to CPU.
    """
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    print("Loading M_full (shared fine-tune)...")
    tokenizer = AutoTokenizer.from_pretrained(shared_adapter_dir)
    base_for_full = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, quantization_config=bnb, device_map="auto"
    )
    model_full = PeftModel.from_pretrained(base_for_full, shared_adapter_dir)
    model_full.eval()

    print("Loading M_base (base model)...")
    use_base_model = True
    try:
        model_base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME, quantization_config=bnb, device_map="auto"
        )
        model_base.eval()
        print("  M_base loaded on GPU")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("  GPU OOM for M_base — loading on CPU (slower but functional)")
            model_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map="cpu")
            model_base.eval()
        else:
            raise

    return ContrastiveDecoder(
        model_full=model_full,
        model_base=model_base,
        tokenizer=tokenizer,
        alpha=alpha,
        token_penalty=token_penalty,
        use_base_model=use_base_model,
    )

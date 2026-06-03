import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from config_utils import load_config


def get_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Gera texto com config YAML.")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    generation_cfg = config.get("generation", {})

    seed = generation_cfg.get("seed")
    if seed is None:
        seed = random.randint(1, 10_000_000)

    set_seed(seed)
    device = get_device(generation_cfg.get("device", "auto"))
    model_dir = paths["output_dir"]
    prompt = generation_cfg.get(
        "prompt",
        "A inteligência artificial aplicada à engenharia de dados",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=generation_cfg.get("max_new_tokens", 200),
            do_sample=True,
            temperature=generation_cfg.get("temperature", 0.8),
            top_p=generation_cfg.get("top_p", 0.95),
            top_k=generation_cfg.get("top_k", 50),
            repetition_penalty=generation_cfg.get("repetition_penalty", 1.1),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("=" * 80)
    print("CONFIG")
    print("=" * 80)
    print(config["name"])

    print("\n" + "=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(prompt)

    print("\n" + "=" * 80)
    print("GERACAO")
    print("=" * 80)
    print(generated)

    print("\n" + "=" * 80)
    print("INFO")
    print("=" * 80)
    print(f"Modelo: {model_dir}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")


if __name__ == "__main__":
    main()

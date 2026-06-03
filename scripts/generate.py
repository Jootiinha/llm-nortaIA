import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def get_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera texto com o modelo treinado."
    )

    parser.add_argument(
        "--model-dir",
        default="checkpoints/joao-gpt-mini",
        help="Diretório do checkpoint treinado.",
    )

    parser.add_argument(
        "--prompt",
        default="A inteligência artificial aplicada à engenharia de dados",
        help="Prompt inicial.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Quantidade máxima de tokens novos.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperatura da geração.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling.",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalidade para repetição.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed opcional.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cpu ou mps.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is None:
        args.seed = random.randint(1, 10_000_000)

    set_seed(args.seed)

    device = get_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(args.prompt)

    print("\n" + "=" * 80)
    print("GERAÇÃO")
    print("=" * 80)
    print(generated)

    print("\n" + "=" * 80)
    print("INFO")
    print("=" * 80)
    print(f"Modelo: {args.model_dir}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()


#python scripts/generate.py \
#   --prompt "Explique de forma simples o que é um modelo de linguagem" \
#   --max-new-tokens 150
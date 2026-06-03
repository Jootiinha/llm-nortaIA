import argparse
from pathlib import Path
from statistics import mean

from transformers import GPT2TokenizerFast

from config_utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Avalia tokenizer via config YAML.")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    tokenizer_dir = config["paths"]["tokenizer_dir"]
    eval_file = Path(config["paths"]["eval_file"])
    max_samples = config.get("tokenizer_eval", {}).get("max_samples", 1000)

    tokenizer = GPT2TokenizerFast(
        vocab_file=f"{tokenizer_dir}/vocab.json",
        merges_file=f"{tokenizer_dir}/merges.txt",
        tokenizer_file=f"{tokenizer_dir}/tokenizer.json",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    texts = []
    with eval_file.open("r", encoding="utf-8") as file:
        buffer = []

        for line in file:
            line = line.strip()

            if not line:
                if buffer:
                    texts.append(" ".join(buffer))
                    buffer = []
                if len(texts) >= max_samples:
                    break
                continue

            buffer.append(line)

    char_per_token = []
    tokens_per_word = []

    for text in texts:
        ids = tokenizer.encode(text)
        if not ids:
            continue

        char_per_token.append(len(text) / len(ids))
        words = text.split()
        if words:
            tokens_per_word.append(len(ids) / len(words))

    print(f"Config: {config['name']}")
    print("Amostras:", len(texts))
    print("Média chars/token:", round(mean(char_per_token), 2))
    print("Média tokens/palavra:", round(mean(tokens_per_word), 2))
    print("Vocab size:", len(tokenizer))


if __name__ == "__main__":
    main()

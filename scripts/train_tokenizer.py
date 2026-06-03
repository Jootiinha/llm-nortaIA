import argparse
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

from config_utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Treina tokenizer a partir de config YAML.")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    tokenizer_cfg = config.get("tokenizer", {})

    tokenizer_corpus_dir = Path(paths["tokenizer_corpus_dir"])
    tokenizer_output_dir = Path(paths["tokenizer_dir"])
    tokenizer_output_dir.mkdir(parents=True, exist_ok=True)

    files = [str(path) for path in tokenizer_corpus_dir.glob("*.txt")]
    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo .txt encontrado em {tokenizer_corpus_dir}"
        )

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=tokenizer_cfg.get("vocab_size", 32000),
        min_frequency=tokenizer_cfg.get("min_frequency", 2),
        special_tokens=tokenizer_cfg.get(
            "special_tokens",
            ["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        ),
        show_progress=True,
    )

    tokenizer.save_model(str(tokenizer_output_dir))
    tokenizer.save(str(tokenizer_output_dir / "tokenizer.json"))

    print("Tokenizer treinado com sucesso.")
    print(f"Config: {config['name']}")
    print(f"Arquivos salvos em: {tokenizer_output_dir}")


if __name__ == "__main__":
    main()

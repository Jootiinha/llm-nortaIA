import argparse
import random
from pathlib import Path

from config_utils import load_config


def read_blocks_from_file(file_path: Path, min_chars: int):
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    raw_blocks = text.split("\n\n")
    blocks = []

    for block in raw_blocks:
        block = block.strip()
        if len(block) >= min_chars:
            blocks.append(block)

    return blocks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepara arquivos de treino e validacao a partir de config YAML."
    )
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    prep_cfg = config.get("dataset_prep", {})

    input_dir = Path(paths["tokenizer_corpus_dir"])
    train_file = Path(paths["train_file"])
    valid_file = Path(paths["valid_file"])
    valid_ratio = prep_cfg.get("valid_ratio", 0.01)
    min_chars = prep_cfg.get("min_chars", 200)
    seed = prep_cfg.get("seed", 42)

    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {input_dir}")

    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Nenhum arquivo .txt encontrado em: {input_dir}")

    print(f"Config: {config['name']}")
    print("Lendo arquivos:")

    all_blocks = []
    for file_path in txt_files:
        blocks = read_blocks_from_file(file_path=file_path, min_chars=min_chars)
        all_blocks.extend(blocks)
        print(f"- {file_path}: {len(blocks)} blocos válidos")

    if len(all_blocks) < 10:
        raise RuntimeError(
            "Poucos blocos encontrados. Reduza min_chars ou verifique os arquivos."
        )

    random.seed(seed)
    random.shuffle(all_blocks)

    valid_size = max(1, int(len(all_blocks) * valid_ratio))
    valid_blocks = all_blocks[:valid_size]
    train_blocks = all_blocks[valid_size:]

    train_file.parent.mkdir(parents=True, exist_ok=True)
    valid_file.parent.mkdir(parents=True, exist_ok=True)

    train_file.write_text("\n\n".join(train_blocks), encoding="utf-8")
    valid_file.write_text("\n\n".join(valid_blocks), encoding="utf-8")

    print("\nArquivos gerados com sucesso:")
    print(f"- Treino: {train_file}")
    print(f"- Validação: {valid_file}")
    print("\nResumo:")
    print(f"- Total de blocos: {len(all_blocks)}")
    print(f"- Blocos de treino: {len(train_blocks)}")
    print(f"- Blocos de validação: {len(valid_blocks)}")
    print(f"- Percentual validação: {valid_ratio * 100:.2f}%")


if __name__ == "__main__":
    main()

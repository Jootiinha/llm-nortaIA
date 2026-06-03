from pathlib import Path
import argparse
import random


def read_blocks_from_file(file_path: Path, min_chars: int):
    """
    Lê um arquivo .txt e separa em blocos usando linha em branco como separador.
    Cada bloco vira uma amostra textual para treino/validação.
    """
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    raw_blocks = text.split("\n\n")

    blocks = []

    for block in raw_blocks:
        block = block.strip()

        if len(block) < min_chars:
            continue

        blocks.append(block)

    return blocks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepara arquivos data/train.txt e data/valid.txt."
    )

    parser.add_argument(
        "--input-dir",
        default="data/tokenizer_corpus",
        help="Diretório com os arquivos .txt limpos.",
    )

    parser.add_argument(
        "--train-file",
        default="data/train.txt",
        help="Arquivo de saída para treino.",
    )

    parser.add_argument(
        "--valid-file",
        default="data/valid.txt",
        help="Arquivo de saída para validação.",
    )

    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.01,
        help="Percentual dos dados usado para validação. Ex: 0.01 = 1%.",
    )

    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Tamanho mínimo de cada bloco de texto.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para embaralhamento.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    train_file = Path(args.train_file)
    valid_file = Path(args.valid_file)

    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {input_dir}")

    txt_files = list(input_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"Nenhum arquivo .txt encontrado em: {input_dir}"
        )

    print("Lendo arquivos:")

    all_blocks = []

    for file_path in txt_files:
        blocks = read_blocks_from_file(
            file_path=file_path,
            min_chars=args.min_chars,
        )

        all_blocks.extend(blocks)

        print(f"- {file_path}: {len(blocks)} blocos válidos")

    if len(all_blocks) < 10:
        raise RuntimeError(
            "Poucos blocos encontrados. Reduza --min-chars ou verifique os arquivos de entrada."
        )

    random.seed(args.seed)
    random.shuffle(all_blocks)

    valid_size = int(len(all_blocks) * args.valid_ratio)
    valid_size = max(1, valid_size)

    valid_blocks = all_blocks[:valid_size]
    train_blocks = all_blocks[valid_size:]

    train_file.parent.mkdir(parents=True, exist_ok=True)
    valid_file.parent.mkdir(parents=True, exist_ok=True)

    train_file.write_text(
        "\n\n".join(train_blocks),
        encoding="utf-8",
    )

    valid_file.write_text(
        "\n\n".join(valid_blocks),
        encoding="utf-8",
    )

    print("\nArquivos gerados com sucesso:")
    print(f"- Treino: {train_file}")
    print(f"- Validação: {valid_file}")

    print("\nResumo:")
    print(f"- Total de blocos: {len(all_blocks)}")
    print(f"- Blocos de treino: {len(train_blocks)}")
    print(f"- Blocos de validação: {len(valid_blocks)}")
    print(f"- Percentual validação: {args.valid_ratio * 100:.2f}%")

if __name__ == "__main__":
    main()
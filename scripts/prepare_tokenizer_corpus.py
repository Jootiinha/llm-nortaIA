import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from config_utils import load_config

load_dotenv(dotenv_path=Path(".env"))
HF_TOKEN = os.getenv("HF_TOKEN")


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = "\n".join(line for line in text.splitlines() if line)
    return text.strip()


def write_stream_sample(
    output_dir: Path,
    dataset_name: str,
    split: str,
    output_name: str,
    max_bytes: int,
    description: str,
    min_text_chars: int,
    min_score: float | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        token=HF_TOKEN,
    )

    output_path = output_dir / output_name
    written = 0
    iterator = iter(dataset)

    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            for row in tqdm(iterator, desc=description):
                text = row.get("text", "")
                score = row.get("score")

                if not text:
                    continue

                if min_score is not None and score is not None and score < min_score:
                    continue

                text = clean_text(text)
                if len(text) < min_text_chars:
                    continue

                block = text + "\n\n"
                encoded = block.encode("utf-8")

                if written + len(encoded) > max_bytes:
                    break

                output_file.write(block)
                written += len(encoded)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    print(f"{description} salvo: {written / 1024 / 1024:.2f} MB")


def parse_args():
    parser = argparse.ArgumentParser(description="Baixa corpus via config YAML.")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    corpus_cfg = config.get("corpus", {})
    output_dir = Path(paths["tokenizer_corpus_dir"])
    min_text_chars = corpus_cfg.get("min_text_chars", 200)

    print(f"Config: {config['name']}")
    write_stream_sample(
        output_dir=output_dir,
        dataset_name=corpus_cfg.get("gigaverbo_dataset", "TucanoBR/GigaVerbo-Text-Filter"),
        split=corpus_cfg.get("gigaverbo_split", "train"),
        output_name=corpus_cfg.get(
            "gigaverbo_output_file",
            "gigaverbo_filter_high_quality.txt",
        ),
        max_bytes=corpus_cfg.get("gigaverbo_max_mb", 200) * 1024 * 1024,
        description="GigaVerbo",
        min_text_chars=min_text_chars,
        min_score=corpus_cfg.get("gigaverbo_min_score", 0.75),
    )
    write_stream_sample(
        output_dir=output_dir,
        dataset_name=corpus_cfg.get("pt_dataset", "nicholasKluge/Pt-Corpus-Instruct"),
        split=corpus_cfg.get("pt_split", "train"),
        output_name=corpus_cfg.get(
            "pt_output_file",
            "pt_corpus_instruct_sample.txt",
        ),
        max_bytes=corpus_cfg.get("pt_max_mb", 800) * 1024 * 1024,
        description="Pt-Corpus",
        min_text_chars=min_text_chars,
    )
    print("Download concluido.")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

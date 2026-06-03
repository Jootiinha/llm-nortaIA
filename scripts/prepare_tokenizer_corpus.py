import os
import sys
import warnings
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(dotenv_path=Path(".env"))

OUTPUT_DIR = Path("./data/tokenizer_corpus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_BYTES_PT_CORPUS = 800 * 1024 * 1024  # 800MB
MAX_BYTES_GIGAVERBO = 200 * 1024 * 1024  # 200MB
HF_TOKEN = os.getenv("HF_TOKEN")


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = "\n".join(line for line in text.splitlines() if line)
    return text.strip()


def write_stream_sample(
    dataset_name: str,
    output_name: str,
    max_bytes: int,
    description: str,
    min_score: float | None = None,
) -> None:
    dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    output_path = OUTPUT_DIR / output_name
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

                if len(text) < 200:
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


def main() -> None:
    write_stream_sample(
        dataset_name="TucanoBR/GigaVerbo-Text-Filter",
        output_name="gigaverbo_filter_high_quality.txt",
        max_bytes=MAX_BYTES_GIGAVERBO,
        description="GigaVerbo",
        min_score=0.75,
    )
    write_stream_sample(
        dataset_name="nicholasKluge/Pt-Corpus-Instruct",
        output_name="pt_corpus_instruct_sample.txt",
        max_bytes=MAX_BYTES_PT_CORPUS,
        description="Pt-Corpus",
    )
    print("Download concluido.")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

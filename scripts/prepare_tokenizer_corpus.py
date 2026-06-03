from pathlib import Path
import os

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

OUTPUT_DIR = Path("./data/tokenizer_corpus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_BYTES_PT_CORPUS = 800 * 1024 * 1024 # 800MB
MAX_BYTES_GIGAVERBO = 200 * 1024 * 1024 # 200MB
HF_TOKEN = os.getenv("HF_TOKEN")

def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = "\n".join(line for line in text.splitlines() if line)
    return text.strip()

def write_pt_corpus():
    dataset = load_dataset(
        "nicholasKluge/Pt-Corpus-Instruct",
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    output_path = OUTPUT_DIR / "pt_corpus_instruct_sample.txt"
    written = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row in tqdm(dataset, desc="Pt-Corpus-Instruct"):
            text = row.get("text", "")

            if not text:
                continue

            text = clean_text(text)

            if len(text) < 200:
                continue

            block = text + "\n\n"
            encoded = block.encode("utf-8")

            if written + len(encoded) > MAX_BYTES_PT_CORPUS:
                break

            f.write(block)
            written += len(encoded)

    print(f"Pt-Corpus salvo: {written / 1024 / 1024:.2f} MB")


def write_gigaverbo_filter():
    dataset = load_dataset(
        "TucanoBR/GigaVerbo-Text-Filter",
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    output_path = OUTPUT_DIR / "gigaverbo_filter_high_quality.txt"
    written = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row in tqdm(dataset, desc="GigaVerbo-Text-Filter"):
            text = row.get("text", "")
            score = row.get("score", 0)

            if not text:
                continue

            if score is not None and score < 0.75:
                continue

            text = clean_text(text)

            if len(text) < 200:
                continue

            block = text + "\n\n"
            encoded = block.encode("utf-8")

            if written + len(encoded) > MAX_BYTES_GIGAVERBO:
                break

            f.write(block)
            written += len(encoded)

    print(f"GigaVerbo salvo: {written / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    write_gigaverbo_filter()
    write_pt_corpus()

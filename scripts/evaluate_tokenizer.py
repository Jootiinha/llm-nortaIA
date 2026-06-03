from pathlib import Path
from statistics import mean
from transformers import GPT2TokenizerFast

TOKENIZER_DIR = "tokenizer/joao-tokenizer-bpe"
EVAL_FILE = Path("data/tokenizer_corpus/pt_corpus_instruct_sample.txt")

tokenizer = GPT2TokenizerFast(
    vocab_file=f"{TOKENIZER_DIR}/vocab.json",
    merges_file=f"{TOKENIZER_DIR}/merges.txt",
    tokenizer_file=f"{TOKENIZER_DIR}/tokenizer.json",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)

texts = []
with EVAL_FILE.open("r", encoding="utf-8") as f:
    buffer = []

    for line in f:
        line = line.strip()

        if not line:
            if buffer:
                texts.append(" ".join(buffer))
                buffer = []
            continue

        buffer.append(line)

        if len(texts) >= 1000:
            break

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

print("Amostras:", len(texts))
print("Média chars/token:", round(mean(char_per_token), 2))
print("Média tokens/palavra:", round(mean(tokens_per_word), 2))
print("Vocab size:", len(tokenizer))
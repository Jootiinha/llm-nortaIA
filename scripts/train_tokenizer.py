from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

TOKENIZER_CORPUS_DIR = Path("data/tokenizer_corpus")
TOKENIZER_OUTPUT_DIR = Path("tokenizer/joao-tokenizer-bpe")
TOKENIZER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = [str(path) for path in TOKENIZER_CORPUS_DIR.glob("*.txt")]

if not files:
    raise FileNotFoundError(
        f"Nenhum arquivo .txt encontrado em {TOKENIZER_CORPUS_DIR}"
    )

SPECIAL_TOKENS = [
    "<s>",      # início de sequência
    "<pad>",    # padding
    "</s>",     # fim de sequência
    "<unk>",    # token desconhecido
    "<mask>",   # opcional, mas útil para compatibilidade
]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=files,
    vocab_size=32_000,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True,
)

tokenizer.save_model(str(TOKENIZER_OUTPUT_DIR))
tokenizer.save(str(TOKENIZER_OUTPUT_DIR / "tokenizer.json"))

print("Tokenizer treinado com sucesso.")
print(f"Arquivos salvos em: {TOKENIZER_OUTPUT_DIR}")
print("Arquivos esperados:")
print("- vocab.json")
print("- merges.txt")
print("- tokenizer.json")
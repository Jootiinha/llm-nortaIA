import argparse

from transformers import GPT2TokenizerFast

from config_utils import load_config


DEFAULT_TESTS = [
    "Olá, João! Tudo bem?",
    "A inteligência artificial aplicada à engenharia de dados pode otimizar pipelines.",
    "SELECT * FROM tabela WHERE created_at >= '2026-01-01';",
    "kubectl get pods -n ifp-ona-merchant",
    "Ação de guarda com pedido de tutela provisória.",
    "R$ 1.234,56 — São Paulo/SP — parâmetro, função, execução.",
    '{"merchant_id": "123", "status": "APPROVED", "amount": 159.90}',
]


def parse_args():
    parser = argparse.ArgumentParser(description="Testa tokenizer via config YAML.")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    tokenizer_dir = config["paths"]["tokenizer_dir"]
    tests = config.get("tokenizer_test", {}).get("samples", DEFAULT_TESTS)

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

    print(f"Config: {config['name']}")

    for text in tests:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        print("=" * 80)
        print("Texto original:")
        print(text)
        print()
        print("Tokens:")
        print(tokenizer.convert_ids_to_tokens(encoded))
        print()
        print("IDs:")
        print(encoded)
        print()
        print("Decoded:")
        print(decoded)
        print()
        print("Qtd tokens:", len(encoded))
        print("Chars/token:", round(len(text) / max(len(encoded), 1), 2))


if __name__ == "__main__":
    main()

Projeto LLM-NortaIA
====================

Rascunho de scaffold para treinar um LLM em Português usando:
- nicholasKluge/Pt-Corpus-Instruct (fine-tune/instruction)
- TucanoBR/GigaVerbo-Text-Filter (pré-treino)

Quickstart
----------

1. Instalar dependencias com Poetry

```bash
poetry install --no-root
```

2. Entrar no ambiente virtual do Poetry

```bash
poetry shell
```

3. Configurar acesso ao Hugging Face Hub

Crie um arquivo `.env` na raiz do projeto:

```bash
HF_TOKEN=hf_seu_token_aqui
```

O script `scripts/prepare_tokenizer_corpus.py` carrega esse valor automaticamente. Sem token, os downloads ainda funcionam, mas com rate limit menor e o warning sobre requests nao autenticadas.

4. Baixar o corpus usado para tokenizer

```bash
make download_corpus
```

5. Preparar datasets manuais em `data/raw/`

O repositório ainda nao contem um script de download. Coloque os arquivos brutos em `data/raw/` antes de seguir.

6. Pré-processar e treinar tokenizer

```bash
poetry run python src/preprocess.py --input-dir data/raw --out-dir data/processed
poetry run python src/tokenizer_train.py --input data/processed/combined.txt --output models/tokenizer
```

7. Treino (pré-treino / fine-tune)

```bash
poetry run python src/train_pretrain.py --config configs/pretrain_config.yaml
poetry run python src/train_finetune.py --config configs/finetune_config.yaml
```

Estrutura
---------

- `data/` — datasets brutos e processados
- `models/` — tokenizer e checkpoints
- `src/` — scripts e pipelines
- `configs/` — configurações de treino
- `scripts/` — utilitários auxiliares

Licença e notas
---------------

Verificar licenças dos datasets antes do uso em produção.

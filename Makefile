help:
	@echo "make install         # instala dependencias com poetry"
	@echo "make shell           # abre shell no ambiente do poetry"
	@echo "make download_corpus # baixa corpus do Hugging Face Hub"
	@echo "make preprocess      # roda preprocessamento"
	@echo "make train-tokenizer # treina tokenizer"
	@echo "make pretrain        # executa pipeline de pre-treino"
	@echo "make finetune        # executa pipeline de fine-tune"

install:
	poetry install --no-root

shell:
	poetry shell

download_corpus:
	poetry run python -B scripts/prepare_tokenizer_corpus.py

export PYTHONDONTWRITEBYTECODE := 1

help:
	@echo "make install         # instala dependencias com poetry"
	@echo "make shell           # abre shell no ambiente do poetry"
	@echo "make download_corpus # baixa corpus do Hugging Face Hub"
	@echo "make preprocess      # roda preprocessamento"
	@echo "make train-tokenizer # treina tokenizer"
	@echo "make tensorboard     # abre tensorboard em checkpoints/"
	@echo "make pretrain        # executa pipeline de pre-treino"
	@echo "make finetune        # executa pipeline de fine-tune"

install:
	poetry install --no-root

shell:
	poetry shell

download_corpus:
	poetry run python -B scripts/prepare_tokenizer_corpus.py

train_tokenizer:
	poetry run python -B scripts/train_tokenizer.py

test_tokenizer:
	poetry run python -B scripts/test_tokenizer.py

evaluate_tokenizer:
	poetry run python -B scripts/evaluate_tokenizer.py

prepare_train_valid:
	poetry run python -B scripts/prepare_train_valid.py

train_model:
	poetry run python -B scripts/train_model.py --max-steps 500 \
		--eval-steps 100 \
		--save-steps 250 \
		--n-layer 4 \
		--n-head 4 \
		--n-embd 256 \
		--batch-size 2

tensorboard:
	poetry run tensorboard --logdir checkpoints --port 6006

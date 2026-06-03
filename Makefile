export PYTHONDONTWRITEBYTECODE := 1
model_name ?= joao-gpt-mini-v1
CONFIG_FILE = configs/models/$(model_name).yaml

help:
	@echo "make install         # instala dependencias com poetry"
	@echo "make shell           # abre shell no ambiente do poetry"
	@echo "make download_corpus model_name=joao-gpt-mini-v1"
	@echo "make prepare_train_valid model_name=joao-gpt-mini-v1"
	@echo "make train_tokenizer model_name=joao-gpt-mini-v1"
	@echo "make test_tokenizer model_name=joao-gpt-mini-v1"
	@echo "make evaluate_tokenizer model_name=joao-gpt-mini-v1"
	@echo "make train_model model_name=joao-gpt-mini-v1"
	@echo "make generate model_name=joao-gpt-mini-v1"
	@echo "make tensorboard     # abre tensorboard em checkpoints/"
	@echo "config atual: $(CONFIG_FILE)"

install:
	poetry install --no-root

shell:
	poetry shell

download_corpus:
	poetry run python -B scripts/prepare_tokenizer_corpus.py --config $(CONFIG_FILE)

train_tokenizer:
	poetry run python -B scripts/train_tokenizer.py --config $(CONFIG_FILE)

test_tokenizer:
	poetry run python -B scripts/test_tokenizer.py --config $(CONFIG_FILE)

evaluate_tokenizer:
	poetry run python -B scripts/evaluate_tokenizer.py --config $(CONFIG_FILE)

prepare_train_valid:
	poetry run python -B scripts/prepare_train_valid.py --config $(CONFIG_FILE)

train_model:
	poetry run python -B scripts/train_model.py --config $(CONFIG_FILE)

generate:
	poetry run python -B scripts/generate.py --config $(CONFIG_FILE)

tensorboard:
	poetry run tensorboard --logdir checkpoints --port 6006

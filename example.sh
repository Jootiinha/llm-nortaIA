python scripts/prepare_tokenizer_corpus.py --pt-mb 100 --giga-mb 25

python scripts/train_tokenizer.py

python scripts/test_tokenizer.py

python scripts/evaluate_tokenizer.py

python scripts/prepare_train_valid.py

python scripts/train_model.py \
  --max-steps 500 \
  --eval-steps 100 \
  --save-steps 250 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 256 \
  --batch-size 2

python scripts/generate.py \
  --prompt "A inteligência artificial no Brasil"
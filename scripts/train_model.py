from pathlib import Path
from typing import Dict, Any
import argparse
import inspect
import math

import torch
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)


def load_tokenizer(tokenizer_dir: str) -> GPT2TokenizerFast:
    tokenizer_path = Path(tokenizer_dir)

    tokenizer_json = tokenizer_path / "tokenizer.json"
    vocab_json = tokenizer_path / "vocab.json"
    merges_txt = tokenizer_path / "merges.txt"

    if tokenizer_json.exists() and vocab_json.exists() and merges_txt.exists():
        tokenizer = GPT2TokenizerFast(
            tokenizer_file=str(tokenizer_json),
            vocab_file=str(vocab_json),
            merges_file=str(merges_txt),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
        )
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    return tokenizer


def build_training_arguments(args) -> TrainingArguments:
    """
    Compatibilidade com versões diferentes do transformers:
    algumas usam eval_strategy, outras evaluation_strategy.
    """
    kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "prediction_loss_only": True,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "save_safetensors": True,
    }

    signature = inspect.signature(TrainingArguments.__init__)

    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    if torch.cuda.is_available():
        if args.bf16 and torch.cuda.is_bf16_supported():
            kwargs["bf16"] = True
        elif args.fp16:
            kwargs["fp16"] = True

    return TrainingArguments(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina um modelo GPT decoder-only do zero."
    )

    parser.add_argument("--tokenizer-dir", default="tokenizer/joao-tokenizer-bpe")
    parser.add_argument("--train-file", default="data/train.txt")
    parser.add_argument("--valid-file", default="data/valid.txt")
    parser.add_argument("--output-dir", default="checkpoints/joao-gpt-mini")

    parser.add_argument("--block-size", type=int, default=512)

    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=20_000)

    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=3)

    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    train_path = Path(args.train_file)
    valid_path = Path(args.valid_file)

    if not train_path.exists():
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {train_path}")

    if not valid_path.exists():
        raise FileNotFoundError(f"Arquivo de validação não encontrado: {valid_path}")

    tokenizer = load_tokenizer(args.tokenizer_dir)

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))

    num_params = sum(p.numel() for p in model.parameters())
    print("=" * 80)
    print("CONFIGURAÇÃO DO MODELO")
    print("=" * 80)
    print(f"Parâmetros: {num_params / 1_000_000:.2f}M")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Contexto: {args.block_size}")
    print(f"Camadas: {args.n_layer}")
    print(f"Heads: {args.n_head}")
    print(f"Embedding dim: {args.n_embd}")

    dataset = load_dataset(
        "text",
        data_files={
            "train": str(train_path),
            "validation": str(valid_path),
        },
    )

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    map_kwargs = {
        "batched": True,
        "remove_columns": ["text"],
        "desc": "Tokenizando dataset",
    }

    if args.num_proc and args.num_proc > 1:
        map_kwargs["num_proc"] = args.num_proc

    tokenized = dataset.map(tokenize_function, **map_kwargs)

    block_size = args.block_size
    eos_token_id = tokenizer.eos_token_id

    def group_texts(examples):
        all_ids = []

        for ids in examples["input_ids"]:
            if ids:
                all_ids.extend(ids)
                all_ids.append(eos_token_id)

        total_length = len(all_ids)

        if total_length < block_size:
            return {"input_ids": [], "labels": []}

        total_length = (total_length // block_size) * block_size

        chunks = [
            all_ids[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]

        return {
            "input_ids": chunks,
            "labels": [chunk.copy() for chunk in chunks],
        }

    group_kwargs = {
        "batched": True,
        "desc": "Agrupando em blocos fixos",
    }

    if args.num_proc and args.num_proc > 1:
        group_kwargs["num_proc"] = args.num_proc

    lm_dataset = tokenized.map(group_texts, **group_kwargs)

    training_args = build_training_arguments(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
    )

    print("\nIniciando treinamento...")
    trainer.train()

    print("\nAvaliando...")
    eval_result = trainer.evaluate()

    loss = eval_result.get("eval_loss")
    if loss is not None:
        try:
            perplexity = math.exp(loss)
        except OverflowError:
            perplexity = float("inf")

        print(f"Eval loss: {loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")

    print("\nSalvando modelo final...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Modelo salvo em: {args.output_dir}")
    print("Próximo passo:")
    print(f"python scripts/generate.py --model-dir {args.output_dir}")


if __name__ == "__main__":
    main()
    # python scripts/train_model.py --bf16
    # python scripts/train_model.py \
    #     --max-steps 500 \
    #     --eval-steps 100 \
    #     --save-steps 250 \
    #     --n-layer 4 \
    #     --n-head 4 \
    #     --n-embd 256 \
    #     --batch-size 2
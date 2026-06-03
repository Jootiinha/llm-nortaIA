from pathlib import Path
from time import perf_counter
from typing import Any, Dict
import argparse
import inspect
import math
import os

import psutil
import torch
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from config_utils import load_config


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


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


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
        "report_to": "tensorboard",
        "remove_unused_columns": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "save_safetensors": True,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    valid_parameters = set(signature.parameters)

    if "eval_strategy" in valid_parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    if "logging_strategy" in valid_parameters:
        kwargs["logging_strategy"] = "steps"

    if detect_device() == "mps":
        kwargs["dataloader_pin_memory"] = False

    if torch.cuda.is_available():
        if args.bf16 and torch.cuda.is_bf16_supported():
            kwargs["bf16"] = True
        elif args.fp16:
            kwargs["fp16"] = True

    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in valid_parameters
    }

    return TrainingArguments(**filtered_kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Treina um modelo GPT via config YAML.")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def build_args_from_config(config: dict[str, Any]) -> argparse.Namespace:
    paths = config["paths"]
    train_cfg = config.get("train", {})
    return argparse.Namespace(
        config_name=config["name"],
        tokenizer_dir=paths["tokenizer_dir"],
        train_file=paths["train_file"],
        valid_file=paths["valid_file"],
        output_dir=paths["output_dir"],
        block_size=train_cfg.get("block_size", 512),
        n_layer=train_cfg.get("n_layer", 6),
        n_head=train_cfg.get("n_head", 6),
        n_embd=train_cfg.get("n_embd", 384),
        batch_size=train_cfg.get("batch_size", 4),
        eval_batch_size=train_cfg.get("eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 0.1),
        warmup_steps=train_cfg.get("warmup_steps", 500),
        max_steps=train_cfg.get("max_steps", 20_000),
        logging_steps=train_cfg.get("logging_steps", 50),
        eval_steps=train_cfg.get("eval_steps", 500),
        save_steps=train_cfg.get("save_steps", 1000),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        num_proc=train_cfg.get("num_proc", 1),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        seed=train_cfg.get("seed", 42),
        bf16=train_cfg.get("bf16", False),
        fp16=train_cfg.get("fp16", False),
    )


class PerformanceCallback(TrainerCallback):
    def __init__(self, writer: SummaryWriter, args, prep_metrics: Dict[str, float]):
        self.writer = writer
        self.process = psutil.Process(os.getpid())
        self.device = detect_device()
        self.tokens_per_step = (
            args.batch_size * args.gradient_accumulation_steps * args.block_size
        )
        self.examples_per_step = args.batch_size * args.gradient_accumulation_steps
        self.step_start_time = None
        self.prev_log_time = None
        self.prev_log_step = 0
        self.prep_metrics = prep_metrics

    def _gpu_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        if self.device == "cuda":
            metrics["system/gpu_memory_allocated_mb"] = (
                torch.cuda.memory_allocated() / 1024 / 1024
            )
            metrics["system/gpu_memory_reserved_mb"] = (
                torch.cuda.memory_reserved() / 1024 / 1024
            )
            metrics["system/gpu_memory_max_allocated_mb"] = (
                torch.cuda.max_memory_allocated() / 1024 / 1024
            )

        if self.device == "mps" and hasattr(torch, "mps"):
            if hasattr(torch.mps, "current_allocated_memory"):
                metrics["system/mps_current_allocated_mb"] = (
                    torch.mps.current_allocated_memory() / 1024 / 1024
                )
            if hasattr(torch.mps, "driver_allocated_memory"):
                metrics["system/mps_driver_allocated_mb"] = (
                    torch.mps.driver_allocated_memory() / 1024 / 1024
                )
            if hasattr(torch.mps, "recommended_max_memory"):
                metrics["system/mps_recommended_max_memory_mb"] = (
                    torch.mps.recommended_max_memory() / 1024 / 1024
                )

        return metrics

    def _system_metrics(self) -> Dict[str, float]:
        with self.process.oneshot():
            rss_mb = self.process.memory_info().rss / 1024 / 1024
            ram_percent = self.process.memory_percent()
            cpu_percent = self.process.cpu_percent(interval=None)
            thread_count = float(self.process.num_threads())

        metrics = {
            "system/process_rss_mb": rss_mb,
            "system/process_ram_percent": ram_percent,
            "system/process_cpu_percent": cpu_percent,
            "system/process_threads": thread_count,
        }
        metrics.update(self._gpu_metrics())
        return metrics

    def on_train_begin(self, args, state, control, **kwargs):
        self.process.cpu_percent(interval=None)
        self.prev_log_time = perf_counter()
        self.prev_log_step = 0

        for name, value in self.prep_metrics.items():
            self.writer.add_scalar(name, value, 0)

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        now = perf_counter()
        current_step = int(state.global_step)
        metrics = self._system_metrics()

        if self.step_start_time is not None:
            metrics["perf/step_time_sec"] = now - self.step_start_time

        if self.prev_log_time is not None and current_step > self.prev_log_step:
            window_time = now - self.prev_log_time
            window_steps = current_step - self.prev_log_step

            if window_time > 0:
                metrics["perf/steps_per_sec_window"] = window_steps / window_time
                metrics["perf/examples_per_sec_window"] = (
                    window_steps * self.examples_per_step
                ) / window_time
                metrics["perf/tokens_per_sec_window"] = (
                    window_steps * self.tokens_per_step
                ) / window_time

        metrics["perf/tokens_per_step"] = float(self.tokens_per_step)
        metrics["perf/examples_per_step"] = float(self.examples_per_step)

        for name, value in metrics.items():
            self.writer.add_scalar(name, value, current_step)

        self.prev_log_time = now
        self.prev_log_step = current_step

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_step = int(state.global_step)

        if metrics and "eval_loss" in metrics:
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")

            self.writer.add_scalar("eval/perplexity", perplexity, current_step)

        for name, value in self._system_metrics().items():
            self.writer.add_scalar(name, value, current_step)

    def on_train_end(self, args, state, control, **kwargs):
        for name, value in self._system_metrics().items():
            self.writer.add_scalar(name, value, int(state.global_step))

        self.writer.flush()


def main():
    cli_args = parse_args()
    experiment_config = load_config(cli_args.config)
    args = build_args_from_config(experiment_config)
    set_seed(args.seed)
    tensorboard_log_dir = str(Path(args.output_dir) / "tb")
    os.environ["TENSORBOARD_LOGGING_DIR"] = tensorboard_log_dir
    training_args = build_training_arguments(args)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    train_path = Path(args.train_file)
    valid_path = Path(args.valid_file)

    if not train_path.exists():
        raise FileNotFoundError(f"Arquivo de treino não encontrado: {train_path}")

    if not valid_path.exists():
        raise FileNotFoundError(f"Arquivo de validação não encontrado: {valid_path}")

    tokenizer = load_tokenizer(args.tokenizer_dir)
    device = detect_device()

    writer.add_scalar("run/status", 0.0, 0)
    writer.add_text("run/device", device, 0)
    writer.add_text("run/output_dir", args.output_dir, 0)
    writer.add_scalar("run/block_size", float(args.block_size), 0)
    writer.add_scalar("run/batch_size", float(args.batch_size), 0)
    writer.flush()

    model_config = GPT2Config(
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

    model = GPT2LMHeadModel(model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.loss_type = "ForCausalLM"
    model.loss_type = "ForCausalLM"

    num_params = sum(p.numel() for p in model.parameters())
    print("=" * 80)
    print("CONFIGURACAO DO MODELO")
    print("=" * 80)
    print(f"Config: {experiment_config['name']}")
    print(f"Parâmetros: {num_params / 1_000_000:.2f}M")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Contexto: {args.block_size}")
    print(f"Camadas: {args.n_layer}")
    print(f"Heads: {args.n_head}")
    print(f"Embedding dim: {args.n_embd}")
    print(f"Device: {device}")
    print(
        "Tokens/step teorico: "
        f"{args.batch_size * args.gradient_accumulation_steps * args.block_size}"
    )

    writer.add_scalar("model/num_parameters", float(num_params), 0)
    writer.add_scalar("model/vocab_size", float(len(tokenizer)), 0)
    writer.add_scalar("model/n_layer", float(args.n_layer), 0)
    writer.add_scalar("model/n_head", float(args.n_head), 0)
    writer.add_scalar("model/n_embd", float(args.n_embd), 0)
    writer.flush()

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

    tokenization_start = perf_counter()
    tokenized = dataset.map(tokenize_function, **map_kwargs)
    tokenization_seconds = perf_counter() - tokenization_start

    block_size = args.block_size
    eos_token_id = tokenizer.eos_token_id

    def group_texts(examples):
        all_input_ids = []
        all_attention_masks = []

        for ids, mask in zip(
            examples["input_ids"],
            examples.get("attention_mask", [[]] * len(examples["input_ids"])),
        ):
            if not ids:
                continue

            all_input_ids.extend(ids)
            all_input_ids.append(eos_token_id)

            if mask:
                all_attention_masks.extend(mask)
                all_attention_masks.append(1)

        total_length = len(all_input_ids)

        if total_length < block_size:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        total_length = (total_length // block_size) * block_size

        input_id_chunks = [
            all_input_ids[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        attention_mask_chunks = []

        if all_attention_masks:
            attention_mask_chunks = [
                all_attention_masks[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
        else:
            attention_mask_chunks = [
                [1] * block_size for _ in range(len(input_id_chunks))
            ]

        return {
            "input_ids": input_id_chunks,
            "attention_mask": attention_mask_chunks,
            "labels": [chunk.copy() for chunk in input_id_chunks],
        }

    group_kwargs = {
        "batched": True,
        "desc": "Agrupando em blocos fixos",
        "remove_columns": tokenized["train"].column_names,
    }

    if args.num_proc and args.num_proc > 1:
        group_kwargs["num_proc"] = args.num_proc

    grouping_start = perf_counter()
    lm_dataset = tokenized.map(group_texts, **group_kwargs)
    grouping_seconds = perf_counter() - grouping_start

    train_blocks = len(lm_dataset["train"])
    valid_blocks = len(lm_dataset["validation"])
    train_tokens = train_blocks * args.block_size
    valid_tokens = valid_blocks * args.block_size

    print(f"Tokenizacao: {tokenization_seconds:.2f}s")
    print(f"Agrupamento: {grouping_seconds:.2f}s")
    print(f"Blocos de treino: {train_blocks}")
    print(f"Blocos de validacao: {valid_blocks}")

    prep_metrics = {
        "prep/tokenization_seconds": tokenization_seconds,
        "prep/grouping_seconds": grouping_seconds,
        "prep/raw_train_examples": float(len(dataset["train"])),
        "prep/raw_validation_examples": float(len(dataset["validation"])),
        "prep/train_blocks": float(train_blocks),
        "prep/validation_blocks": float(valid_blocks),
        "prep/train_tokens": float(train_tokens),
        "prep/validation_tokens": float(valid_tokens),
        "model/num_parameters": float(num_params),
        "model/vocab_size": float(len(tokenizer)),
        "model/block_size": float(args.block_size),
        "model/n_layer": float(args.n_layer),
        "model/n_head": float(args.n_head),
        "model/n_embd": float(args.n_embd),
        "run/batch_size": float(args.batch_size),
        "run/eval_batch_size": float(args.eval_batch_size),
        "run/gradient_accumulation_steps": float(
            args.gradient_accumulation_steps
        ),
        "run/effective_examples_per_step": float(
            args.batch_size * args.gradient_accumulation_steps
        ),
        "run/effective_tokens_per_step": float(
            args.batch_size * args.gradient_accumulation_steps * args.block_size
        ),
    }

    writer.add_text("run/device", device, 0)
    writer.add_text("run/output_dir", args.output_dir, 0)
    writer.add_scalar("run/status", 1.0, 0)
    writer.flush()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        callbacks=[PerformanceCallback(writer, args, prep_metrics)],
    )

    print("\nIniciando treinamento...")
    train_result = trainer.train()

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

    metrics = train_result.metrics
    if metrics:
        print("\nRESUMO DE TREINO")
        print(f"Train runtime: {metrics.get('train_runtime', 0):.2f}s")
        print(f"Train steps/s: {metrics.get('train_steps_per_second', 0):.2f}")
        print(f"Train samples/s: {metrics.get('train_samples_per_second', 0):.2f}")

    print("\nSalvando modelo final...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"TensorBoard log dir: {tensorboard_log_dir}")
    print(f"Modelo salvo em: {args.output_dir}")
    print("Proximo passo:")
    print("1. make tensorboard")
    print(
        "2. poetry run python -B scripts/generate.py "
        f"--config {experiment_config['_config_path']}"
    )
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
